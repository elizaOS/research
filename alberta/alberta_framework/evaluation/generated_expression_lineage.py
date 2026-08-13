"""Host-only exact expanded-expression lineage compilation and validation.

This module reconstructs compositional feature trees exclusively through the
public canonical ``GeneratedExpression`` constructors.  Their tree grouping is
preserved exactly, including per-node clipping semantics; sum/product children
and tanh child/coefficient pairs receive the public canonicalization, while an
ordered gate remains ordered.  No associative flattening or numerical probe is
used as identity.

The compiler returns fixed-shape masks for every active expanded tree that
contains a canonical target, all transitive active descendants, and every
candidate that contains the target or depends on a masked active parent.  The
post-scrub validator independently recompiles the pre-scrub plan and rebuilds
every post-scrub tree before reporting target absence.

This is development-only host work.  It is not JIT-compatible, has no target-D
special case, does not claim behavioral-memory erasure, constructs no runner,
and grants no execution, campaign, artifact, evidence, or promotion authority.
Its claim is limited to exact canonical identities in the supplied current
trees; it does not cover alpha-renamed or broader algebraic equivalence, hidden
archives, or prevention of future reacquisition.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import struct
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from alberta_framework.core.compositional_features import (
    NUM_OPS,
    OP_GATED,
    OP_PRODUCT,
    OP_RAW,
    OP_SUM,
    OP_TANH,
    CompositionalFeatureState,
)
from alberta_framework.core.resource_manager import GeneratorMetaResourceManagerState
from alberta_framework.evaluation.generated_class_lifecycle_scrub import (
    COMPOSITIONAL_STATE_LEAF_PATHS,
    GeneratedClassScrubConfig,
    compositional_state_leaf_paths,
    persistent_compositional_state_nbytes,
    scrub_compositional_feature_state,
)
from alberta_framework.evaluation.generated_class_recurrence import (
    GeneratedExpression,
    expression_digest,
    gate_expression,
    product_expression,
    raw_expression,
    sum_expression,
    tanh_expression,
)

EXPANDED_EXPRESSION_LINEAGE_SCHEMA = "alberta.generated-expression-lineage.development.v0"
EXPANDED_EXPRESSION_LINEAGE_STATUS = "DEVELOPMENT_HOST_ONLY_NO_EXECUTION_OR_EVIDENCE_AUTHORITY"


@dataclasses.dataclass(frozen=True, slots=True)
class ExpandedExpressionLineageConfig:
    """Fixed state/resource contract and immutable non-authority disclosures."""

    feature_dim: int
    active_slots: int
    candidate_slots: int
    n_tasks: int
    generator_contexts: int
    generator_policy_count: int
    filler_op: int = OP_GATED
    filler_parent_a: int = 0
    filler_parent_b: int = 1
    schema: str = EXPANDED_EXPRESSION_LINEAGE_SCHEMA
    status: str = EXPANDED_EXPRESSION_LINEAGE_STATUS
    development_only: bool = True
    host_only_not_jittable: bool = True
    target_d_special_casing: bool = False
    behavioral_memory_erasure_claimed: bool = False
    execution_authorized: bool = False
    runner_authorized: bool = False
    campaign_authorized: bool = False
    evidence_authorized: bool = False
    artifact_writes_authorized: bool = False
    scientific_promotion_allowed: bool = False

    def __post_init__(self) -> None:
        integer_fields = (
            "feature_dim",
            "active_slots",
            "candidate_slots",
            "n_tasks",
            "generator_contexts",
            "generator_policy_count",
            "filler_op",
            "filler_parent_a",
            "filler_parent_b",
        )
        for name in integer_fields:
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an exact Python integer")
        if self.feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        if self.active_slots < self.feature_dim:
            raise ValueError("active_slots must be at least feature_dim")
        if self.candidate_slots < 0:
            raise ValueError("candidate_slots must be non-negative")
        if self.n_tasks < 1:
            raise ValueError("n_tasks must be positive")
        if self.generator_contexts < 1:
            raise ValueError("generator_contexts must be positive")
        if self.generator_policy_count < 1:
            raise ValueError("generator_policy_count must be positive")
        if self.filler_op not in {OP_PRODUCT, OP_SUM, OP_GATED}:
            raise ValueError("filler_op must be a parameter-free binary operation")
        for name in ("filler_parent_a", "filler_parent_b"):
            value = getattr(self, name)
            if not 0 <= value < self.feature_dim:
                raise ValueError(f"{name} must name a raw-prefix slot")
        if self.filler_op in {OP_PRODUCT, OP_SUM} and (self.filler_parent_a > self.filler_parent_b):
            raise ValueError("commutative filler parents must be canonicalized")

        for name in ("schema", "status"):
            if type(getattr(self, name)) is not str:
                raise TypeError(f"{name} must be an exact Python string")
        boolean_fields = (
            "development_only",
            "host_only_not_jittable",
            "target_d_special_casing",
            "behavioral_memory_erasure_claimed",
            "execution_authorized",
            "runner_authorized",
            "campaign_authorized",
            "evidence_authorized",
            "artifact_writes_authorized",
            "scientific_promotion_allowed",
        )
        for name in boolean_fields:
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact Python boolean")
        if self.schema != EXPANDED_EXPRESSION_LINEAGE_SCHEMA:
            raise ValueError("expanded-expression lineage schema is not canonical")
        if self.status != EXPANDED_EXPRESSION_LINEAGE_STATUS:
            raise ValueError("expanded-expression lineage status is not canonical")
        if not self.development_only or not self.host_only_not_jittable:
            raise ValueError("lineage compilation must remain development host-only work")
        if self.target_d_special_casing:
            raise ValueError("target-D special casing is forbidden")
        if (
            self.behavioral_memory_erasure_claimed
            or self.execution_authorized
            or self.runner_authorized
            or self.campaign_authorized
            or self.evidence_authorized
            or self.artifact_writes_authorized
            or self.scientific_promotion_allowed
        ):
            raise ValueError("lineage configuration cannot grant erasure or external authority")


@dataclasses.dataclass(frozen=True, slots=True)
class ExpandedExpressionLineageAudit:
    """Raw hashes, counts, resource bytes, and host-operation accounting."""

    schema: str
    status: str
    config_sha256: str
    target_expression_sha256: str
    filler_expression_sha256: str
    pre_state_bit_sha256: str
    persistent_resource_signature_sha256: str
    active_root_bank_sha256: str
    candidate_root_bank_sha256: str
    active_mask_sha256: str
    candidate_mask_sha256: str
    plan_sha256: str
    active_root_count: int
    candidate_root_count: int
    active_exact_target_root_count: int
    candidate_exact_target_root_count: int
    active_roots_containing_target: int
    candidate_roots_containing_target: int
    active_target_subtree_occurrences: int
    candidate_target_subtree_occurrences: int
    active_descendant_dependency_roots: int
    candidate_active_dependency_roots: int
    active_mask_count: int
    candidate_mask_count: int
    pre_target_present: bool
    nonempty_causal_plan: bool
    active_expanded_node_visits: int
    candidate_expanded_node_visits: int
    logical_subtree_identity_comparisons: int
    active_parent_edges_audited: int
    candidate_parent_edges_audited: int
    public_expression_constructor_invocations: int
    public_expression_digest_invocations: int
    state_persistent_array_nbytes: int
    expected_state_persistent_array_nbytes: int
    expected_state_persistent_array_nbytes_formula: str
    mask_persistent_array_nbytes: int
    host_audit_metadata_bytes_included: bool
    host_timing_metadata_bound_in_pre_state_hash: bool
    host_only_not_jittable: bool
    learner_update_jax_kernel_operations: int
    operation_accounting_scope: str
    wall_clock_threshold: float | None
    target_d_special_casing: bool
    behavioral_memory_erasure_claimed: bool
    execution_authorized: bool
    runner_authorized: bool
    campaign_authorized: bool
    evidence_authorized: bool
    artifact_writes_authorized: bool
    scientific_promotion_allowed: bool


@dataclasses.dataclass(frozen=True, slots=True)
class ExpandedExpressionLineagePlan:
    """Fixed-shape canonical active/candidate masks plus their binding audit."""

    active_mask: Array
    candidate_mask: Array
    audit: ExpandedExpressionLineageAudit


@dataclasses.dataclass(frozen=True, slots=True)
class ExpandedExpressionScrubValidation:
    """Raw result of binding a canonical pre-plan to expanded post-state absence."""

    schema: str
    status: str
    target_expression_sha256: str
    canonical_plan_sha256: str
    supplied_plan_sha256: str
    post_active_root_bank_sha256: str
    post_candidate_root_bank_sha256: str
    pre_target_present: bool
    nonempty_causal_plan: bool
    plan_matches_canonical: bool
    recomputed_commit_succeeded: bool
    transaction_matches_recomputed_commit: bool
    static_resources_preserved: bool
    persistent_array_nbytes_preserved: bool
    pre_state_persistent_array_nbytes: int
    post_state_persistent_array_nbytes: int
    active_roots_containing_target_after: int
    candidate_roots_containing_target_after: int
    active_target_subtree_occurrences_after: int
    candidate_target_subtree_occurrences_after: int
    target_absent_from_all_expanded_trees: bool
    valid: bool
    validation_sha256: str
    target_d_special_casing: bool
    behavioral_memory_erasure_claimed: bool
    execution_authorized: bool
    runner_authorized: bool
    campaign_authorized: bool
    evidence_authorized: bool
    artifact_writes_authorized: bool
    scientific_promotion_allowed: bool


@dataclasses.dataclass(frozen=True, slots=True)
class _ValidatedState:
    arrays: dict[str, np.ndarray[Any, Any]]
    persistent_array_nbytes: int


@dataclasses.dataclass(frozen=True, slots=True)
class _ExpressionAnalysis:
    active: tuple[GeneratedExpression, ...]
    candidates: tuple[GeneratedExpression, ...]
    active_root_digests: tuple[str, ...]
    candidate_root_digests: tuple[str, ...]
    active_target_occurrences: tuple[int, ...]
    candidate_target_occurrences: tuple[int, ...]
    active_mask: np.ndarray[Any, np.dtype[np.bool_]]
    candidate_mask: np.ndarray[Any, np.dtype[np.bool_]]
    active_descendant_dependency: np.ndarray[Any, np.dtype[np.bool_]]
    candidate_active_dependency: np.ndarray[Any, np.dtype[np.bool_]]
    active_expanded_node_visits: int
    candidate_expanded_node_visits: int
    active_parent_edges_audited: int
    candidate_parent_edges_audited: int
    persistent_array_nbytes: int


def _state_array_specs(
    config: ExpandedExpressionLineageConfig,
) -> dict[str, tuple[tuple[int, ...], Any]]:
    n = config.active_slots
    c = config.candidate_slots
    h = config.n_tasks
    return {
        "ops": ((n,), jnp.int32),
        "parent_a": ((n,), jnp.int32),
        "parent_b": ((n,), jnp.int32),
        "theta": ((n, 2), jnp.float32),
        "depth": ((n,), jnp.int32),
        "output_weights": ((h, n), jnp.float32),
        "output_bias": ((h,), jnp.float32),
        "utilities": ((n,), jnp.float32),
        "utility_contribution_trace": ((h, n), jnp.float32),
        "utility_error_trace": ((h,), jnp.float32),
        "utility_feature_trace": ((n,), jnp.float32),
        "utility_feature_energy_trace": ((n,), jnp.float32),
        "utility_signal_second_moment": ((n,), jnp.float32),
        "feature_score_residual_trace": ((h, n), jnp.float32),
        "feature_score_energy_trace": ((n,), jnp.float32),
        "retention_slow_utilities": ((n,), jnp.float32),
        "task_activity_ema": ((h,), jnp.float32),
        "ages": ((n,), jnp.int32),
        "candidate_ops": ((c,), jnp.int32),
        "candidate_parent_a": ((c,), jnp.int32),
        "candidate_parent_b": ((c,), jnp.int32),
        "candidate_theta": ((c, 2), jnp.float32),
        "candidate_depth": ((c,), jnp.int32),
        "candidate_output_weights": ((h, c), jnp.float32),
        "candidate_utilities": ((c,), jnp.float32),
        "candidate_utility_contribution_trace": ((h, c), jnp.float32),
        "candidate_utility_feature_trace": ((c,), jnp.float32),
        "candidate_utility_feature_energy_trace": ((c,), jnp.float32),
        "candidate_utility_signal_second_moment": ((c,), jnp.float32),
        "candidate_score_residual_trace": ((h, c), jnp.float32),
        "candidate_score_energy_trace": ((c,), jnp.float32),
        "candidate_retention_slow_utilities": ((c,), jnp.float32),
        "candidate_active_correlation_trace": ((c, n), jnp.float32),
        "candidate_ages": ((c,), jnp.int32),
        "candidate_selector_log_weights": ((c,), jnp.float32),
        "candidate_selector_cumulative_loss": ((c,), jnp.float32),
        "candidate_selector_action_counts": ((c,), jnp.float32),
        "feature_generator_policy": ((n,), jnp.int32),
        "candidate_generator_policy": ((c,), jnp.int32),
        "replacement_accumulator": ((), jnp.float32),
        "step_count": ((), jnp.int32),
        "step_words": ((2,), jnp.uint32),
        "replacement_phase": ((), jnp.int32),
    }


def _expected_persistent_array_nbytes(config: ExpandedExpressionLineageConfig) -> int:
    specs = _state_array_specs(config)
    top_level = sum(math.prod(shape) * np.dtype(dtype).itemsize for shape, dtype in specs.values())
    generator_matrix_values = 3 * config.generator_contexts * config.generator_policy_count
    generator = generator_matrix_values * np.dtype(np.float32).itemsize
    generator += np.dtype(np.int32).itemsize
    typed_threefry_key_data = 2 * np.dtype(np.uint32).itemsize
    return int(top_level + generator + typed_threefry_key_data)


def _require_host_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> np.ndarray[Any, Any]:
    if not isinstance(value, Array):
        raise TypeError(f"{name} must be a concrete JAX array for host-only compilation")
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if array.dtype != np.dtype(dtype):
        raise TypeError(f"{name} must have dtype {np.dtype(dtype)}, got {array.dtype}")
    return array


def _require_typed_threefry_key(state: CompositionalFeatureState) -> None:
    key = state.key
    if not isinstance(key, Array):
        raise TypeError("state.key must be one concrete scalar typed Threefry key")
    if key.shape != () or not jax.dtypes.issubdtype(  # type: ignore[attr-defined]
        key.dtype,
        jax.dtypes.prng_key,
    ):
        raise TypeError("state.key must be one scalar typed Threefry key")
    if str(jax.random.key_impl(key)) != "threefry2x32":
        raise TypeError("state.key must use the typed Threefry implementation")
    key_data = jax.random.key_data(key)
    if key_data.shape != (2,) or key_data.dtype != jnp.uint32:
        raise TypeError("state.key must expose exactly two uint32 Threefry words")


def _validate_state(
    state: CompositionalFeatureState,
    config: ExpandedExpressionLineageConfig,
) -> _ValidatedState:
    if type(state) is not CompositionalFeatureState:
        raise TypeError("state must be an exact CompositionalFeatureState")
    compositional_state_leaf_paths(state)
    specs = _state_array_specs(config)
    expected_top_arrays = {
        path
        for path in COMPOSITIONAL_STATE_LEAF_PATHS
        if "." not in path and path not in {"birth_timestamp", "key", "uptime_s"}
    }
    if set(specs) != expected_top_arrays:
        missing = sorted(expected_top_arrays - set(specs))
        extra = sorted(set(specs) - expected_top_arrays)
        raise RuntimeError(
            f"lineage state-array contract is stale; missing={missing}, extra={extra}"
        )

    arrays: dict[str, np.ndarray[Any, Any]] = {}
    for name, (shape, dtype) in specs.items():
        array = _require_host_array(
            getattr(state, name),
            name=f"state.{name}",
            shape=shape,
            dtype=dtype,
        )
        if array.dtype == np.dtype(np.float32) and not bool(np.all(np.isfinite(array))):
            raise ValueError(f"state.{name} must be finite")
        arrays[name] = array

    for name in ("birth_timestamp", "uptime_s"):
        timing = np.asarray(getattr(state, name))
        if timing.shape != () or not np.issubdtype(timing.dtype, np.floating):
            raise TypeError(f"state.{name} must be one finite host floating scalar")
        if not bool(np.isfinite(timing)):
            raise ValueError(f"state.{name} must be finite")

    _require_typed_threefry_key(state)
    generator = state.generator_resource_state
    if type(generator) is not GeneratorMetaResourceManagerState:
        raise TypeError(
            "state.generator_resource_state must be an exact GeneratorMetaResourceManagerState"
        )
    generator_shape = (config.generator_contexts, config.generator_policy_count)
    generator_arrays: dict[str, np.ndarray[Any, Any]] = {}
    for name in ("log_weights", "reward_ema", "action_counts"):
        array = _require_host_array(
            getattr(generator, name),
            name=f"state.generator_resource_state.{name}",
            shape=generator_shape,
            dtype=jnp.float32,
        )
        if not bool(np.all(np.isfinite(array))):
            raise ValueError(f"state.generator_resource_state.{name} must be finite")
        generator_arrays[name] = array
    generator_step = _require_host_array(
        generator.step_count,
        name="state.generator_resource_state.step_count",
        shape=(),
        dtype=jnp.int32,
    )

    if int(arrays["step_count"]) < 0 or int(generator_step) < 0:
        raise ValueError("state counters must be non-negative")
    if bool(np.any(arrays["ages"] < 0)) or bool(np.any(arrays["candidate_ages"] < 0)):
        raise ValueError("state ages must be non-negative")
    if bool(np.any(arrays["candidate_selector_action_counts"] < 0.0)):
        raise ValueError("state candidate selector counts must be non-negative")
    if bool(np.any(generator_arrays["action_counts"] < 0.0)):
        raise ValueError("state generator action counts must be non-negative")
    if float(arrays["replacement_accumulator"]) < 0.0:
        raise ValueError("state.replacement_accumulator must be non-negative")

    feature_policy = arrays["feature_generator_policy"]
    if bool(np.any(feature_policy < 0)) or bool(
        np.any(feature_policy >= config.generator_policy_count)
    ):
        raise ValueError("state.feature_generator_policy is outside the policy width")
    candidate_policy = arrays["candidate_generator_policy"]
    if bool(np.any(candidate_policy < 0)) or bool(
        np.any(candidate_policy >= config.generator_policy_count)
    ):
        raise ValueError("state.candidate_generator_policy is outside the policy width")

    actual_nbytes = persistent_compositional_state_nbytes(state)
    expected_nbytes = _expected_persistent_array_nbytes(config)
    if actual_nbytes != expected_nbytes:
        raise ValueError(
            "state persistent resource bytes drifted from the static contract; "
            f"got {actual_nbytes}, expected {expected_nbytes}"
        )
    return _ValidatedState(arrays=arrays, persistent_array_nbytes=actual_nbytes)


def _target_digest(target: GeneratedExpression, feature_dim: int) -> str:
    if type(target) is not GeneratedExpression:
        raise TypeError("target must be an exact canonical GeneratedExpression")
    digest = expression_digest(target)
    if target.op == "raw":
        raise ValueError("raw target lineage would require scrubbing the immutable raw prefix")

    def check_raw_indices(node: GeneratedExpression) -> None:
        if node.op == "raw":
            if node.raw_index is None or not 0 <= node.raw_index < feature_dim:
                raise ValueError("target raw index is outside config.feature_dim")
            return
        if node.left is None or node.right is None:
            raise ValueError("canonical target binary node is missing a child")
        check_raw_indices(node.left)
        check_raw_indices(node.right)

    check_raw_indices(target)
    return digest


def _binary_expression(
    op: int,
    left: GeneratedExpression,
    right: GeneratedExpression,
    theta: np.ndarray[Any, Any],
) -> GeneratedExpression:
    if op == OP_PRODUCT:
        return product_expression(left, right)
    if op == OP_SUM:
        return sum_expression(left, right)
    if op == OP_TANH:
        return tanh_expression(
            left,
            right,
            theta0=float(theta[0]),
            theta1=float(theta[1]),
        )
    if op == OP_GATED:
        return gate_expression(left, right)
    raise ValueError("compositional state contains an unsupported operation")


def _filler_expression(config: ExpandedExpressionLineageConfig) -> GeneratedExpression:
    left = raw_expression(config.filler_parent_a)
    right = raw_expression(config.filler_parent_b)
    if config.filler_op == OP_PRODUCT:
        return product_expression(left, right)
    if config.filler_op == OP_SUM:
        return sum_expression(left, right)
    if config.filler_op == OP_GATED:
        return gate_expression(left, right)
    raise RuntimeError("validated lineage filler operation became unsupported")


def _build_expression_banks(
    validated: _ValidatedState,
    config: ExpandedExpressionLineageConfig,
) -> tuple[
    tuple[GeneratedExpression, ...],
    tuple[GeneratedExpression, ...],
    int,
    int,
]:
    arrays = validated.arrays
    ops = arrays["ops"]
    parent_a = arrays["parent_a"]
    parent_b = arrays["parent_b"]
    theta = arrays["theta"]
    depth = arrays["depth"]
    active: list[GeneratedExpression] = []
    active_edges = 0
    for slot in range(config.active_slots):
        op = int(ops[slot])
        left_index = int(parent_a[slot])
        right_index = int(parent_b[slot])
        if not OP_RAW <= op < NUM_OPS:
            raise ValueError("active compositional operation is outside the grammar")
        if slot < config.feature_dim:
            if op != OP_RAW or left_index != slot or right_index != -1 or int(depth[slot]) != 0:
                raise ValueError("active raw prefix is not exact")
            expression = raw_expression(left_index)
        else:
            if op == OP_RAW:
                raise ValueError("active raw operation appears outside the exact prefix")
            if not (0 <= left_index < slot and 0 <= right_index < slot):
                raise ValueError("active compositional state is not topologically ordered")
            expected_depth = max(int(depth[left_index]), int(depth[right_index])) + 1
            if int(depth[slot]) != expected_depth:
                raise ValueError("active compositional depth is inconsistent with its parents")
            expression = _binary_expression(
                op,
                active[left_index],
                active[right_index],
                theta[slot],
            )
            active_edges += 2
        active.append(expression)

    candidate_ops = arrays["candidate_ops"]
    candidate_parent_a = arrays["candidate_parent_a"]
    candidate_parent_b = arrays["candidate_parent_b"]
    candidate_theta = arrays["candidate_theta"]
    candidate_depth = arrays["candidate_depth"]
    candidates: list[GeneratedExpression] = []
    candidate_edges = 0
    for slot in range(config.candidate_slots):
        op = int(candidate_ops[slot])
        left_index = int(candidate_parent_a[slot])
        right_index = int(candidate_parent_b[slot])
        if not OP_RAW <= op < NUM_OPS:
            raise ValueError("candidate compositional operation is outside the grammar")
        if op == OP_RAW:
            if (
                not 0 <= left_index < config.feature_dim
                or right_index != -1
                or int(candidate_depth[slot]) != 0
            ):
                raise ValueError("candidate raw expression is malformed")
            expression = raw_expression(left_index)
        else:
            if not (
                0 <= left_index < config.active_slots and 0 <= right_index < config.active_slots
            ):
                raise ValueError("candidate compositional state has an invalid active parent")
            expected_depth = max(int(depth[left_index]), int(depth[right_index])) + 1
            if int(candidate_depth[slot]) != expected_depth:
                raise ValueError("candidate compositional depth is inconsistent with its parents")
            expression = _binary_expression(
                op,
                active[left_index],
                active[right_index],
                candidate_theta[slot],
            )
            candidate_edges += 2
        candidates.append(expression)
    return tuple(active), tuple(candidates), active_edges, candidate_edges


def _expanded_nodes(expression: GeneratedExpression) -> tuple[GeneratedExpression, ...]:
    if expression.op == "raw":
        return (expression,)
    if expression.left is None or expression.right is None:
        raise ValueError("canonical expression is missing a child")
    return (
        expression,
        *_expanded_nodes(expression.left),
        *_expanded_nodes(expression.right),
    )


def _analyze_state(
    state: CompositionalFeatureState,
    target: GeneratedExpression,
    config: ExpandedExpressionLineageConfig,
) -> tuple[_ExpressionAnalysis, str]:
    target_sha256 = _target_digest(target, config.feature_dim)
    validated = _validate_state(state, config)
    active, candidates, active_edges, candidate_edges = _build_expression_banks(
        validated,
        config,
    )
    active_root_digests = tuple(expression_digest(item) for item in active)
    candidate_root_digests = tuple(expression_digest(item) for item in candidates)
    active_nodes = tuple(_expanded_nodes(item) for item in active)
    candidate_nodes = tuple(_expanded_nodes(item) for item in candidates)
    active_occurrences = tuple(
        sum(expression_digest(node) == target_sha256 for node in nodes) for nodes in active_nodes
    )
    candidate_occurrences = tuple(
        sum(expression_digest(node) == target_sha256 for node in nodes) for nodes in candidate_nodes
    )
    active_contains = np.asarray(
        tuple(count > 0 for count in active_occurrences),
        dtype=np.bool_,
    )
    candidate_contains = np.asarray(
        tuple(count > 0 for count in candidate_occurrences),
        dtype=np.bool_,
    )

    arrays = validated.arrays
    active_mask = active_contains.copy()
    active_dependency = np.zeros((config.active_slots,), dtype=np.bool_)
    for slot in range(config.feature_dim, config.active_slots):
        left_index = int(arrays["parent_a"][slot])
        right_index = int(arrays["parent_b"][slot])
        depends = bool(active_mask[left_index] or active_mask[right_index])
        active_dependency[slot] = depends
        active_mask[slot] = bool(active_mask[slot] or depends)

    candidate_dependency = np.zeros((config.candidate_slots,), dtype=np.bool_)
    for slot in range(config.candidate_slots):
        if int(arrays["candidate_ops"][slot]) == OP_RAW:
            continue
        left_index = int(arrays["candidate_parent_a"][slot])
        right_index = int(arrays["candidate_parent_b"][slot])
        candidate_dependency[slot] = bool(active_mask[left_index] or active_mask[right_index])
    candidate_mask = candidate_contains | candidate_dependency

    return (
        _ExpressionAnalysis(
            active=active,
            candidates=candidates,
            active_root_digests=active_root_digests,
            candidate_root_digests=candidate_root_digests,
            active_target_occurrences=active_occurrences,
            candidate_target_occurrences=candidate_occurrences,
            active_mask=active_mask,
            candidate_mask=candidate_mask,
            active_descendant_dependency=active_dependency,
            candidate_active_dependency=candidate_dependency,
            active_expanded_node_visits=sum(len(nodes) for nodes in active_nodes),
            candidate_expanded_node_visits=sum(len(nodes) for nodes in candidate_nodes),
            active_parent_edges_audited=active_edges,
            candidate_parent_edges_audited=candidate_edges,
            persistent_array_nbytes=validated.persistent_array_nbytes,
        ),
        target_sha256,
    )


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _path_text(path: tuple[Any, ...]) -> str:
    names: list[str] = []
    for key in path:
        name = getattr(key, "name", None)
        if not isinstance(name, str):
            raise TypeError(f"unsupported compositional state path key: {key!r}")
        names.append(name)
    return ".".join(names)


def _state_leaf_records(
    state: CompositionalFeatureState,
) -> tuple[tuple[str, str, str, tuple[int, ...], bytes], ...]:
    records: list[tuple[str, str, str, tuple[int, ...], bytes]] = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(state)[0]:
        path_name = _path_text(path)
        if isinstance(leaf, Array) and jax.dtypes.issubdtype(  # type: ignore[attr-defined]
            leaf.dtype,
            jax.dtypes.prng_key,
        ):
            array = np.asarray(jax.random.key_data(leaf))
            records.append((path_name, "typed-prng", str(leaf.dtype), array.shape, array.tobytes()))
        elif isinstance(leaf, Array):
            array = np.asarray(leaf)
            records.append((path_name, "jax-array", array.dtype.str, array.shape, array.tobytes()))
        elif type(leaf) is float:
            records.append((path_name, "python-float", ">f8", (), struct.pack(">d", leaf)))
        else:
            raise TypeError(f"unsupported state leaf for exact binding: {path_name}")
    return tuple(records)


def _records_sha256(
    records: tuple[tuple[str, str, str, tuple[int, ...], bytes], ...],
    *,
    include_values: bool,
) -> str:
    digest = hashlib.sha256()
    for path, kind, dtype, shape, raw_bytes in records:
        metadata = json.dumps(
            {
                "path": path,
                "kind": kind,
                "dtype": dtype,
                "shape": list(shape),
                "nbytes": len(raw_bytes),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        if include_values:
            digest.update(len(raw_bytes).to_bytes(8, "big"))
            digest.update(raw_bytes)
    return digest.hexdigest()


def _state_bit_sha256(state: CompositionalFeatureState) -> str:
    return _records_sha256(_state_leaf_records(state), include_values=True)


def _persistent_resource_signature_sha256(state: CompositionalFeatureState) -> str:
    records = tuple(
        record
        for record in _state_leaf_records(state)
        if record[0] not in {"birth_timestamp", "uptime_s"}
    )
    return _records_sha256(records, include_values=False)


def _states_bit_exact(
    first: CompositionalFeatureState,
    second: CompositionalFeatureState,
) -> bool:
    return _state_leaf_records(first) == _state_leaf_records(second)


def _mask_sha256(mask: np.ndarray[Any, np.dtype[np.bool_]]) -> str:
    return _sha256_json(
        {
            "dtype": "bool",
            "shape": list(mask.shape),
            "values": [bool(value) for value in mask],
        }
    )


def _config_sha256(config: ExpandedExpressionLineageConfig) -> str:
    return _sha256_json(dataclasses.asdict(config))


def compile_expanded_expression_lineage_masks(
    state: CompositionalFeatureState,
    target: GeneratedExpression,
    *,
    config: ExpandedExpressionLineageConfig,
) -> ExpandedExpressionLineagePlan:
    """Compile exact expanded-AST lineage masks using public canonical identity.

    All reconstruction, hashing, and closure work happens concretely on the
    host.  The returned masks alone are JAX bool arrays of the two configured
    fixed shapes.  Malformed state, noncanonical targets, and resource drift
    raise instead of yielding a partial plan.
    """

    if type(config) is not ExpandedExpressionLineageConfig:
        raise TypeError("config must be an exact ExpandedExpressionLineageConfig")
    analysis, target_sha256 = _analyze_state(state, target, config)
    filler_sha256 = expression_digest(_filler_expression(config))
    if filler_sha256 == target_sha256:
        raise ValueError("canonical scrub filler cannot equal the retired target expression")
    active_mask = jnp.asarray(analysis.active_mask, dtype=jnp.bool_)
    candidate_mask = jnp.asarray(analysis.candidate_mask, dtype=jnp.bool_)
    active_mask_sha256 = _mask_sha256(analysis.active_mask)
    candidate_mask_sha256 = _mask_sha256(analysis.candidate_mask)
    active_bank_sha256 = _sha256_json(
        {"ordered_active_root_digests": list(analysis.active_root_digests)}
    )
    candidate_bank_sha256 = _sha256_json(
        {"ordered_candidate_root_digests": list(analysis.candidate_root_digests)}
    )
    config_sha256 = _config_sha256(config)
    pre_state_bit_sha256 = _state_bit_sha256(state)
    resource_signature_sha256 = _persistent_resource_signature_sha256(state)
    expected_nbytes = _expected_persistent_array_nbytes(config)
    active_visits = analysis.active_expanded_node_visits
    candidate_visits = analysis.candidate_expanded_node_visits
    pre_target_present = bool(
        sum(analysis.active_target_occurrences) + sum(analysis.candidate_target_occurrences) > 0
    )
    nonempty_causal_plan = bool(np.any(analysis.active_mask) or np.any(analysis.candidate_mask))
    plan_sha256 = _sha256_json(
        {
            "schema": config.schema,
            "status": config.status,
            "config_sha256": config_sha256,
            "target_expression_sha256": target_sha256,
            "filler_expression_sha256": filler_sha256,
            "pre_state_bit_sha256": pre_state_bit_sha256,
            "persistent_resource_signature_sha256": resource_signature_sha256,
            "active_root_bank_sha256": active_bank_sha256,
            "candidate_root_bank_sha256": candidate_bank_sha256,
            "active_mask_sha256": active_mask_sha256,
            "candidate_mask_sha256": candidate_mask_sha256,
            "state_persistent_array_nbytes": analysis.persistent_array_nbytes,
        }
    )
    audit = ExpandedExpressionLineageAudit(
        schema=config.schema,
        status=config.status,
        config_sha256=config_sha256,
        target_expression_sha256=target_sha256,
        filler_expression_sha256=filler_sha256,
        pre_state_bit_sha256=pre_state_bit_sha256,
        persistent_resource_signature_sha256=resource_signature_sha256,
        active_root_bank_sha256=active_bank_sha256,
        candidate_root_bank_sha256=candidate_bank_sha256,
        active_mask_sha256=active_mask_sha256,
        candidate_mask_sha256=candidate_mask_sha256,
        plan_sha256=plan_sha256,
        active_root_count=len(analysis.active),
        candidate_root_count=len(analysis.candidates),
        active_exact_target_root_count=sum(
            digest == target_sha256 for digest in analysis.active_root_digests
        ),
        candidate_exact_target_root_count=sum(
            digest == target_sha256 for digest in analysis.candidate_root_digests
        ),
        active_roots_containing_target=sum(
            count > 0 for count in analysis.active_target_occurrences
        ),
        candidate_roots_containing_target=sum(
            count > 0 for count in analysis.candidate_target_occurrences
        ),
        active_target_subtree_occurrences=sum(analysis.active_target_occurrences),
        candidate_target_subtree_occurrences=sum(analysis.candidate_target_occurrences),
        active_descendant_dependency_roots=int(
            np.count_nonzero(analysis.active_descendant_dependency)
        ),
        candidate_active_dependency_roots=int(
            np.count_nonzero(analysis.candidate_active_dependency)
        ),
        active_mask_count=int(np.count_nonzero(analysis.active_mask)),
        candidate_mask_count=int(np.count_nonzero(analysis.candidate_mask)),
        pre_target_present=pre_target_present,
        nonempty_causal_plan=nonempty_causal_plan,
        active_expanded_node_visits=active_visits,
        candidate_expanded_node_visits=candidate_visits,
        logical_subtree_identity_comparisons=active_visits + candidate_visits,
        active_parent_edges_audited=analysis.active_parent_edges_audited,
        candidate_parent_edges_audited=analysis.candidate_parent_edges_audited,
        public_expression_constructor_invocations=(
            len(analysis.active) + len(analysis.candidates) + 3
        ),
        public_expression_digest_invocations=(
            2 + len(analysis.active) + len(analysis.candidates) + active_visits + candidate_visits
        ),
        state_persistent_array_nbytes=analysis.persistent_array_nbytes,
        expected_state_persistent_array_nbytes=expected_nbytes,
        expected_state_persistent_array_nbytes_formula=(
            "all checked top-level JAX arrays + three float32 generator matrices + "
            "one int32 generator step + two uint32 exact-step words + two uint32 "
            "typed-Threefry key words; "
            "birth_timestamp and uptime_s excluded"
        ),
        mask_persistent_array_nbytes=int(active_mask.nbytes + candidate_mask.nbytes),
        host_audit_metadata_bytes_included=False,
        host_timing_metadata_bound_in_pre_state_hash=True,
        host_only_not_jittable=True,
        learner_update_jax_kernel_operations=0,
        operation_accounting_scope=(
            "learner updates only; host AST compilation, JAX mask materialization, "
            "and validator scrub-kernel work are excluded"
        ),
        wall_clock_threshold=None,
        target_d_special_casing=False,
        behavioral_memory_erasure_claimed=False,
        execution_authorized=False,
        runner_authorized=False,
        campaign_authorized=False,
        evidence_authorized=False,
        artifact_writes_authorized=False,
        scientific_promotion_allowed=False,
    )
    return ExpandedExpressionLineagePlan(
        active_mask=active_mask,
        candidate_mask=candidate_mask,
        audit=audit,
    )


def _require_supplied_plan(
    plan: ExpandedExpressionLineagePlan,
    config: ExpandedExpressionLineageConfig,
) -> tuple[np.ndarray[Any, np.dtype[np.bool_]], np.ndarray[Any, np.dtype[np.bool_]]]:
    if type(plan) is not ExpandedExpressionLineagePlan:
        raise TypeError("plan must be an exact ExpandedExpressionLineagePlan")
    if type(plan.audit) is not ExpandedExpressionLineageAudit:
        raise TypeError("plan.audit must be an exact ExpandedExpressionLineageAudit")
    active = _require_host_array(
        plan.active_mask,
        name="plan.active_mask",
        shape=(config.active_slots,),
        dtype=jnp.bool_,
    )
    candidate = _require_host_array(
        plan.candidate_mask,
        name="plan.candidate_mask",
        shape=(config.candidate_slots,),
        dtype=jnp.bool_,
    )
    return active, candidate


def _require_scrub_config_binding(
    scrub_config: GeneratedClassScrubConfig,
    config: ExpandedExpressionLineageConfig,
) -> None:
    if type(scrub_config) is not GeneratedClassScrubConfig:
        raise TypeError("scrub_config must be an exact GeneratedClassScrubConfig")
    expected = (
        config.feature_dim,
        config.active_slots,
        config.candidate_slots,
        config.n_tasks,
        config.filler_op,
        config.filler_parent_a,
        config.filler_parent_b,
    )
    actual = (
        scrub_config.feature_dim,
        scrub_config.active_slots,
        scrub_config.candidate_slots,
        scrub_config.n_tasks,
        scrub_config.filler_op,
        scrub_config.filler_parent_a,
        scrub_config.filler_parent_b,
    )
    if actual != expected:
        raise ValueError("scrub_config shape or filler contract does not match lineage config")


def validate_post_scrub_expanded_expression_absence(
    pre_scrub_state: CompositionalFeatureState,
    post_scrub_state: CompositionalFeatureState,
    target: GeneratedExpression,
    plan: ExpandedExpressionLineagePlan,
    *,
    config: ExpandedExpressionLineageConfig,
    scrub_config: GeneratedClassScrubConfig,
) -> ExpandedExpressionScrubValidation:
    """Validate an untrusted pre-plan and exact target absence after scrubbing.

    The supplied masks and audit are never trusted.  They are compared with an
    independently recompiled canonical plan bound to every pre-state bit and
    ``target``.  The canonical scrub transaction is independently recomputed,
    the supplied post-state must equal that committed result bit-for-bit, and
    every post-scrub active/candidate AST is searched at every expanded node.
    This proves structural target absence for that exact transaction only;
    behavioral memory, archives, algebraic/alpha-renamed equivalence, and future
    reacquisition are outside the claim.
    """

    if type(config) is not ExpandedExpressionLineageConfig:
        raise TypeError("config must be an exact ExpandedExpressionLineageConfig")
    _require_scrub_config_binding(scrub_config, config)
    supplied_active, supplied_candidate = _require_supplied_plan(plan, config)
    canonical = compile_expanded_expression_lineage_masks(
        pre_scrub_state,
        target,
        config=config,
    )
    canonical_active = np.asarray(canonical.active_mask)
    canonical_candidate = np.asarray(canonical.candidate_mask)
    plan_matches = bool(
        np.array_equal(supplied_active, canonical_active)
        and np.array_equal(supplied_candidate, canonical_candidate)
        and plan.audit == canonical.audit
    )
    recomputed = scrub_compositional_feature_state(
        pre_scrub_state,
        canonical.active_mask,
        canonical.candidate_mask,
        jnp.asarray(True, dtype=jnp.bool_),
        config=scrub_config,
    )
    recomputed_commit_succeeded = bool(recomputed.diagnostics.committed)
    transaction_matches_recomputed_commit = bool(
        recomputed_commit_succeeded and _states_bit_exact(post_scrub_state, recomputed.state)
    )

    post_analysis, target_sha256 = _analyze_state(post_scrub_state, target, config)
    post_active_bank_sha256 = _sha256_json(
        {"ordered_active_root_digests": list(post_analysis.active_root_digests)}
    )
    post_candidate_bank_sha256 = _sha256_json(
        {"ordered_candidate_root_digests": list(post_analysis.candidate_root_digests)}
    )
    expected_nbytes = _expected_persistent_array_nbytes(config)
    pre_nbytes = canonical.audit.state_persistent_array_nbytes
    post_nbytes = post_analysis.persistent_array_nbytes
    static_resources_preserved = bool(
        pre_nbytes == post_nbytes == expected_nbytes
        and len(post_analysis.active) == config.active_slots
        and len(post_analysis.candidates) == config.candidate_slots
    )
    persistent_nbytes_preserved = pre_nbytes == post_nbytes
    active_roots_after = sum(count > 0 for count in post_analysis.active_target_occurrences)
    candidate_roots_after = sum(count > 0 for count in post_analysis.candidate_target_occurrences)
    active_occurrences_after = sum(post_analysis.active_target_occurrences)
    candidate_occurrences_after = sum(post_analysis.candidate_target_occurrences)
    target_absent = bool(active_occurrences_after == 0 and candidate_occurrences_after == 0)
    valid = bool(
        canonical.audit.pre_target_present
        and canonical.audit.nonempty_causal_plan
        and plan_matches
        and recomputed_commit_succeeded
        and transaction_matches_recomputed_commit
        and static_resources_preserved
        and persistent_nbytes_preserved
        and target_absent
    )
    validation_sha256 = _sha256_json(
        {
            "schema": config.schema,
            "status": config.status,
            "target_expression_sha256": target_sha256,
            "canonical_plan_sha256": canonical.audit.plan_sha256,
            "supplied_plan_sha256": plan.audit.plan_sha256,
            "post_active_root_bank_sha256": post_active_bank_sha256,
            "post_candidate_root_bank_sha256": post_candidate_bank_sha256,
            "pre_target_present": canonical.audit.pre_target_present,
            "nonempty_causal_plan": canonical.audit.nonempty_causal_plan,
            "plan_matches_canonical": plan_matches,
            "recomputed_commit_succeeded": recomputed_commit_succeeded,
            "transaction_matches_recomputed_commit": (transaction_matches_recomputed_commit),
            "static_resources_preserved": static_resources_preserved,
            "persistent_array_nbytes_preserved": persistent_nbytes_preserved,
            "active_target_subtree_occurrences_after": active_occurrences_after,
            "candidate_target_subtree_occurrences_after": candidate_occurrences_after,
            "valid": valid,
        }
    )
    return ExpandedExpressionScrubValidation(
        schema=config.schema,
        status=config.status,
        target_expression_sha256=target_sha256,
        canonical_plan_sha256=canonical.audit.plan_sha256,
        supplied_plan_sha256=plan.audit.plan_sha256,
        post_active_root_bank_sha256=post_active_bank_sha256,
        post_candidate_root_bank_sha256=post_candidate_bank_sha256,
        pre_target_present=canonical.audit.pre_target_present,
        nonempty_causal_plan=canonical.audit.nonempty_causal_plan,
        plan_matches_canonical=plan_matches,
        recomputed_commit_succeeded=recomputed_commit_succeeded,
        transaction_matches_recomputed_commit=transaction_matches_recomputed_commit,
        static_resources_preserved=static_resources_preserved,
        persistent_array_nbytes_preserved=persistent_nbytes_preserved,
        pre_state_persistent_array_nbytes=pre_nbytes,
        post_state_persistent_array_nbytes=post_nbytes,
        active_roots_containing_target_after=active_roots_after,
        candidate_roots_containing_target_after=candidate_roots_after,
        active_target_subtree_occurrences_after=active_occurrences_after,
        candidate_target_subtree_occurrences_after=candidate_occurrences_after,
        target_absent_from_all_expanded_trees=target_absent,
        valid=valid,
        validation_sha256=validation_sha256,
        target_d_special_casing=False,
        behavioral_memory_erasure_claimed=False,
        execution_authorized=False,
        runner_authorized=False,
        campaign_authorized=False,
        evidence_authorized=False,
        artifact_writes_authorized=False,
        scientific_promotion_allowed=False,
    )


__all__ = [
    "EXPANDED_EXPRESSION_LINEAGE_SCHEMA",
    "EXPANDED_EXPRESSION_LINEAGE_STATUS",
    "ExpandedExpressionLineageAudit",
    "ExpandedExpressionLineageConfig",
    "ExpandedExpressionLineagePlan",
    "ExpandedExpressionScrubValidation",
    "compile_expanded_expression_lineage_masks",
    "validate_post_scrub_expanded_expression_absence",
]
