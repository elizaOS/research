"""Development-only causal scrub kernel for compositional feature identities.

This module implements one deliberately narrow transaction: callers supply a
fixed-shape mask covering retired active expressions, every active descendant,
and every shadow candidate whose meaning depends on those active slots.  The
transaction replaces those descriptors with a canonical parameter-free filler
and zeros every slot-local head, trace, utility, age, selector, correlation,
and generator-provenance leaf.

The kernel does *not* erase all information about prior experience.  Shared
output bias/error traces and unmasked feature state can still encode it.  The
only supported interpretation is deletion of the supplied identity/head
lineage.  This is development infrastructure, has no target-D knowledge, does
not mutate any recurrence prerequisite, and grants no execution, evidence, or
promotion authority.

Descriptor checks in this module are identity-local: they compare only each
slot's encoded operation, immediate parents, effective parameters, and depth.
They do not expand expression trees and cannot establish absence from any
caller-defined target, environment, or latent variable.  They also do not
canonicalize algebraic or reparameterized equivalence, such as swapping a
tanh slot's parents together with its coefficients.  A caller-owned expression
compiler must separately prove expanded-tree absence when that stronger
postcondition is required.

The two host timing leaves must already use a canonical array representation
when callers require bit-exact equality between eager and JIT results.  Array
rollback itself remains bit-exact; this kernel does not coerce host timing
metadata during the transaction.
"""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Int

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

GENERATED_CLASS_SCRUB_SCHEMA = "alberta.generated-class-lifecycle-scrub.development.v0"
GENERATED_CLASS_SCRUB_STATUS = "DEVELOPMENT_IDENTITY_LINEAGE_ONLY"


# Every path is a concrete PyTree leaf path, including the nested generator
# manager.  These four sets are intentionally exhaustive and pairwise disjoint.
# A future CompositionalFeatureState leaf therefore fails the contract instead
# of silently becoming an untracked memory channel.
ACTIVE_MASKED_LEAF_PATHS = frozenset(
    {
        "ages",
        "depth",
        "feature_generator_policy",
        "feature_score_energy_trace",
        "feature_score_residual_trace",
        "ops",
        "output_weights",
        "parent_a",
        "parent_b",
        "retention_slow_utilities",
        "theta",
        "utilities",
        "utility_contribution_trace",
        "utility_feature_energy_trace",
        "utility_feature_trace",
        "utility_signal_second_moment",
    }
)

CANDIDATE_MASKED_LEAF_PATHS = frozenset(
    {
        "candidate_ages",
        "candidate_depth",
        "candidate_generator_policy",
        "candidate_ops",
        "candidate_output_weights",
        "candidate_parent_a",
        "candidate_parent_b",
        "candidate_retention_slow_utilities",
        "candidate_score_energy_trace",
        "candidate_score_residual_trace",
        "candidate_selector_action_counts",
        "candidate_selector_cumulative_loss",
        "candidate_selector_log_weights",
        "candidate_theta",
        "candidate_utilities",
        "candidate_utility_contribution_trace",
        "candidate_utility_feature_energy_trace",
        "candidate_utility_feature_trace",
        "candidate_utility_signal_second_moment",
    }
)

CROSS_MASKED_LEAF_PATHS = frozenset({"candidate_active_correlation_trace"})

PRESERVED_LEAF_PATHS = frozenset(
    {
        "birth_timestamp",
        "generator_resource_state.action_counts",
        "generator_resource_state.log_weights",
        "generator_resource_state.reward_ema",
        "generator_resource_state.step_count",
        "key",
        "output_bias",
        "replacement_accumulator",
        "replacement_phase",
        "step_count",
        "step_words",
        "task_activity_ema",
        "uptime_s",
        "utility_error_trace",
    }
)

COMPOSITIONAL_STATE_LEAF_PATHS = frozenset().union(
    ACTIVE_MASKED_LEAF_PATHS,
    CANDIDATE_MASKED_LEAF_PATHS,
    CROSS_MASKED_LEAF_PATHS,
    PRESERVED_LEAF_PATHS,
)


def _assert_declared_partition_matches_dataclasses() -> None:
    top_level = {
        field.name
        for field in dataclasses.fields(
            CompositionalFeatureState  # type: ignore[arg-type]
        )
    }
    declared_top_level = {path.split(".", maxsplit=1)[0] for path in COMPOSITIONAL_STATE_LEAF_PATHS}
    if declared_top_level != top_level:
        missing = sorted(top_level - declared_top_level)
        extra = sorted(declared_top_level - top_level)
        raise RuntimeError(
            "compositional scrub leaf partition is stale; "
            f"missing top-level fields={missing}, extra top-level fields={extra}"
        )
    generator_fields = {
        f"generator_resource_state.{field.name}"
        for field in dataclasses.fields(
            GeneratorMetaResourceManagerState  # type: ignore[arg-type]
        )
    }
    declared_generator_fields = {
        path
        for path in COMPOSITIONAL_STATE_LEAF_PATHS
        if path.startswith("generator_resource_state.")
    }
    if declared_generator_fields != generator_fields:
        missing = sorted(generator_fields - declared_generator_fields)
        extra = sorted(declared_generator_fields - generator_fields)
        raise RuntimeError(
            "compositional scrub generator-state partition is stale; "
            f"missing fields={missing}, extra fields={extra}"
        )


_assert_declared_partition_matches_dataclasses()


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedClassScrubConfig:
    """Static shape and canonical filler contract for one scrub kernel."""

    feature_dim: int
    active_slots: int
    candidate_slots: int
    n_tasks: int
    filler_op: int = OP_GATED
    filler_parent_a: int = 0
    filler_parent_b: int = 1
    schema: str = GENERATED_CLASS_SCRUB_SCHEMA
    status: str = GENERATED_CLASS_SCRUB_STATUS
    development_only: bool = True
    identity_head_lineage_only: bool = True
    behavioral_information_erasure_claimed: bool = False
    expanded_expression_absence_claimed: bool = False
    host_timing_canonicalization_required_for_jit_bit_equality: bool = True
    execution_authorized: bool = False
    evidence_authorized: bool = False
    scientific_promotion_allowed: bool = False

    def __post_init__(self) -> None:
        for name in ("feature_dim", "active_slots", "candidate_slots", "n_tasks"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an exact Python integer")
        if self.feature_dim < 2:
            raise ValueError("feature_dim must be at least two for the canonical filler")
        if self.active_slots < self.feature_dim:
            raise ValueError("active_slots must be at least feature_dim")
        if self.candidate_slots < 0:
            raise ValueError("candidate_slots must be non-negative")
        if self.n_tasks < 1:
            raise ValueError("n_tasks must be positive")
        if type(self.filler_op) is not int:
            raise TypeError("filler_op must be an exact Python integer")
        if self.filler_op not in {OP_PRODUCT, OP_SUM, OP_GATED}:
            raise ValueError("filler_op must be a parameter-free binary operation")
        for name in ("filler_parent_a", "filler_parent_b"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an exact Python integer")
            if not 0 <= value < self.feature_dim:
                raise ValueError(f"{name} must name a raw-prefix slot")
        if self.filler_op in {OP_PRODUCT, OP_SUM} and (self.filler_parent_a > self.filler_parent_b):
            raise ValueError("commutative filler parents must be canonicalized")
        for name in ("schema", "status"):
            if type(getattr(self, name)) is not str:
                raise TypeError(f"{name} must be an exact Python string")
        boolean_fields = (
            "development_only",
            "identity_head_lineage_only",
            "behavioral_information_erasure_claimed",
            "expanded_expression_absence_claimed",
            "host_timing_canonicalization_required_for_jit_bit_equality",
            "execution_authorized",
            "evidence_authorized",
            "scientific_promotion_allowed",
        )
        for name in boolean_fields:
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact Python boolean")
        if self.schema != GENERATED_CLASS_SCRUB_SCHEMA:
            raise ValueError("scrub schema is not canonical")
        if self.status != GENERATED_CLASS_SCRUB_STATUS:
            raise ValueError("scrub status is not canonical")
        if not self.development_only or not self.identity_head_lineage_only:
            raise ValueError("scrub must remain development-only and identity-local")
        if not self.host_timing_canonicalization_required_for_jit_bit_equality:
            raise ValueError("host timing canonicalization disclosure must remain enabled")
        if (
            self.behavioral_information_erasure_claimed
            or self.expanded_expression_absence_claimed
            or self.execution_authorized
            or self.evidence_authorized
            or self.scientific_promotion_allowed
        ):
            raise ValueError(
                "scrub configuration cannot grant behavioral/expanded-expression/evidence authority"
            )


@chex.dataclass(frozen=True)
class GeneratedClassScrubDiagnostics:
    """Raw, threshold-free outcome of one proposed scrub transaction."""

    finite_state: Bool[Array, ""]
    dag_valid: Bool[Array, ""]
    counters_valid: Bool[Array, ""]
    generator_policy_ids_valid: Bool[Array, ""]
    state_valid: Bool[Array, ""]
    raw_prefix_mask_clear: Bool[Array, ""]
    nonempty_plan: Bool[Array, ""]
    active_descendant_closed: Bool[Array, ""]
    candidate_dependency_closed: Bool[Array, ""]
    filler_parameter_free: Bool[Array, ""]
    filler_raw_parents_valid: Bool[Array, ""]
    filler_distinct_from_masked_local_descriptors: Bool[Array, ""]
    proposed_dag_valid: Bool[Array, ""]
    masked_local_state_reset_exact: Bool[Array, ""]
    no_masked_old_local_descriptor_remains: Bool[Array, ""]
    plan_valid: Bool[Array, ""]
    commit_requested: Bool[Array, ""]
    committed: Bool[Array, ""]
    sham_noop: Bool[Array, ""]
    rolled_back: Bool[Array, ""]
    resource_shape_preserved: Bool[Array, ""]
    identity_head_lineage_only: Bool[Array, ""]
    behavioral_information_erasure_claimed: Bool[Array, ""]
    expanded_expression_absence_claimed: Bool[Array, ""]
    host_timing_canonicalization_required_for_jit_bit_equality: Bool[Array, ""]
    active_mask: Bool[Array, " n_features"]
    candidate_mask: Bool[Array, " n_candidates"]
    active_scrub_count: Int[Array, ""]
    candidate_scrub_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class GeneratedClassScrubResult:
    """State and diagnostics returned by the atomic scrub kernel."""

    state: CompositionalFeatureState
    diagnostics: GeneratedClassScrubDiagnostics


def _path_text(path: tuple[Any, ...]) -> str:
    names: list[str] = []
    for key in path:
        name = getattr(key, "name", None)
        if not isinstance(name, str):
            raise TypeError(f"unsupported compositional state path key: {key!r}")
        names.append(name)
    return ".".join(names)


def compositional_state_leaf_paths(
    state: CompositionalFeatureState,
) -> frozenset[str]:
    """Return concrete PyTree leaf paths after exact type validation."""

    if type(state) is not CompositionalFeatureState:
        raise TypeError("state must be an exact CompositionalFeatureState")
    paths = frozenset(
        _path_text(path) for path, _ in jax.tree_util.tree_flatten_with_path(state)[0]
    )
    if paths != COMPOSITIONAL_STATE_LEAF_PATHS:
        missing = sorted(COMPOSITIONAL_STATE_LEAF_PATHS - paths)
        extra = sorted(paths - COMPOSITIONAL_STATE_LEAF_PATHS)
        raise RuntimeError(
            "live CompositionalFeatureState leaves do not match the scrub partition; "
            f"missing={missing}, extra={extra}"
        )
    return paths


def persistent_compositional_state_nbytes(state: CompositionalFeatureState) -> int:
    """Count persistent array bytes, excluding the two host timing leaves."""

    if type(state) is not CompositionalFeatureState:
        raise TypeError("state must be an exact CompositionalFeatureState")
    total = 0
    for path, leaf in jax.tree_util.tree_flatten_with_path(state)[0]:
        if _path_text(path) in {"birth_timestamp", "uptime_s"}:
            continue
        if isinstance(leaf, Array) and jax.dtypes.issubdtype(  # type: ignore[attr-defined]
            leaf.dtype,
            jax.dtypes.prng_key,
        ):
            value = np.asarray(jax.random.key_data(leaf))
        else:
            value = np.asarray(leaf)
        total += int(value.nbytes)
    return total


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
    config: GeneratedClassScrubConfig,
) -> None:
    if type(state) is not CompositionalFeatureState:
        raise TypeError("state must be an exact CompositionalFeatureState")
    n = config.active_slots
    c = config.candidate_slots
    h = config.n_tasks
    if jnp.asarray(state.ops).shape != (n,):
        raise ValueError(
            "state active-slot width must match config.active_slots; "
            f"got {jnp.asarray(state.ops).shape}, expected {(n,)}"
        )
    active_int = ("ops", "parent_a", "parent_b", "depth", "ages", "feature_generator_policy")
    active_float = (
        "utilities",
        "utility_feature_trace",
        "utility_feature_energy_trace",
        "utility_signal_second_moment",
        "feature_score_energy_trace",
        "retention_slow_utilities",
    )
    for name in active_int:
        _require_array(getattr(state, name), name=f"state.{name}", shape=(n,), dtype=jnp.int32)
    for name in active_float:
        _require_array(getattr(state, name), name=f"state.{name}", shape=(n,), dtype=jnp.float32)
    _require_array(state.theta, name="state.theta", shape=(n, 2), dtype=jnp.float32)
    for name in ("output_weights", "utility_contribution_trace", "feature_score_residual_trace"):
        _require_array(getattr(state, name), name=f"state.{name}", shape=(h, n), dtype=jnp.float32)
    for name in ("output_bias", "utility_error_trace", "task_activity_ema"):
        _require_array(getattr(state, name), name=f"state.{name}", shape=(h,), dtype=jnp.float32)

    candidate_int = (
        "candidate_ops",
        "candidate_parent_a",
        "candidate_parent_b",
        "candidate_depth",
        "candidate_ages",
        "candidate_generator_policy",
    )
    candidate_float = (
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
    for name in candidate_int:
        _require_array(getattr(state, name), name=f"state.{name}", shape=(c,), dtype=jnp.int32)
    for name in candidate_float:
        _require_array(getattr(state, name), name=f"state.{name}", shape=(c,), dtype=jnp.float32)
    _require_array(
        state.candidate_theta,
        name="state.candidate_theta",
        shape=(c, 2),
        dtype=jnp.float32,
    )
    for name in (
        "candidate_output_weights",
        "candidate_utility_contribution_trace",
        "candidate_score_residual_trace",
    ):
        _require_array(getattr(state, name), name=f"state.{name}", shape=(h, c), dtype=jnp.float32)
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
        or not jax.dtypes.issubdtype(  # type: ignore[attr-defined]
            key_dtype,
            jax.dtypes.prng_key,
        )
    ):
        raise TypeError("state.key must be one scalar typed Threefry key")
    if str(jax.random.key_impl(state.key)) != "threefry2x32":
        raise TypeError("state.key must use the typed Threefry implementation")
    key_data = jax.random.key_data(state.key)
    if key_data.shape != (2,) or key_data.dtype != jnp.uint32:
        raise TypeError("state.key must expose exactly two uint32 Threefry words")

    generator = state.generator_resource_state
    if type(generator) is not GeneratorMetaResourceManagerState:
        raise TypeError(
            "state.generator_resource_state must be an exact GeneratorMetaResourceManagerState"
        )
    log_weights = jnp.asarray(generator.log_weights)
    if log_weights.ndim != 2:
        raise ValueError("state.generator_resource_state.log_weights must be rank two")
    generator_shape = log_weights.shape
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
    floating_values = (
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
    for value in floating_values:
        valid = valid & jnp.all(jnp.isfinite(jnp.asarray(value)))
    return valid


def _dag_valid(state: CompositionalFeatureState, feature_dim: int) -> Array:
    n = state.ops.shape[0]
    indices = jnp.arange(n, dtype=jnp.int32)
    op_valid = (state.ops >= OP_RAW) & (state.ops < NUM_OPS)
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
    slot_valid = op_valid & jnp.where(is_raw, raw_valid, binary_valid)

    prefix = jnp.arange(feature_dim, dtype=jnp.int32)
    raw_prefix_valid = (
        jnp.all(state.ops[:feature_dim] == OP_RAW)
        & jnp.all(state.parent_a[:feature_dim] == prefix)
        & jnp.all(state.parent_b[:feature_dim] == -1)
        & jnp.all(state.depth[:feature_dim] == 0)
    )
    raw_suffix_clear = jnp.all(state.ops[feature_dim:] != OP_RAW)

    c_is_raw = state.candidate_ops == OP_RAW
    c_op_valid = (state.candidate_ops >= OP_RAW) & (state.candidate_ops < NUM_OPS)
    c_safe_a = jnp.clip(state.candidate_parent_a, 0, n - 1)
    c_safe_b = jnp.clip(state.candidate_parent_b, 0, n - 1)
    c_raw_valid = (
        (state.candidate_parent_a >= 0)
        & (state.candidate_parent_a < feature_dim)
        & (state.candidate_parent_b == -1)
        & (state.candidate_depth == 0)
    )
    c_binary_valid = (
        (state.candidate_parent_a >= 0)
        & (state.candidate_parent_a < n)
        & (state.candidate_parent_b >= 0)
        & (state.candidate_parent_b < n)
        & (state.candidate_depth == jnp.maximum(state.depth[c_safe_a], state.depth[c_safe_b]) + 1)
    )
    candidate_valid = c_op_valid & jnp.where(c_is_raw, c_raw_valid, c_binary_valid)
    return raw_prefix_valid & raw_suffix_clear & jnp.all(slot_valid) & jnp.all(candidate_valid)


def _counters_valid(state: CompositionalFeatureState) -> Array:
    return (
        (state.step_count >= 0)
        & (state.generator_resource_state.step_count >= 0)
        & jnp.all(state.ages >= 0)
        & jnp.all(state.candidate_ages >= 0)
        & jnp.all(state.candidate_selector_action_counts >= 0.0)
        & jnp.all(state.generator_resource_state.action_counts >= 0.0)
        & (state.replacement_accumulator >= 0.0)
    )


def _generator_policy_ids_valid(state: CompositionalFeatureState) -> Array:
    """Validate every provenance id against the live generator policy width."""

    policy_count = state.generator_resource_state.log_weights.shape[1]
    return (
        jnp.asarray(policy_count > 0, dtype=jnp.bool_)
        & jnp.all(state.feature_generator_policy >= 0)
        & jnp.all(state.feature_generator_policy < policy_count)
        & jnp.all(state.candidate_generator_policy >= 0)
        & jnp.all(state.candidate_generator_policy < policy_count)
    )


def _descriptor_words(
    ops: Array,
    parent_a: Array,
    parent_b: Array,
    theta: Array,
    depth: Array,
) -> Array:
    """Encode local descriptors exactly, ignoring theta for parameter-free ops."""

    commutative = (ops == OP_SUM) | (ops == OP_PRODUCT)
    canonical_a = jnp.where(commutative, jnp.minimum(parent_a, parent_b), parent_a)
    canonical_b = jnp.where(commutative, jnp.maximum(parent_a, parent_b), parent_b)
    used_theta = jnp.where((ops == OP_TANH)[:, None], theta, jnp.zeros_like(theta))
    return jnp.stack(
        (
            jax.lax.bitcast_convert_type(ops, jnp.uint32),
            jax.lax.bitcast_convert_type(canonical_a, jnp.uint32),
            jax.lax.bitcast_convert_type(canonical_b, jnp.uint32),
            jax.lax.bitcast_convert_type(used_theta[:, 0], jnp.uint32),
            jax.lax.bitcast_convert_type(used_theta[:, 1], jnp.uint32),
            jax.lax.bitcast_convert_type(depth, jnp.uint32),
        ),
        axis=-1,
    )


def _build_proposal(
    state: CompositionalFeatureState,
    active_mask: Array,
    candidate_mask: Array,
    config: GeneratedClassScrubConfig,
) -> CompositionalFeatureState:
    active_column_mask = active_mask[None, :]
    candidate_column_mask = candidate_mask[None, :]
    correlation_mask = candidate_mask[:, None] | active_mask[None, :]
    filler_op = jnp.asarray(config.filler_op, dtype=jnp.int32)
    filler_a = jnp.asarray(config.filler_parent_a, dtype=jnp.int32)
    filler_b = jnp.asarray(config.filler_parent_b, dtype=jnp.int32)
    filler_depth = jnp.asarray(1, dtype=jnp.int32)

    return cast(
        CompositionalFeatureState,
        state.replace(  # type: ignore[attr-defined]
            ops=jnp.where(active_mask, filler_op, state.ops),
            parent_a=jnp.where(active_mask, filler_a, state.parent_a),
            parent_b=jnp.where(active_mask, filler_b, state.parent_b),
            theta=jnp.where(active_mask[:, None], jnp.zeros_like(state.theta), state.theta),
            depth=jnp.where(active_mask, filler_depth, state.depth),
            output_weights=jnp.where(
                active_column_mask,
                jnp.zeros_like(state.output_weights),
                state.output_weights,
            ),
            utilities=jnp.where(active_mask, 0.0, state.utilities),
            utility_contribution_trace=jnp.where(
                active_column_mask,
                jnp.zeros_like(state.utility_contribution_trace),
                state.utility_contribution_trace,
            ),
            utility_feature_trace=jnp.where(active_mask, 0.0, state.utility_feature_trace),
            utility_feature_energy_trace=jnp.where(
                active_mask,
                0.0,
                state.utility_feature_energy_trace,
            ),
            utility_signal_second_moment=jnp.where(
                active_mask,
                0.0,
                state.utility_signal_second_moment,
            ),
            feature_score_residual_trace=jnp.where(
                active_column_mask,
                jnp.zeros_like(state.feature_score_residual_trace),
                state.feature_score_residual_trace,
            ),
            feature_score_energy_trace=jnp.where(
                active_mask,
                0.0,
                state.feature_score_energy_trace,
            ),
            retention_slow_utilities=jnp.where(
                active_mask,
                0.0,
                state.retention_slow_utilities,
            ),
            ages=jnp.where(active_mask, 0, state.ages),
            feature_generator_policy=jnp.where(active_mask, 0, state.feature_generator_policy),
            candidate_ops=jnp.where(candidate_mask, filler_op, state.candidate_ops),
            candidate_parent_a=jnp.where(
                candidate_mask,
                filler_a,
                state.candidate_parent_a,
            ),
            candidate_parent_b=jnp.where(
                candidate_mask,
                filler_b,
                state.candidate_parent_b,
            ),
            candidate_theta=jnp.where(
                candidate_mask[:, None],
                jnp.zeros_like(state.candidate_theta),
                state.candidate_theta,
            ),
            candidate_depth=jnp.where(candidate_mask, filler_depth, state.candidate_depth),
            candidate_output_weights=jnp.where(
                candidate_column_mask,
                jnp.zeros_like(state.candidate_output_weights),
                state.candidate_output_weights,
            ),
            candidate_utilities=jnp.where(candidate_mask, 0.0, state.candidate_utilities),
            candidate_utility_contribution_trace=jnp.where(
                candidate_column_mask,
                jnp.zeros_like(state.candidate_utility_contribution_trace),
                state.candidate_utility_contribution_trace,
            ),
            candidate_utility_feature_trace=jnp.where(
                candidate_mask,
                0.0,
                state.candidate_utility_feature_trace,
            ),
            candidate_utility_feature_energy_trace=jnp.where(
                candidate_mask,
                0.0,
                state.candidate_utility_feature_energy_trace,
            ),
            candidate_utility_signal_second_moment=jnp.where(
                candidate_mask,
                0.0,
                state.candidate_utility_signal_second_moment,
            ),
            candidate_score_residual_trace=jnp.where(
                candidate_column_mask,
                jnp.zeros_like(state.candidate_score_residual_trace),
                state.candidate_score_residual_trace,
            ),
            candidate_score_energy_trace=jnp.where(
                candidate_mask,
                0.0,
                state.candidate_score_energy_trace,
            ),
            candidate_retention_slow_utilities=jnp.where(
                candidate_mask,
                0.0,
                state.candidate_retention_slow_utilities,
            ),
            candidate_active_correlation_trace=jnp.where(
                correlation_mask,
                jnp.zeros_like(state.candidate_active_correlation_trace),
                state.candidate_active_correlation_trace,
            ),
            candidate_ages=jnp.where(candidate_mask, 0, state.candidate_ages),
            candidate_selector_log_weights=jnp.where(
                candidate_mask,
                0.0,
                state.candidate_selector_log_weights,
            ),
            candidate_selector_cumulative_loss=jnp.where(
                candidate_mask,
                0.0,
                state.candidate_selector_cumulative_loss,
            ),
            candidate_selector_action_counts=jnp.where(
                candidate_mask,
                0.0,
                state.candidate_selector_action_counts,
            ),
            candidate_generator_policy=jnp.where(
                candidate_mask,
                0,
                state.candidate_generator_policy,
            ),
        ),
    )


def _masked_local_state_reset_exact(
    state: CompositionalFeatureState,
    active_mask: Array,
    candidate_mask: Array,
    config: GeneratedClassScrubConfig,
) -> Array:
    """Audit every masked local leaf, including exact positive-zero bits."""

    def float_zero_axis0(value: Array, mask: Array) -> Array:
        bits = jax.lax.bitcast_convert_type(value, jnp.uint32)
        return jnp.all(jnp.where(mask, bits == 0, True))

    def float_zero_axis1(value: Array, mask: Array) -> Array:
        bits = jax.lax.bitcast_convert_type(value, jnp.uint32)
        return jnp.all(jnp.where(mask[None, :], bits == 0, True))

    active_exact = (
        jnp.all(jnp.where(active_mask, state.ops == config.filler_op, True))
        & jnp.all(jnp.where(active_mask, state.parent_a == config.filler_parent_a, True))
        & jnp.all(jnp.where(active_mask, state.parent_b == config.filler_parent_b, True))
        & float_zero_axis0(state.theta[:, 0], active_mask)
        & float_zero_axis0(state.theta[:, 1], active_mask)
        & jnp.all(jnp.where(active_mask, state.depth == 1, True))
        & float_zero_axis1(state.output_weights, active_mask)
        & float_zero_axis0(state.utilities, active_mask)
        & float_zero_axis1(state.utility_contribution_trace, active_mask)
        & float_zero_axis0(state.utility_feature_trace, active_mask)
        & float_zero_axis0(state.utility_feature_energy_trace, active_mask)
        & float_zero_axis0(state.utility_signal_second_moment, active_mask)
        & float_zero_axis1(state.feature_score_residual_trace, active_mask)
        & float_zero_axis0(state.feature_score_energy_trace, active_mask)
        & float_zero_axis0(state.retention_slow_utilities, active_mask)
        & jnp.all(jnp.where(active_mask, state.ages == 0, True))
        & jnp.all(jnp.where(active_mask, state.feature_generator_policy == 0, True))
    )
    candidate_exact = (
        jnp.all(jnp.where(candidate_mask, state.candidate_ops == config.filler_op, True))
        & jnp.all(
            jnp.where(
                candidate_mask,
                state.candidate_parent_a == config.filler_parent_a,
                True,
            )
        )
        & jnp.all(
            jnp.where(
                candidate_mask,
                state.candidate_parent_b == config.filler_parent_b,
                True,
            )
        )
        & float_zero_axis0(state.candidate_theta[:, 0], candidate_mask)
        & float_zero_axis0(state.candidate_theta[:, 1], candidate_mask)
        & jnp.all(jnp.where(candidate_mask, state.candidate_depth == 1, True))
        & float_zero_axis1(state.candidate_output_weights, candidate_mask)
        & float_zero_axis0(state.candidate_utilities, candidate_mask)
        & float_zero_axis1(
            state.candidate_utility_contribution_trace,
            candidate_mask,
        )
        & float_zero_axis0(state.candidate_utility_feature_trace, candidate_mask)
        & float_zero_axis0(
            state.candidate_utility_feature_energy_trace,
            candidate_mask,
        )
        & float_zero_axis0(
            state.candidate_utility_signal_second_moment,
            candidate_mask,
        )
        & float_zero_axis1(state.candidate_score_residual_trace, candidate_mask)
        & float_zero_axis0(state.candidate_score_energy_trace, candidate_mask)
        & float_zero_axis0(
            state.candidate_retention_slow_utilities,
            candidate_mask,
        )
        & jnp.all(jnp.where(candidate_mask, state.candidate_ages == 0, True))
        & float_zero_axis0(state.candidate_selector_log_weights, candidate_mask)
        & float_zero_axis0(
            state.candidate_selector_cumulative_loss,
            candidate_mask,
        )
        & float_zero_axis0(
            state.candidate_selector_action_counts,
            candidate_mask,
        )
        & jnp.all(jnp.where(candidate_mask, state.candidate_generator_policy == 0, True))
    )
    correlation_bits = jax.lax.bitcast_convert_type(
        state.candidate_active_correlation_trace,
        jnp.uint32,
    )
    correlation_mask = candidate_mask[:, None] | active_mask[None, :]
    correlation_exact = jnp.all(jnp.where(correlation_mask, correlation_bits == 0, True))
    return active_exact & candidate_exact & correlation_exact


def _select_committed_state(
    committed: Array,
    proposal: CompositionalFeatureState,
    original: CompositionalFeatureState,
) -> CompositionalFeatureState:
    """Select every mutable leaf while preserving shared and host metadata."""

    changes = {
        name: jnp.where(committed, getattr(proposal, name), getattr(original, name))
        for name in (
            "ops",
            "parent_a",
            "parent_b",
            "theta",
            "depth",
            "output_weights",
            "utilities",
            "utility_contribution_trace",
            "utility_feature_trace",
            "utility_feature_energy_trace",
            "utility_signal_second_moment",
            "feature_score_residual_trace",
            "feature_score_energy_trace",
            "retention_slow_utilities",
            "ages",
            "candidate_ops",
            "candidate_parent_a",
            "candidate_parent_b",
            "candidate_theta",
            "candidate_depth",
            "candidate_output_weights",
            "candidate_utilities",
            "candidate_utility_contribution_trace",
            "candidate_utility_feature_trace",
            "candidate_utility_feature_energy_trace",
            "candidate_utility_signal_second_moment",
            "candidate_score_residual_trace",
            "candidate_score_energy_trace",
            "candidate_retention_slow_utilities",
            "candidate_active_correlation_trace",
            "candidate_ages",
            "candidate_selector_log_weights",
            "candidate_selector_cumulative_loss",
            "candidate_selector_action_counts",
            "feature_generator_policy",
            "candidate_generator_policy",
        )
    }
    return cast(
        CompositionalFeatureState,
        original.replace(**changes),  # type: ignore[attr-defined]
    )


def scrub_compositional_feature_state(
    state: CompositionalFeatureState,
    active_mask: Array,
    candidate_mask: Array,
    commit: Array,
    *,
    config: GeneratedClassScrubConfig,
) -> GeneratedClassScrubResult:
    """Plan and optionally commit one fixed-shape identity-lineage scrub.

    ``commit=False`` is the matched sham: it performs the same validation,
    descriptor comparisons, proposal construction, and postcondition checks,
    then returns the input learner state bit-exactly.  Any invalid plan also
    returns the full input state bit-exactly.  Shape and dtype mismatches are
    static programming errors and fail before tracing because no same-shape
    atomic result can represent them.

    Descriptor postconditions are local only.  Callers requiring absence from
    expanded expression trees must establish that separately with their own
    expression compiler; this function has no target-expression knowledge.
    Canonical array-valued host timing leaves are required for eager/JIT
    bit-equality because the transaction intentionally preserves those leaves.
    """

    if type(config) is not GeneratedClassScrubConfig:
        raise TypeError("config must be an exact GeneratedClassScrubConfig")
    _require_static_state_contract(state, config)
    active = _require_array(
        active_mask,
        name="active_mask",
        shape=(config.active_slots,),
        dtype=jnp.bool_,
    )
    candidate = _require_array(
        candidate_mask,
        name="candidate_mask",
        shape=(config.candidate_slots,),
        dtype=jnp.bool_,
    )
    requested = _require_array(commit, name="commit", shape=(), dtype=jnp.bool_)

    finite_state = _finite_state(state)
    dag_valid = _dag_valid(state, config.feature_dim)
    counters_valid = _counters_valid(state)
    generator_policy_ids_valid = _generator_policy_ids_valid(state)
    state_valid = finite_state & dag_valid & counters_valid & generator_policy_ids_valid
    raw_prefix_mask_clear = ~jnp.any(active[: config.feature_dim])
    nonempty_plan = jnp.any(active) | jnp.any(candidate)

    n = config.active_slots
    safe_a = jnp.clip(state.parent_a, 0, n - 1)
    safe_b = jnp.clip(state.parent_b, 0, n - 1)
    active_depends_on_masked = (state.ops != OP_RAW) & (active[safe_a] | active[safe_b])
    active_descendant_closed = ~jnp.any((~active) & active_depends_on_masked)

    c_safe_a = jnp.clip(state.candidate_parent_a, 0, n - 1)
    c_safe_b = jnp.clip(state.candidate_parent_b, 0, n - 1)
    candidate_depends_on_masked = (state.candidate_ops != OP_RAW) & (
        active[c_safe_a] | active[c_safe_b]
    )
    candidate_dependency_closed = ~jnp.any((~candidate) & candidate_depends_on_masked)

    proposal = _build_proposal(state, active, candidate, config)
    proposed_dag_valid = _dag_valid(proposal, config.feature_dim)
    masked_local_state_reset_exact = _masked_local_state_reset_exact(
        proposal,
        active,
        candidate,
        config,
    )
    old_descriptors = jnp.concatenate(
        (
            _descriptor_words(
                state.ops,
                state.parent_a,
                state.parent_b,
                state.theta,
                state.depth,
            ),
            _descriptor_words(
                state.candidate_ops,
                state.candidate_parent_a,
                state.candidate_parent_b,
                state.candidate_theta,
                state.candidate_depth,
            ),
        ),
        axis=0,
    )
    proposed_descriptors = jnp.concatenate(
        (
            _descriptor_words(
                proposal.ops,
                proposal.parent_a,
                proposal.parent_b,
                proposal.theta,
                proposal.depth,
            ),
            _descriptor_words(
                proposal.candidate_ops,
                proposal.candidate_parent_a,
                proposal.candidate_parent_b,
                proposal.candidate_theta,
                proposal.candidate_depth,
            ),
        ),
        axis=0,
    )
    masked_descriptors = jnp.concatenate((active, candidate), axis=0)
    descriptor_matches = jnp.all(
        proposed_descriptors[:, None, :] == old_descriptors[None, :, :],
        axis=-1,
    )
    no_masked_old_local_descriptor_remains = ~jnp.any(
        descriptor_matches & masked_descriptors[None, :]
    )

    filler_descriptor = _descriptor_words(
        jnp.asarray((config.filler_op,), dtype=jnp.int32),
        jnp.asarray((config.filler_parent_a,), dtype=jnp.int32),
        jnp.asarray((config.filler_parent_b,), dtype=jnp.int32),
        jnp.zeros((1, 2), dtype=jnp.float32),
        jnp.asarray((1,), dtype=jnp.int32),
    )[0]
    filler_matches_old = jnp.all(
        old_descriptors == filler_descriptor[None, :],
        axis=-1,
    )
    filler_distinct_from_masked_local_descriptors = ~jnp.any(
        filler_matches_old & masked_descriptors
    )
    filler_parameter_free = jnp.asarray(
        config.filler_op in {OP_PRODUCT, OP_SUM, OP_GATED},
        dtype=jnp.bool_,
    )
    filler_raw_parents_valid = jnp.asarray(
        0 <= config.filler_parent_a < config.feature_dim
        and 0 <= config.filler_parent_b < config.feature_dim,
        dtype=jnp.bool_,
    )
    plan_valid = (
        state_valid
        & raw_prefix_mask_clear
        & nonempty_plan
        & active_descendant_closed
        & candidate_dependency_closed
        & filler_parameter_free
        & filler_raw_parents_valid
        & filler_distinct_from_masked_local_descriptors
        & proposed_dag_valid
        & masked_local_state_reset_exact
        & no_masked_old_local_descriptor_remains
    )
    committed = requested & plan_valid
    returned_state = _select_committed_state(committed, proposal, state)

    diagnostics = GeneratedClassScrubDiagnostics(  # type: ignore[call-arg]
        finite_state=finite_state,
        dag_valid=dag_valid,
        counters_valid=counters_valid,
        generator_policy_ids_valid=generator_policy_ids_valid,
        state_valid=state_valid,
        raw_prefix_mask_clear=raw_prefix_mask_clear,
        nonempty_plan=nonempty_plan,
        active_descendant_closed=active_descendant_closed,
        candidate_dependency_closed=candidate_dependency_closed,
        filler_parameter_free=filler_parameter_free,
        filler_raw_parents_valid=filler_raw_parents_valid,
        filler_distinct_from_masked_local_descriptors=(
            filler_distinct_from_masked_local_descriptors
        ),
        proposed_dag_valid=proposed_dag_valid,
        masked_local_state_reset_exact=masked_local_state_reset_exact,
        no_masked_old_local_descriptor_remains=no_masked_old_local_descriptor_remains,
        plan_valid=plan_valid,
        commit_requested=requested,
        committed=committed,
        sham_noop=plan_valid & (~requested),
        rolled_back=requested & (~plan_valid),
        resource_shape_preserved=jnp.asarray(True, dtype=jnp.bool_),
        identity_head_lineage_only=jnp.asarray(True, dtype=jnp.bool_),
        behavioral_information_erasure_claimed=jnp.asarray(False, dtype=jnp.bool_),
        expanded_expression_absence_claimed=jnp.asarray(False, dtype=jnp.bool_),
        host_timing_canonicalization_required_for_jit_bit_equality=jnp.asarray(
            True,
            dtype=jnp.bool_,
        ),
        active_mask=active,
        candidate_mask=candidate,
        active_scrub_count=jnp.sum(active, dtype=jnp.int32),
        candidate_scrub_count=jnp.sum(candidate, dtype=jnp.int32),
    )
    return GeneratedClassScrubResult(  # type: ignore[call-arg]
        state=returned_state,
        diagnostics=diagnostics,
    )


__all__ = [
    "ACTIVE_MASKED_LEAF_PATHS",
    "CANDIDATE_MASKED_LEAF_PATHS",
    "COMPOSITIONAL_STATE_LEAF_PATHS",
    "CROSS_MASKED_LEAF_PATHS",
    "GENERATED_CLASS_SCRUB_SCHEMA",
    "GENERATED_CLASS_SCRUB_STATUS",
    "PRESERVED_LEAF_PATHS",
    "GeneratedClassScrubConfig",
    "GeneratedClassScrubDiagnostics",
    "GeneratedClassScrubResult",
    "compositional_state_leaf_paths",
    "persistent_compositional_state_nbytes",
    "scrub_compositional_feature_state",
]
