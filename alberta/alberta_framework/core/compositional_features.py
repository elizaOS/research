"""Compositional feature discovery for Alberta Plan Step 2.

This module implements ``CompositionalFeatureLearner``, a fixed-budget
feature learner whose features form a directed acyclic graph (DAG) of
compositional operations.  Where ``FixedBudgetFeatureLearner`` only constructs
features that are direct functions of the raw input, and
``FixedBudgetInteractionLearner`` only forms pairwise products of raw inputs,
this learner explicitly composes features OF features.

The Alberta Plan Step 2 calls for "new features made by combining existing
features."  Each feature slot here records an op type (raw, product, sum,
tanh of a learned linear combination, or gated multiplication), two parent
indices, a small per-feature parameter vector, a topological depth, and the
familiar utility/age machinery used elsewhere in the framework.

The forward pass uses ``jax.lax.scan`` over slots in topological order so that
parents are always evaluated before children.  Generation enforces
``depth[new] > max(depth[parent_a], depth[parent_b])``, and replacement
cascades through descendants of a replaced slot to keep the DAG well-formed
under JIT compilation.
"""

import dataclasses
import functools
import math
import time
from collections.abc import Mapping
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Bool, Float, Int, PRNGKeyArray, UInt

from alberta_framework.core.future_utility import (
    contribution_trace_output_loss_reduction,
    normalize_future_utility_signal,
    trace_output_loss_reduction,
)
from alberta_framework.core.resource_manager import (
    GeneratorMetaResourceManager,
    GeneratorMetaResourceManagerState,
)

OP_RAW = 0
OP_PRODUCT = 1
OP_SUM = 2
OP_TANH = 3
OP_GATED = 4

NUM_OPS = 5

GENERATION_UNIFORM = "uniform"
GENERATION_UTILITY = "utility"
GENERATION_MUTATION = "mutation"
GENERATION_RESIDUAL_IMPRINT = "residual_imprint"
GENERATION_RECURSIVE_PRODUCT = "recursive_product"
GENERATION_ROBUST_RECURSIVE = "robust_recursive"
GENERATION_DOVETAIL_PRODUCT_COVERAGE = "dovetail_product_coverage"

PARENT_MODE_UNIFORM = 0
PARENT_MODE_UTILITY = 1
PARENT_MODE_MUTATION = 2
PARENT_MODE_RESIDUAL_IMPRINT = 3

# Generator meta-policy tables (used when ``learn_generator_resources=True``).
# The tuples below are column-aligned: entry ``i`` of every table describes one
# policy, pairing an op type and parent-selection mode with multipliers that
# scale the learner's base replacement rate, promotion margin, candidate
# minimum age, and residual-imprint scale. Policies are ordered conservative to
# aggressive: "safe" halves replacement, demands a 25% larger promotion margin,
# extends candidate trials 1.5x, and never imprints; "aggressive" doubles
# replacement, relaxes margin and trial age, and fully imprints the residual.
DEFAULT_GENERATOR_META_POLICY_NAMES = (
    "random_product_safe",
    "mutation_product_nominal",
    "residual_tanh",
    "residual_gated_aggressive",
)
DEFAULT_GENERATOR_META_OP_IDS = (
    OP_PRODUCT,
    OP_PRODUCT,
    OP_TANH,
    OP_GATED,
)
DEFAULT_GENERATOR_META_PARENT_MODES = (
    PARENT_MODE_UNIFORM,
    PARENT_MODE_MUTATION,
    PARENT_MODE_RESIDUAL_IMPRINT,
    PARENT_MODE_RESIDUAL_IMPRINT,
)
DEFAULT_GENERATOR_META_REPLACEMENT_MULTIPLIERS = (0.5, 1.0, 1.0, 2.0)
DEFAULT_GENERATOR_META_PROMOTION_MARGIN_MULTIPLIERS = (1.25, 1.0, 0.9, 0.75)
DEFAULT_GENERATOR_META_CANDIDATE_MIN_AGE_MULTIPLIERS = (1.5, 1.0, 0.75, 0.5)
DEFAULT_GENERATOR_META_IMPRINT_SCALES = (0.0, 0.25, 1.0, 1.0)

# Generator-policy provenance arrays keep their existing non-negative,
# fixed-width representation even when the optional meta-resource learner is
# disabled.  In that mode zero is a deterministic placeholder only; callers
# must consult ``learn_generator_resources`` before interpreting it as a
# sampled policy identity.
FIXED_GENERATOR_POLICY_PLACEHOLDER = 0

# Stable, named domains below separate a fresh composition proposal, descendant
# refills that may follow its application, and corrective candidate
# regeneration after an active parent becomes too deep.  The integer tags are
# big-endian ASCII words and are part of the public deterministic key-derivation
# contract.
COMPOSITIONAL_CURATION_PROPOSAL_CHANNEL = 0x50524F50
COMPOSITIONAL_CURATION_CASCADE_CHANNEL = 0x43415343
COMPOSITIONAL_CURATION_OVERDEPTH_REGENERATION_CHANNEL = 0x4F445247

# Fixed integer tags used by ``CompositionalCurationTrace``.  ``NONE`` is a
# sentinel rather than a bank, while active and candidate banks are distinct
# identity namespaces for runner-side lifecycle accounting.
CURATION_DESTINATION_NONE = -1
CURATION_DESTINATION_ACTIVE = 0
CURATION_DESTINATION_CANDIDATE = 1

PROMOTION_SCALED_CANDIDATE = "scaled_candidate"
PROMOTION_BLEND = "blend"

CANDIDATE_SELECTOR_LEGACY = "legacy"
CANDIDATE_SELECTOR_HEDGE = "hedge"
CANDIDATE_SELECTOR_EXP3 = "exp3"

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1
_UINT16_BASE = 2**16
_UINT16_MASK = _UINT16_BASE - 1


def _saturating_nonnegative_int32_increment(value: Array) -> Array:
    """Advance non-negative int32 telemetry without signed wraparound."""

    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    counter = jnp.asarray(value, dtype=jnp.int32)
    return jnp.minimum(jnp.maximum(counter, 0), maximum - 1) + 1


def _checked_step_words_increment(
    step_words: Array,
) -> tuple[UInt[Array, " 2"], Bool[Array, ""]]:
    """Increment a big-endian uint32 pair, refusing the all-ones terminal value."""

    if getattr(step_words, "shape", None) != (2,):
        raise ValueError("step_words must have shape (2,)")
    if getattr(step_words, "dtype", None) != jnp.dtype(jnp.uint32):
        raise TypeError("step_words must have dtype uint32")
    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    capacity_available = ~jnp.all(step_words == maximum)
    one = jnp.asarray(1, dtype=jnp.uint32)
    low = step_words[1] + one
    carry = (low == jnp.asarray(0, dtype=jnp.uint32)).astype(jnp.uint32)
    high = step_words[0] + carry
    candidate = jnp.stack((high, low))
    return (
        jnp.where(capacity_available, candidate, step_words).astype(jnp.uint32),
        capacity_available,
    )


def _advance_replacement_phase(
    phase: Array,
    replacement_interval: int,
) -> tuple[Int[Array, ""], Bool[Array, ""]]:
    """Advance the bounded fixed-resource clock and report a due curation."""

    if replacement_interval == 0:
        zero = jnp.asarray(0, dtype=jnp.int32)
        return zero, jnp.asarray(False, dtype=jnp.bool_)
    last = jnp.asarray(replacement_interval - 1, dtype=jnp.int32)
    counter = jnp.asarray(phase, dtype=jnp.int32)
    next_phase = jnp.where(counter == last, 0, counter + 1).astype(jnp.int32)
    return next_phase, next_phase == 0


def _dynamic_bool_scalar(value: Array | bool, *, name: str) -> Bool[Array, ""]:
    """Require one dynamic boolean scalar without accepting integer coercion."""

    scalar = jnp.asarray(value)
    if getattr(scalar, "shape", None) != ():
        raise ValueError(f"{name} must be a scalar")
    if getattr(scalar, "dtype", None) != jnp.dtype(jnp.bool_):
        raise TypeError(f"{name} must have dtype bool")
    return scalar


def _dovetail_product_coverage_cycle(
    *,
    n_features: int,
    feature_dim: int,
) -> int:
    """Return the finite raw-pair/extension dovetail cycle length.

    Even cursors enumerate distinct raw pairs. Odd cursors enumerate every
    composed-slot/raw-parent extension blueprint. The least-common-multiple
    cycle covers both finite spaces without adding persistent cursor state.
    """

    if type(n_features) is not int or type(feature_dim) is not int:
        raise TypeError("dovetail dimensions must be exact Python integers")
    if feature_dim < 2:
        raise ValueError("dovetail product coverage requires feature_dim >= 2")
    if n_features <= feature_dim:
        raise ValueError(
            "dovetail product coverage requires at least one composed slot"
        )
    raw_pair_count = feature_dim * (feature_dim - 1) // 2
    extension_count = (n_features - feature_dim) * feature_dim
    return 2 * math.lcm(raw_pair_count, extension_count)


def _step_words_mod_uint16_limb(step_words: Array, modulus: int) -> Array:
    """Reduce one uint64-as-two-uint32 counter without enabling uint64.

    ``modulus <= 2**16`` ensures every ``remainder * 2**16 + limb``
    intermediate is representable exactly in uint32, including the maximum
    ``2**32 - 1`` value.
    """

    if getattr(step_words, "shape", None) != (2,):
        raise ValueError("step_words must have shape (2,)")
    if getattr(step_words, "dtype", None) != jnp.dtype(jnp.uint32):
        raise TypeError("step_words must have dtype uint32")
    if type(modulus) is not int or not 1 <= modulus <= _UINT16_BASE:
        raise ValueError("16-bit-limb modulus must be an exact integer in [1, 65536]")
    shift = jnp.asarray(16, dtype=jnp.uint32)
    mask = jnp.asarray(_UINT16_MASK, dtype=jnp.uint32)
    limbs = (
        step_words[0] >> shift,
        step_words[0] & mask,
        step_words[1] >> shift,
        step_words[1] & mask,
    )
    remainder = jnp.asarray(0, dtype=jnp.uint32)
    base = jnp.asarray(_UINT16_BASE, dtype=jnp.uint32)
    divisor = jnp.asarray(modulus, dtype=jnp.uint32)
    for limb in limbs:
        remainder = (remainder * base + limb) % divisor
    return remainder


def _dovetail_product_coverage_cursor(
    pre_step_words: Array,
    *,
    replacement_interval: int,
    n_features: int,
    feature_dim: int,
) -> Array:
    """Derive the zero-based curation cursor from exact pre-step identity.

    At fixed cadence ``R``, the k-th curation sees authoritative pre-update
    lifetime ``k*R - 1``. Reducing that identity modulo ``R * cycle`` and
    integer-dividing by ``R`` yields cursor ``(k - 1) % cycle``. Using the
    pre-step identity makes the first curation cursor zero and preserves the
    schedule through uint32-low carry and saturated int32 telemetry.
    """

    if type(replacement_interval) is not int or replacement_interval < 1:
        raise ValueError("dovetail coverage requires a positive fixed replacement interval")
    cycle = _dovetail_product_coverage_cycle(
        n_features=n_features,
        feature_dim=feature_dim,
    )
    modulus = replacement_interval * cycle
    if modulus > _UINT16_BASE:
        raise ValueError(
            "16-bit-limb modulus replacement_interval * cycle must be <= 65536"
        )
    within_cycle = _step_words_mod_uint16_limb(pre_step_words, modulus)
    return (within_cycle // jnp.asarray(replacement_interval, dtype=jnp.uint32)).astype(
        jnp.int32
    )


def compositional_curation_keys(curation_root_key: Array) -> tuple[Array, Array]:
    """Derive disjoint proposal and cascade keys from one curation root.

    The returned keys preserve the root key's PRNG implementation.  This
    helper assigns stable domains only; callers remain responsible for giving
    every curation event a fresh root key.
    """

    proposal_key = jr.fold_in(
        curation_root_key,
        jnp.uint32(COMPOSITIONAL_CURATION_PROPOSAL_CHANNEL),
    )
    cascade_key = jr.fold_in(
        curation_root_key,
        jnp.uint32(COMPOSITIONAL_CURATION_CASCADE_CHANNEL),
    )
    return proposal_key, cascade_key


@chex.dataclass(frozen=True)
class CompositionalFeatureState:
    """State for ``CompositionalFeatureLearner``.

    Slots ``[0, feature_dim)`` are reserved for ``OP_RAW`` features that
    expose individual raw observation entries.  The remaining active slots
    hold composed features whose parent indices refer to earlier slots.
    Candidates mirror this structure with their own bank.

    Attributes:
        key: PRNG key for generation/replacement decisions.
        ops: Op type per active slot.
        parent_a: First parent index (raw-input index for ``OP_RAW``,
            otherwise feature slot index strictly less than ``i``).
        parent_b: Second parent index (``-1`` for ``OP_RAW``, else
            feature slot index strictly less than ``i``).
        theta: Per-feature parameter vector of length two used only by
            ``OP_TANH``. ``OP_GATED`` is the fixed ordered operation
            ``value_a * sigmoid(value_b)`` and does not use ``theta``.
        depth: Topological depth (raw inputs at depth 0).
        output_weights: Output head weights, shape ``(n_tasks, n_features)``.
        output_bias: Output head biases.
        utilities: EMA utility per active slot.
        utility_contribution_trace: Discounted ``error * feature`` trace for
            TD(lambda)-style future-utility estimates.
        utility_error_trace: Discounted residual trace for temporally extended
            marginal future-utility estimates.
        utility_feature_trace: Discounted active feature-value trace.
        utility_feature_energy_trace: Discounted active squared-feature trace.
        utility_signal_second_moment: Online second moment for optional
            uncertainty normalization.
        feature_score_residual_trace: Discounted ``error * feature`` trace
            used by opt-in matching-pursuit candidate scoring.
        feature_score_energy_trace: Discounted feature-energy trace used by
            opt-in matching-pursuit candidate scoring.
        retention_slow_utilities: Optional slow utility EMA used by opt-in
            hysteretic replacement. Disabled configurations leave it at zero.
        task_activity_ema: Per-task activity frequency for rare-task credit.
        ages: Age in steps per active slot.
        candidate_*: Candidate slot bank with the same fields.
        candidate_utility_contribution_trace: Discounted candidate
            ``error * feature`` trace.
        candidate_utility_feature_trace: Discounted candidate value trace.
        candidate_utility_feature_energy_trace: Discounted candidate squared
            value trace.
        candidate_utility_signal_second_moment: Online second moment for
            optional candidate uncertainty normalization.
        candidate_score_residual_trace: Discounted candidate
            ``error * feature`` trace used by opt-in matching-pursuit scoring.
        candidate_score_energy_trace: Discounted candidate feature-energy
            trace used by opt-in matching-pursuit scoring.
        candidate_retention_slow_utilities: Optional candidate slow utility EMA
            used by opt-in hysteretic promotion/probation.
        candidate_active_correlation_trace: Discounted cross-feature trace
            used to penalize candidates that duplicate active features.
        candidate_selector_*: Optional finite-candidate selector state used
            only when ``candidate_selector`` is not ``"legacy"``.  The default
            promote heuristic ignores these fields.
        feature_generator_policy: Meta-resource policy that created each
            active feature slot.  When generator-resource learning is
            disabled, entries use ``FIXED_GENERATOR_POLICY_PLACEHOLDER`` and
            are not sampled-policy identities.
        candidate_generator_policy: Meta-resource policy that created each
            candidate slot, with the same deterministic-placeholder semantics
            when generator-resource learning is disabled.
        generator_resource_state: Contextual policy-allocation state for
            generator-internal choices.
        replacement_accumulator: Fractional replacement clock used when the
            learned policy controls replacement rate.
        step_count: Saturating int32 compatibility telemetry for the number of
            committed updates.
        step_words: Exact big-endian ``[high, low]`` uint32 lifetime counter.
            The all-ones value is terminal and is never wrapped or committed.
        replacement_phase: Bounded phase for fixed-resource curation.  A phase
            of zero is the just-fired boundary; learned-resource configurations
            preserve the field without consulting it.
        birth_timestamp: Wall-clock initialization time.
        uptime_s: Cumulative seconds spent inside scan loops.
    """

    key: PRNGKeyArray
    ops: Int[Array, " n_features"]
    parent_a: Int[Array, " n_features"]
    parent_b: Int[Array, " n_features"]
    theta: Float[Array, "n_features 2"]
    depth: Int[Array, " n_features"]
    output_weights: Float[Array, "n_tasks n_features"]
    output_bias: Float[Array, " n_tasks"]
    utilities: Float[Array, " n_features"]
    utility_contribution_trace: Float[Array, "n_tasks n_features"]
    utility_error_trace: Float[Array, " n_tasks"]
    utility_feature_trace: Float[Array, " n_features"]
    utility_feature_energy_trace: Float[Array, " n_features"]
    utility_signal_second_moment: Float[Array, " n_features"]
    feature_score_residual_trace: Float[Array, "n_tasks n_features"]
    feature_score_energy_trace: Float[Array, " n_features"]
    retention_slow_utilities: Float[Array, " n_features"]
    task_activity_ema: Float[Array, " n_tasks"]
    ages: Int[Array, " n_features"]
    candidate_ops: Int[Array, " n_candidates"]
    candidate_parent_a: Int[Array, " n_candidates"]
    candidate_parent_b: Int[Array, " n_candidates"]
    candidate_theta: Float[Array, "n_candidates 2"]
    candidate_depth: Int[Array, " n_candidates"]
    candidate_output_weights: Float[Array, "n_tasks n_candidates"]
    candidate_utilities: Float[Array, " n_candidates"]
    candidate_utility_contribution_trace: Float[Array, "n_tasks n_candidates"]
    candidate_utility_feature_trace: Float[Array, " n_candidates"]
    candidate_utility_feature_energy_trace: Float[Array, " n_candidates"]
    candidate_utility_signal_second_moment: Float[Array, " n_candidates"]
    candidate_score_residual_trace: Float[Array, "n_tasks n_candidates"]
    candidate_score_energy_trace: Float[Array, " n_candidates"]
    candidate_retention_slow_utilities: Float[Array, " n_candidates"]
    candidate_active_correlation_trace: Float[Array, "n_candidates n_features"]
    candidate_ages: Int[Array, " n_candidates"]
    candidate_selector_log_weights: Float[Array, " n_candidates"]
    candidate_selector_cumulative_loss: Float[Array, " n_candidates"]
    candidate_selector_action_counts: Float[Array, " n_candidates"]
    feature_generator_policy: Int[Array, " n_features"]
    candidate_generator_policy: Int[Array, " n_candidates"]
    generator_resource_state: GeneratorMetaResourceManagerState
    replacement_accumulator: Float[Array, ""]
    step_count: Int[Array, ""]
    step_words: UInt[Array, " 2"]
    replacement_phase: Int[Array, ""]
    birth_timestamp: float = 0.0
    uptime_s: float = 0.0


def _compositional_counter_state_valid(
    state: CompositionalFeatureState,
    replacement_interval: int,
) -> Bool[Array, ""]:
    """Validate the exact/telemetry counter relation and bounded age clocks."""

    if getattr(state.step_words, "shape", None) != (2,):
        raise ValueError("state.step_words must have shape (2,)")
    if getattr(state.step_words, "dtype", None) != jnp.dtype(jnp.uint32):
        raise TypeError("state.step_words must have dtype uint32")
    maximum_i32 = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    maximum_u32 = jnp.asarray(_INT32_MAX, dtype=jnp.uint32)
    high = state.step_words[0]
    low = state.step_words[1]
    exact_step_fits_telemetry = (high == 0) & (low <= maximum_u32)
    expected_telemetry = jnp.where(
        exact_step_fits_telemetry,
        low.astype(jnp.int32),
        maximum_i32,
    )
    if replacement_interval == 0:
        phase_valid = state.replacement_phase == 0
    else:
        phase_valid = (state.replacement_phase >= 0) & (
            state.replacement_phase < replacement_interval
        )
    return (
        (state.step_count >= 0)
        & (state.step_count == expected_telemetry)
        & phase_valid
        & jnp.all(state.ages >= 0)
        & jnp.all(state.candidate_ages >= 0)
    )


def migrate_legacy_compositional_feature_state(
    legacy_state: Mapping[str, Any],
    *,
    replacement_interval: int,
) -> CompositionalFeatureState:
    """Migrate one pre-counter-repair state mapping without inventing history.

    This host-only helper accepts the exact old dataclass field mapping.  A
    negative counter indicates prior signed wrap, while ``INT32_MAX`` is
    ambiguous between an exact terminal event and prior saturation; both are
    rejected.  Callers must begin a fresh authenticated lifecycle namespace in
    those cases.
    """

    if not isinstance(legacy_state, Mapping):
        raise TypeError("legacy_state must be a mapping")
    if type(replacement_interval) is not int:
        raise TypeError("replacement_interval must be an exact Python integer")
    if not 0 <= replacement_interval <= _INT32_MAX:
        raise ValueError("replacement_interval must be inside [0, int32_max]")
    current_fields = {
        field.name
        for field in dataclasses.fields(
            CompositionalFeatureState  # type: ignore[arg-type]
        )
    }
    legacy_fields = current_fields - {"step_words", "replacement_phase"}
    supplied_fields = set(legacy_state)
    if supplied_fields != legacy_fields:
        missing = sorted(legacy_fields - supplied_fields)
        extra = sorted(supplied_fields - legacy_fields)
        raise ValueError(
            f"legacy state field manifest is not exact; missing={missing}, extra={extra}"
        )

    step_array = jnp.asarray(legacy_state["step_count"])
    if step_array.shape != () or step_array.dtype != jnp.dtype(jnp.int32):
        raise TypeError("legacy step_count must be a scalar int32 array")
    step = int(step_array)
    if step < 0:
        raise ValueError("negative legacy step_count indicates signed wrap")
    if step >= _INT32_MAX:
        raise ValueError("saturated legacy step_count has ambiguous lifetime history")
    for name in ("ages", "candidate_ages"):
        value = jnp.asarray(legacy_state[name])
        if value.dtype != jnp.dtype(jnp.int32):
            raise TypeError(f"legacy {name} must have dtype int32")
        if bool(jnp.any(value < 0)):
            raise ValueError(f"negative legacy {name} indicates signed wrap")
    manager_step = jnp.asarray(legacy_state["generator_resource_state"].step_count)
    if manager_step.shape != () or manager_step.dtype != jnp.dtype(jnp.int32):
        raise TypeError("legacy generator resource step_count must be scalar int32")
    if int(manager_step) < 0:
        raise ValueError("negative generator resource step_count indicates signed wrap")

    migrated = dict(legacy_state)
    migrated["step_words"] = jnp.asarray((0, step), dtype=jnp.uint32)
    migrated["replacement_phase"] = jnp.asarray(
        0 if replacement_interval == 0 else step % replacement_interval,
        dtype=jnp.int32,
    )
    return CompositionalFeatureState(**migrated)


@chex.dataclass(frozen=True)
class CompositionalCurationTrace:
    """Raw fixed-shape facts emitted by one curation decision.

    This trace is public learner output for runner-side birth-ledger
    construction.  It records actual applied events and exact descriptors; it
    is not authenticated lifecycle evidence by itself.  Descriptor arrays
    retain ``parent_a`` even for ``OP_RAW`` so raw-input identity is explicit.
    Masks select meaningful rows in full-bank descriptor snapshots.
    """

    pre_step: Int[Array, ""]
    post_step: Int[Array, ""]
    pre_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    pre_replacement_phase: Int[Array, ""]
    post_replacement_phase: Int[Array, ""]
    lifetime_counter_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    decision_key: PRNGKeyArray
    curation_key: PRNGKeyArray
    proposal_key: PRNGKeyArray
    cascade_key: PRNGKeyArray
    candidate_overdepth_regeneration_key: PRNGKeyArray
    should_try_replace: Bool[Array, ""]
    has_event: Bool[Array, ""]
    generator_policy_sampled: Bool[Array, ""]
    generator_policy_id: Int[Array, ""]

    # Ephemeral decision-time audit.  These are the exact updated values used
    # by admission before any promotion, refresh, cascade, or trace reset.
    # They are learner output only and never enter persistent state or RNG.
    decision_update_available: Bool[Array, ""]
    decision_commit_available: Bool[Array, ""]
    decision_active_ops: Int[Array, " n_features"]
    decision_active_parent_a: Int[Array, " n_features"]
    decision_active_parent_b: Int[Array, " n_features"]
    decision_active_theta: Float[Array, "n_features 2"]
    decision_active_depth: Int[Array, " n_features"]
    decision_active_generator_policy: Int[Array, " n_features"]
    decision_active_ages: Int[Array, " n_features"]
    decision_active_fast_utilities: Float[Array, " n_features"]
    decision_active_slow_utilities: Float[Array, " n_features"]
    decision_active_direct_scores: Float[Array, " n_features"]
    decision_active_backed_scores: Float[Array, " n_features"]
    decision_active_eligible: Bool[Array, " n_features"]
    decision_active_selection_scores: Float[Array, " n_features"]
    decision_worst_active: Int[Array, ""]
    decision_has_active_slot: Bool[Array, ""]
    decision_candidate_ops: Int[Array, " n_candidates"]
    decision_candidate_parent_a: Int[Array, " n_candidates"]
    decision_candidate_parent_b: Int[Array, " n_candidates"]
    decision_candidate_theta: Float[Array, "n_candidates 2"]
    decision_candidate_depth: Int[Array, " n_candidates"]
    decision_candidate_generator_policy: Int[Array, " n_candidates"]
    decision_candidate_ages: Int[Array, " n_candidates"]
    decision_candidate_fast_utilities: Float[Array, " n_candidates"]
    decision_candidate_slow_utilities: Float[Array, " n_candidates"]
    decision_candidate_direct_scores: Float[Array, " n_candidates"]
    decision_candidate_novelty_scores: Float[Array, " n_candidates"]
    decision_candidate_augmented_scores: Float[Array, " n_candidates"]
    decision_candidate_mature: Bool[Array, " n_candidates"]
    decision_candidate_recomputed_depth: Int[Array, " n_candidates"]
    decision_candidate_topology_compatible: Bool[Array, "n_candidates n_features"]
    decision_candidate_depth_compatible: Bool[Array, "n_candidates n_features"]
    decision_candidate_headroom_compatible: Bool[
        Array, "n_candidates n_features"
    ]
    decision_candidate_margin_eligible: Bool[Array, "n_candidates n_features"]
    decision_candidate_destination_compatible: Bool[
        Array, "n_candidates n_features"
    ]
    decision_candidate_has_destination: Bool[Array, " n_candidates"]
    decision_candidate_ranking_scores: Float[Array, " n_candidates"]
    decision_candidate_refresh_utilities: Float[Array, " n_candidates"]
    decision_selected_candidate: Int[Array, ""]
    decision_has_candidate: Bool[Array, ""]
    decision_selected_destination: Int[Array, ""]
    decision_selected_refresh_candidate: Int[Array, ""]
    decision_has_refresh_candidate: Bool[Array, ""]
    decision_left_pack_destinations_enabled: Bool[Array, ""]
    decision_left_pack_destination_available: Bool[Array, ""]
    decision_effective_promotion_margin: Float[Array, ""]
    decision_selected_candidate_score: Float[Array, ""]
    decision_selected_destination_backed_score: Float[Array, ""]
    decision_margin_rhs: Float[Array, ""]
    decision_margin_passed: Bool[Array, ""]
    decision_selected_topology_ok: Bool[Array, ""]
    decision_selected_depth_ok: Bool[Array, ""]
    decision_selected_headroom_ok: Bool[Array, ""]
    decision_selected_can_promote: Bool[Array, ""]
    decision_should_promote: Bool[Array, ""]
    decision_should_refresh: Bool[Array, ""]

    proposal_formed: Bool[Array, ""]
    proposal_destination_bank: Int[Array, ""]
    proposal_destination_slot: Int[Array, ""]
    proposal_op: Int[Array, ""]
    proposal_parent_a: Int[Array, ""]
    proposal_parent_b: Int[Array, ""]
    proposal_theta: Float[Array, " 2"]
    proposal_depth: Int[Array, ""]
    proposal_generator_policy: Int[Array, ""]

    root_change_mask: Bool[Array, " n_features"]
    root_change_applied: Bool[Array, ""]
    post_root_pre_cascade_slot: Int[Array, ""]
    post_root_pre_cascade_op: Int[Array, ""]
    post_root_pre_cascade_parent_a: Int[Array, ""]
    post_root_pre_cascade_parent_b: Int[Array, ""]
    post_root_pre_cascade_theta: Float[Array, " 2"]
    post_root_pre_cascade_depth: Int[Array, ""]
    post_root_pre_cascade_generator_policy: Int[Array, ""]

    promotion_applied: Bool[Array, ""]
    promotion_source_candidate: Int[Array, ""]
    promotion_destination_active: Int[Array, ""]
    promoted_pre_refresh_op: Int[Array, ""]
    promoted_pre_refresh_parent_a: Int[Array, ""]
    promoted_pre_refresh_parent_b: Int[Array, ""]
    promoted_pre_refresh_theta: Float[Array, " 2"]
    promoted_pre_refresh_depth: Int[Array, ""]
    promoted_pre_refresh_generator_policy: Int[Array, ""]

    cascade_refill_mask: Bool[Array, " n_features"]
    cascade_final_ops: Int[Array, " n_features"]
    cascade_final_parent_a: Int[Array, " n_features"]
    cascade_final_parent_b: Int[Array, " n_features"]
    cascade_final_theta: Float[Array, "n_features 2"]
    cascade_final_depth: Int[Array, " n_features"]
    cascade_final_generator_policy: Int[Array, " n_features"]
    active_change_mask: Bool[Array, " n_features"]

    ordinary_candidate_refresh_mask: Bool[Array, " n_candidates"]
    post_promotion_candidate_refresh_mask: Bool[Array, " n_candidates"]
    candidate_refresh_mask: Bool[Array, " n_candidates"]
    candidate_rebound_mask: Bool[Array, " n_candidates"]
    candidate_overdepth_regeneration_mask: Bool[Array, " n_candidates"]
    candidate_final_ops: Int[Array, " n_candidates"]
    candidate_final_parent_a: Int[Array, " n_candidates"]
    candidate_final_parent_b: Int[Array, " n_candidates"]
    candidate_final_theta: Float[Array, "n_candidates 2"]
    candidate_final_depth: Int[Array, " n_candidates"]
    candidate_final_generator_policy: Int[Array, " n_candidates"]

    proposal_count: Int[Array, ""]
    root_change_count: Int[Array, ""]
    promotion_count: Int[Array, ""]
    cascade_refill_count: Int[Array, ""]
    ordinary_candidate_refresh_count: Int[Array, ""]
    post_promotion_candidate_refresh_count: Int[Array, ""]
    candidate_refresh_count: Int[Array, ""]
    candidate_rebound_count: Int[Array, ""]
    candidate_overdepth_regeneration_count: Int[Array, ""]
    logical_event_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class CompositionalFeatureUpdateResult:
    """Result of one compositional-feature update."""

    state: CompositionalFeatureState
    predictions: Float[Array, " n_tasks"]
    errors: Float[Array, " n_tasks"]
    metrics: Float[Array, " 7"]
    replaced_slot: Int[Array, ""]
    promoted_candidate: Int[Array, ""]
    curation_trace: CompositionalCurationTrace


@chex.dataclass(frozen=True)
class CompositionalRankingDiagnostics:
    """Inspectable pre-update scores for opt-in compositional curation.

    The direct arrays are learned utility only.  ``backed_active_scores`` adds
    the configured transitive descendant-to-ancestor retention credit, while
    ``augmented_candidate_scores`` adds only the configured novelty-admission
    bonus.  These diagnostics do not mutate state or predict that a promotion
    has a compatible destination; the update still applies maturity,
    topology, depth, and finite-input gates before any curation.
    """

    contract_valid: Bool[Array, ""]
    direct_active_scores: Float[Array, " n_features"]
    backed_active_scores: Float[Array, " n_features"]
    direct_candidate_scores: Float[Array, " n_candidates"]
    candidate_novelty_scores: Float[Array, " n_candidates"]
    augmented_candidate_scores: Float[Array, " n_candidates"]
    candidate_mature: Bool[Array, " n_candidates"]


@chex.dataclass(frozen=True)
class CompositionalFeatureLearningResult:
    """Result from a scan-based compositional feature run."""

    state: CompositionalFeatureState
    metrics: Float[Array, "num_steps 7"]


@chex.dataclass(frozen=True)
class FiniteCandidateSelectorState:
    """State for a finite-candidate bounded-loss selector.

    The selector treats candidate ids as fixed experts.  If a caller reuses a
    slot for a different candidate, the caller should reset that slot's state
    before interpreting the finite-expert regret metadata.
    """

    log_weights: Float[Array, " n_candidates"]
    cumulative_loss: Float[Array, " n_candidates"]
    action_counts: Float[Array, " n_candidates"]
    step_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class FiniteCandidateSelectorUpdateResult:
    """Result of one finite-candidate selector update."""

    state: FiniteCandidateSelectorState
    probabilities: Float[Array, " n_candidates"]
    bounded_losses: Float[Array, " n_candidates"]
    selected_action: Int[Array, ""]


class FiniteCandidateSelector:
    """Hedge/Exp3-style selector over a fixed finite candidate set.

    This is a generic no-regret selector abstraction for externally supplied
    bounded losses.  It is intentionally separate from the compositional
    promote heuristic: the theorem metadata applies to this fixed-candidate
    loss sequence, not to dynamic feature generation or candidate refresh.
    """

    def __init__(
        self,
        n_candidates: int,
        learning_rate: float = 1.0,
        exploration: float = 0.0,
        loss_lower_bound: float = 0.0,
        loss_upper_bound: float = 1.0,
        update_rule: str = CANDIDATE_SELECTOR_HEDGE,
    ) -> None:
        """Initialize a finite-candidate selector.

        Args:
            n_candidates: Number of fixed candidate experts.
            learning_rate: Exponentiated-gradient learning rate.
            exploration: Uniform probability floor mixed into action
                probabilities.
            loss_lower_bound: Lower bound assumed for finite observed losses.
            loss_upper_bound: Upper bound assumed for finite observed losses.
            update_rule: ``"hedge"`` for full-information losses, or
                ``"exp3"`` for selected-action importance-weighted updates.
        """
        if n_candidates < 1:
            raise ValueError("n_candidates must be positive")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= exploration < 1.0:
            raise ValueError("exploration must be in [0, 1)")
        if loss_lower_bound >= loss_upper_bound:
            raise ValueError("loss_lower_bound must be < loss_upper_bound")
        if update_rule not in {CANDIDATE_SELECTOR_HEDGE, CANDIDATE_SELECTOR_EXP3}:
            raise ValueError("update_rule must be 'hedge' or 'exp3'")
        if update_rule == CANDIDATE_SELECTOR_EXP3 and exploration <= 0.0:
            raise ValueError("exp3 selector requires positive exploration")

        self._n_candidates = int(n_candidates)
        self._learning_rate = float(learning_rate)
        self._exploration = float(exploration)
        self._loss_lower_bound = float(loss_lower_bound)
        self._loss_upper_bound = float(loss_upper_bound)
        self._update_rule = update_rule

    @property
    def n_candidates(self) -> int:
        """Number of fixed candidates."""
        return self._n_candidates

    @property
    def update_rule(self) -> str:
        """Selector update rule."""
        return self._update_rule

    def to_config(self) -> dict[str, Any]:
        """Serialize selector configuration."""
        return {
            "type": "FiniteCandidateSelector",
            "n_candidates": self._n_candidates,
            "learning_rate": self._learning_rate,
            "exploration": self._exploration,
            "loss_lower_bound": self._loss_lower_bound,
            "loss_upper_bound": self._loss_upper_bound,
            "update_rule": self._update_rule,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FiniteCandidateSelector":
        """Reconstruct a selector from :meth:`to_config` output."""
        config = dict(config)
        config.pop("type", None)
        return cls(**config)

    def init(self) -> FiniteCandidateSelectorState:
        """Create a uniform selector state."""
        return FiniteCandidateSelectorState(
            log_weights=jnp.zeros(self._n_candidates, dtype=jnp.float32),
            cumulative_loss=jnp.zeros(self._n_candidates, dtype=jnp.float32),
            action_counts=jnp.zeros(self._n_candidates, dtype=jnp.float32),
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    def probabilities(
        self,
        state: FiniteCandidateSelectorState,
    ) -> Float[Array, " n_candidates"]:
        """Return the current candidate probabilities."""
        logits = state.log_weights - jnp.max(state.log_weights)
        weights = jax.nn.softmax(logits)
        if self._exploration > 0.0:
            uniform = jnp.full_like(weights, 1.0 / float(self._n_candidates))
            weights = (1.0 - self._exploration) * weights + self._exploration * uniform
        return weights / jnp.sum(weights)

    def validate_bounded_losses(self, losses: Array) -> None:
        """Raise if finite losses violate the selector's theorem range."""
        losses = jnp.asarray(losses, dtype=jnp.float32)
        finite = jnp.isfinite(losses)
        outside = finite & (
            (losses < self._loss_lower_bound) | (losses > self._loss_upper_bound)
        )
        if bool(jnp.any(outside)):
            raise ValueError(
                "finite-candidate selector assumes finite losses in "
                f"[{self._loss_lower_bound}, {self._loss_upper_bound}]"
            )

    def _unit_losses(self, losses: Array) -> Array:
        losses = jnp.asarray(losses, dtype=jnp.float32)
        finite = jnp.isfinite(losses)
        width = jnp.asarray(
            self._loss_upper_bound - self._loss_lower_bound,
            dtype=jnp.float32,
        )
        unit = (losses - self._loss_lower_bound) / width
        unit = jnp.clip(unit, 0.0, 1.0)
        return jnp.where(finite, unit, jnp.nan)

    def update(
        self,
        state: FiniteCandidateSelectorState,
        losses: Float[Array, " n_candidates"],
        selected_action: Array | int | None = None,
    ) -> FiniteCandidateSelectorUpdateResult:
        """Update selector preferences from bounded losses.

        ``NaN`` losses are ignored.  For ``update_rule="hedge"``, all finite
        losses are observed.  For ``update_rule="exp3"``, only
        ``selected_action`` receives an importance-weighted update.
        """
        bounded_losses = self._unit_losses(losses)
        finite = jnp.isfinite(bounded_losses)
        probabilities = self.probabilities(state)
        if self._update_rule == CANDIDATE_SELECTOR_EXP3:
            action = (
                jnp.argmax(probabilities).astype(jnp.int32)
                if selected_action is None
                else jnp.asarray(selected_action, dtype=jnp.int32)
            )
            probability = jnp.maximum(probabilities[action], 1e-6)
            selected_finite = finite[action]
            loss_hat = jnp.where(
                selected_finite,
                bounded_losses[action] / probability,
                jnp.array(0.0, dtype=jnp.float32),
            )
            update_losses = jnp.zeros_like(bounded_losses).at[action].set(loss_hat)
            update_finite = jnp.zeros_like(finite).at[action].set(selected_finite)
        else:
            action = jnp.argmin(
                jnp.where(finite, bounded_losses, jnp.inf)
            ).astype(jnp.int32)
            update_losses = jnp.where(finite, bounded_losses, 0.0)
            update_finite = finite

        log_weights = state.log_weights - self._learning_rate * update_losses
        log_weights = log_weights - jnp.mean(log_weights)
        cumulative_loss = state.cumulative_loss + jnp.where(
            update_finite,
            jnp.nan_to_num(bounded_losses, nan=0.0),
            0.0,
        )
        action_counts = state.action_counts + update_finite.astype(jnp.float32)
        next_state = FiniteCandidateSelectorState(
            log_weights=log_weights,
            cumulative_loss=cumulative_loss,
            action_counts=action_counts,
            step_count=state.step_count + 1,
        )
        return FiniteCandidateSelectorUpdateResult(
            state=next_state,
            probabilities=probabilities,
            bounded_losses=bounded_losses,
            selected_action=action,
        )

    def regret_metadata(self, horizon: int) -> dict[str, Any]:
        """Return finite-candidate regret assumptions and bound metadata."""
        if horizon < 1:
            raise ValueError("horizon must be positive")
        width = self._loss_upper_bound - self._loss_lower_bound
        log_k = math.log(self._n_candidates)
        if self._n_candidates == 1:
            regret_bound = 0.0
        elif self._update_rule == CANDIDATE_SELECTOR_HEDGE:
            regret_bound = width * (
                log_k / self._learning_rate + self._learning_rate * horizon / 8.0
            )
        else:
            regret_bound = width * 2.0 * math.sqrt(horizon * self._n_candidates * log_k)

        return {
            "algorithm": self._update_rule,
            "candidate_count": self._n_candidates,
            "horizon": horizon,
            "assumptions": {
                "finite_candidate_set": True,
                "fixed_candidate_identities": True,
                "loss_lower_bound": self._loss_lower_bound,
                "loss_upper_bound": self._loss_upper_bound,
                "finite_losses_only": True,
                "comparator": "best fixed candidate in hindsight",
                "exp3_requires_unbiased_importance_weighted_losses": (
                    self._update_rule == CANDIDATE_SELECTOR_EXP3
                ),
            },
            "regret_bound": regret_bound,
            "regret_statement": (
                "Hedge full-information regret is bounded by "
                "(b-a)(ln(K)/eta + eta*T/8) for losses in [a,b]. "
                "The Exp3-style entry records the usual order bound and "
                "requires positive exploration plus unbiased bandit losses."
            ),
        }


def _candidate_scores_to_unit_losses(
    scores: Array,
    finite_mask: Array,
) -> Array:
    """Map candidate promotion scores to bounded losses for the selector.

    Higher utility-like scores become lower losses.  The conversion is only
    for the opt-in finite-candidate selector; it is not a theorem for the
    default ``"legacy"`` argmax-utility promotion path.
    """
    scores = jnp.asarray(scores, dtype=jnp.float32)
    finite_mask = jnp.asarray(finite_mask, dtype=jnp.bool_) & jnp.isfinite(scores)
    high = jnp.max(jnp.where(finite_mask, scores, -jnp.inf))
    low = jnp.min(jnp.where(finite_mask, scores, jnp.inf))
    span = high - low
    normalized_score = jnp.where(
        span > 1e-6,
        (scores - low) / jnp.maximum(span, 1e-6),
        0.5,
    )
    losses = 1.0 - normalized_score
    return jnp.where(finite_mask, jnp.clip(losses, 0.0, 1.0), jnp.nan)


FEATURE_VALUE_CLIP = 10.0
CANDIDATE_IMPRINT_SCALE = 0.1


def _compute_feature_values(
    ops: Array,
    parent_a: Array,
    parent_b: Array,
    theta: Array,
    observation: Array,
) -> Array:
    """Forward-evaluate compositional features in topological order.

    Slots are evaluated in index order ``0, 1, ..., n_features - 1``; the
    invariant maintained at construction time is that every parent index is
    strictly smaller than its child (or, for ``OP_RAW``, that ``parent_a`` is
    a raw-input index).  This guarantees a valid topological evaluation.

    Each slot's output is clipped to ``[-FEATURE_VALUE_CLIP, FEATURE_VALUE_CLIP]``
    so that chains of multiplications cannot drive values to infinity on
    rare large-input observations.  The clip is intentionally generous
    (default ``10.0``) so that typical product magnitudes are unaffected;
    it acts as a safety rail rather than a learned nonlinearity.

    Args:
        ops: Op type per slot.
        parent_a: First parent indices.
        parent_b: Second parent indices (``-1`` allowed for ``OP_RAW``).
        theta: Per-feature parameter vectors of length two.
        observation: Raw observation vector.

    Returns:
        Vector of feature values, one per active slot.
    """
    n_features = ops.shape[0]
    feature_dim = observation.shape[0]

    def step_fn(values: Array, i: Array) -> tuple[Array, None]:
        op = ops[i]
        a = parent_a[i]
        b = parent_b[i]
        # Safe indexing: clamp to valid ranges, then mask via jnp.where.
        safe_a_obs = jnp.clip(a, 0, feature_dim - 1)
        safe_a_feat = jnp.clip(a, 0, n_features - 1)
        safe_b_feat = jnp.clip(b, 0, n_features - 1)

        raw = observation[safe_a_obs]
        val_a = values[safe_a_feat]
        val_b = jnp.where(b >= 0, values[safe_b_feat], 0.0)

        product = val_a * val_b
        summ = val_a + val_b
        pre_tanh = theta[i, 0] * val_a + theta[i, 1] * val_b
        tanh_val = jnp.tanh(pre_tanh)
        gated = val_a * jax.nn.sigmoid(val_b)

        new_val = jnp.select(
            [
                op == OP_RAW,
                op == OP_PRODUCT,
                op == OP_SUM,
                op == OP_TANH,
                op == OP_GATED,
            ],
            [raw, product, summ, tanh_val, gated],
            default=jnp.array(0.0, dtype=jnp.float32),
        )
        new_val = jnp.clip(new_val, -FEATURE_VALUE_CLIP, FEATURE_VALUE_CLIP)
        return values.at[i].set(new_val), None

    init_values = jnp.zeros(n_features, dtype=jnp.float32)
    values, _ = jax.lax.scan(step_fn, init_values, jnp.arange(n_features))
    return values


def _theta_local_grads(
    ops: Array,
    parent_a: Array,
    parent_b: Array,
    theta: Array,
    feature_values: Array,
) -> tuple[Array, Array]:
    """Compute per-feature theta gradients for the local op output.

    For ``OP_TANH``, ``d val_i / d theta_i = (1 - tanh^2(pre_i)) * [val_a, val_b]``.
    For ``OP_GATED``, theta is unused so the gradient is zero.

    Args:
        ops: Op types per slot.
        parent_a: First parent indices.
        parent_b: Second parent indices.
        theta: Per-feature parameter vectors.
        feature_values: Already-computed feature values for this observation.

    Returns:
        ``(d_val_d_theta0, d_val_d_theta1)`` arrays of shape ``(n_features,)``.
        Entries for slots whose op does not use theta are zero.
    """
    n_features = ops.shape[0]
    safe_a = jnp.clip(parent_a, 0, n_features - 1)
    safe_b = jnp.clip(parent_b, 0, n_features - 1)
    val_a = jnp.where(parent_a >= 0, feature_values[safe_a], 0.0)
    val_b = jnp.where(parent_b >= 0, feature_values[safe_b], 0.0)

    is_tanh = (ops == OP_TANH).astype(jnp.float32)
    tanh_factor = is_tanh * (1.0 - feature_values * feature_values)
    d_theta0 = tanh_factor * val_a
    d_theta1 = tanh_factor * val_b
    return d_theta0, d_theta1


def _compute_candidate_value(
    op: Array,
    parent_a: Array,
    parent_b: Array,
    theta: Array,
    active_values: Array,
    observation: Array,
) -> Array:
    """Evaluate one candidate op against the current active feature values."""
    n_features = active_values.shape[0]
    feature_dim = observation.shape[0]
    safe_a_obs = jnp.clip(parent_a, 0, feature_dim - 1)
    safe_a_feat = jnp.clip(parent_a, 0, n_features - 1)
    safe_b_feat = jnp.clip(parent_b, 0, n_features - 1)

    raw = observation[safe_a_obs]
    val_a = active_values[safe_a_feat]
    val_b = jnp.where(parent_b >= 0, active_values[safe_b_feat], 0.0)
    product = val_a * val_b
    summ = val_a + val_b
    tanh_val = jnp.tanh(theta[0] * val_a + theta[1] * val_b)
    gated = val_a * jax.nn.sigmoid(val_b)
    value = jnp.select(
        [
            op == OP_RAW,
            op == OP_PRODUCT,
            op == OP_SUM,
            op == OP_TANH,
            op == OP_GATED,
        ],
        [raw, product, summ, tanh_val, gated],
        default=jnp.array(0.0, dtype=jnp.float32),
    )
    return jnp.clip(value, -FEATURE_VALUE_CLIP, FEATURE_VALUE_CLIP)


def _candidate_theta_local_grads(
    ops: Array,
    parent_a: Array,
    parent_b: Array,
    theta: Array,
    candidate_values: Array,
    active_values: Array,
) -> tuple[Array, Array]:
    """Compute local theta gradients for one-step candidate features.

    Candidate slots are evaluated as shallow ops over the active feature bank,
    so only their own parameters receive a local gradient.  This mirrors the
    active-bank ``OP_TANH`` update without backpropagating through candidate
    parents.
    """
    n_features = active_values.shape[0]
    safe_a = jnp.clip(parent_a, 0, n_features - 1)
    safe_b = jnp.clip(parent_b, 0, n_features - 1)
    val_a = active_values[safe_a]
    val_b = jnp.where(parent_b >= 0, active_values[safe_b], 0.0)

    is_tanh = (ops == OP_TANH).astype(jnp.float32)
    tanh_factor = is_tanh * (1.0 - candidate_values * candidate_values)
    d_theta0 = tanh_factor * val_a
    d_theta1 = tanh_factor * val_b
    return d_theta0, d_theta1


def _imprint_candidate_output_weights(
    errors: Array,
    candidate_value: Array,
    active_count: Array,
    scale: Array | float = CANDIDATE_IMPRINT_SCALE,
) -> Array:
    """Initialize a candidate head with a small residual-aligned coefficient.

    This is a one-sample least-squares imprint, damped so a refreshed shadow
    feature gets an immediate utility signal without dominating later LMS
    updates.  The undamped one-sample solution is ``errors * v / v**2``; the
    ``+ 1`` in the denominator ridges it so near-zero feature values cannot
    blow the coefficient up.  Inactive heads already have zero error, so they
    stay zero.

    ``CompositionalFeatureLearner._initial_candidate_output_weights`` is the
    production call site: it evaluates the candidate's feature value and
    delegates here with the learner's configured ``candidate_imprint_scale``.
    """
    denom = candidate_value * candidate_value + 1.0
    return scale * errors * candidate_value / (denom * active_count)


class CompositionalFeatureLearner:
    """Fixed-budget DAG feature learner that composes features of features.

    Each feature slot stores an op type, two parent indices, a small parameter
    vector ``theta`` (used only by ``OP_TANH``), a topological depth, and
    standard utility/age tracking. ``OP_GATED`` is the fixed ordered operation
    ``value_a * sigmoid(value_b)``. Output is

    ``y_k = sum_i output_weights[k, i] * feature_values[i] + output_bias[k]``.

    A fixed prefix of ``feature_dim`` slots holds raw-input features
    (``OP_RAW``); the rest are composed.  Generation enforces strict-less-than
    parent indices so ``jax.lax.scan`` over slots in index order is a valid
    topological traversal.  Replacing a slot cascades through its descendants
    so dangling parent references never appear at evaluation time.
    """

    def __init__(
        self,
        n_features: int,
        n_tasks: int,
        candidate_count: int = 0,
        step_size_output: float = 0.03,
        step_size_theta: float = 0.003,
        utility_decay: float = 0.995,
        replacement_interval: int = 200,
        min_feature_age: int = 100,
        candidate_min_age: int = 50,
        promotion_margin: float = 1.05,
        promotion_blend: float = 0.5,
        promotion_output_mode: str = PROMOTION_SCALED_CANDIDATE,
        max_depth: int = 4,
        topology_headroom_reserve: bool = False,
        topology_left_pack_destinations: bool = False,
        use_obgd: bool = True,
        obgd_kappa: float = 2.0,
        generation_strategy: str = GENERATION_UTILITY,
        parent_temperature: float = 1.0,
        parent_novelty_weight: float = 0.0,
        parent_depth_prior: float = 0.0,
        retention_depth_bonus: float = 0.0,
        residual_guidance: float = 1.0,
        candidate_imprint_scale: float = CANDIDATE_IMPRINT_SCALE,
        train_candidate_theta: bool = False,
        signed_tanh_scaffold_count: int = 0,
        future_utility_mix: float = 0.0,
        future_utility_trace_decay: float = 0.0,
        future_utility_trace_mode: str = "marginal",
        future_utility_normalization: str = "none",
        future_utility_normalization_decay: float = 0.99,
        future_utility_rare_task_power: float = 0.0,
        future_utility_task_activity_decay: float = 0.995,
        candidate_scoring_mode: str = "legacy",
        candidate_score_trace_decay: float = 0.0,
        candidate_score_energy_epsilon: float = 1e-6,
        candidate_novelty_weight: float = 0.0,
        candidate_novelty_power: float = 1.0,
        candidate_novelty_floor: float = 0.05,
        candidate_novelty_admission_bonus: float = 0.0,
        candidate_selector: str = CANDIDATE_SELECTOR_LEGACY,
        candidate_selector_learning_rate: float = 1.0,
        candidate_selector_exploration: float = 0.0,
        retention_slow_utility_decay: float = 0.0,
        ancestor_utility_backup_decay: float = 0.0,
        retention_tanh_min_count: int = 0,
        retention_product_min_count: int = 0,
        operation_prior: tuple[float, ...] | None = None,
        learn_generator_resources: bool = False,
        generator_resource_contexts: int = 1,
        generator_resource_learning_rate: float = 1.0,
        generator_resource_discount: float = 0.995,
        generator_resource_exploration: float = 0.01,
        generator_resource_advantage_clip: float = 10.0,
        generator_resource_cost_weight: float = 0.0,
        generator_resource_update_rule: str = "hedge",
        generator_resource_promotion_credit: float = 0.0,
        generator_resource_initial_preferences: tuple[float, ...] | None = None,
    ):
        """Initialize the compositional feature learner.

        Args:
            n_features: Number of active feature slots.  Must exceed the
                raw-input dimension passed to ``init`` so at least one
                composed slot is available.
            n_tasks: Number of supervised output heads.
            candidate_count: Number of shadow candidate slots.
            step_size_output: LMS step-size for output weights.
            step_size_theta: LMS step-size for per-feature theta updates.
            utility_decay: EMA decay for utility estimates.
            replacement_interval: Steps between replacement attempts (``0``
                disables replacement).
            min_feature_age: Minimum active age before a feature is eligible
                for replacement.
            candidate_min_age: Minimum candidate age before promotion.
            promotion_margin: Candidate utility must exceed
                ``promotion_margin * worst_active_utility`` to promote.
            promotion_blend: Fraction of candidate output weights copied on
                promotion. With ``promotion_output_mode="scaled_candidate"``,
                promoted output weights are ``promotion_blend * candidate``.
                With ``promotion_output_mode="blend"``, promoted output weights
                are ``(1 - promotion_blend) * old + promotion_blend * candidate``.
            promotion_output_mode: How to initialize output weights when a
                candidate is promoted. ``"scaled_candidate"`` discards the
                replaced slot's weights and installs the scaled candidate
                head; ``"blend"`` interpolates old and candidate weights,
                which reduces output churn at promotion.
            max_depth: Maximum allowed topological depth for any feature.
            topology_headroom_reserve: If true, candidate admission reserves
                one strictly later active slot for every composition level
                remaining below ``max_depth``. This is task-blind and affects
                only candidate/destination compatibility.
            topology_left_pack_destinations: If true, candidate admission
                chooses the lowest-index structurally compatible destination
                that also passes the unchanged strict promotion margin. This
                is task-blind beyond the already learned ranking scores.
            use_obgd: Whether to bound effective updates ObGD-style.
            obgd_kappa: ObGD bounding sensitivity.
            generation_strategy: Parent-generation strategy for fresh
                candidates/replacements. ``"utility"`` biases parent
                selection toward high-utility features, ``"uniform"`` is a
                control,
                ``"mutation"`` anchors one parent on high-utility features and
                samples the other from shallow eligible features, and
                ``"residual_imprint"`` additionally uses one-step residual
                credit and can initialize fresh candidate output weights from
                the current residual. ``"recursive_product"`` is an opt-in
                experimental policy for product-structured recursive targets:
                active initialization builds depth-1 product scaffolds and
                candidates are generated as products of an existing composed
                feature with a raw/shallow parent. ``"robust_recursive"`` uses
                the same causal utility path with product-biased op priors,
                utility/novelty parent selection, and protected recursive
                scaffolds. ``"dovetail_product_coverage"`` is a defaults-off,
                fixed-cadence, product-only coverage strategy. It alternates a
                deterministic enumeration of distinct raw pairs with a
                deterministic enumeration of admitted depth-1/raw extensions,
                deriving its bounded cursor from the exact lifetime counter.
            parent_temperature: Softmax temperature for non-uniform parent
                selection. Lower values make the parent search greedier.
            parent_novelty_weight: Extra parent score for low-utility/young
                eligible parents. This keeps search from repeatedly sampling
                the same already-dominant parent.
            parent_depth_prior: Extra parent score for deeper parents, used to
                make feature-of-feature construction more likely without
                hard-coding a target.
            retention_depth_bonus: Additive replacement-score bonus for deeper
                active features. Higher values protect recursive structure from
                immediate churn once discovered.
            residual_guidance: Weight on one-step residual/credit scores in
                ``generation_strategy="residual_imprint"``.
            candidate_imprint_scale: Scale for initializing freshly generated
                candidate output weights from the current residual. Set to
                ``0.0`` to disable imprint initialization.
            train_candidate_theta: If true, candidate ``OP_TANH`` parameters
                receive online shadow updates through their candidate output
                heads before promotion.
            signed_tanh_scaffold_count: Number of deterministic signed
                ``OP_TANH`` raw-pair scaffolds inserted after product
                scaffolds for ``generation_strategy="robust_recursive"``.
                These are task-agnostic local nonlinear basis functions.
            future_utility_mix: Mixture weight for one-step counterfactual
                output-loss-reduction utility. ``0`` uses only the
                backward-looking magnitude/credit utility. When
                ``future_utility_trace_decay > 0``, the future term uses
                causal residual/feature traces instead of only the current
                sample.
            future_utility_trace_decay: Discount for temporally extended
                future-utility traces. ``0`` reduces the trace to the
                one-step counterfactual. Values near ``1`` credit features
                whose residual alignment recurs over multiple recent steps.
            future_utility_trace_mode: ``"contribution"`` traces
                ``error * feature`` directly; ``"marginal"`` approximates it
                with the product of separate residual and feature traces,
                retained as an ablation control.
            future_utility_normalization: Optional normalization for the
                future term: ``"none"``, ``"age"``, ``"uncertainty"``, or
                ``"uncertainty_age"``.
            future_utility_normalization_decay: EMA decay for the future-signal
                second moment used by uncertainty normalization.
            future_utility_rare_task_power: Extra inverse-frequency weighting
                applied only to future-utility task credit. ``0`` disables it.
            future_utility_task_activity_decay: EMA decay for task activity
                frequencies used by rare-task future credit.
            candidate_scoring_mode: ``"legacy"`` scores candidates with the
                same magnitude/credit blend used for active slots.
                ``"energy_novelty"`` uses matching-pursuit residual
                alignment normalized by feature energy, with candidate scores
                optionally downweighted by active-feature correlation.
            candidate_score_trace_decay: Discount for the opt-in residual,
                energy, and candidate-active correlation traces.
            candidate_score_energy_epsilon: Positive stabilizer for
                energy-normalized candidate scoring.
            candidate_novelty_weight: Interpolation between no novelty
                penalty (``0``) and full correlation novelty gating (``1``).
            candidate_novelty_power: Exponent applied to the novelty gate.
            candidate_novelty_floor: Minimum novelty gate for highly
                correlated candidates.
            candidate_novelty_admission_bonus: Opt-in additive promotion score
                for a mature candidate, scaled by correlation novelty.  This
                permits a novel intermediate with zero direct predictive
                utility to enter the active DAG without changing its learned
                utility.  Positive values require ``candidate_scoring_mode``
                ``"energy_novelty"`` so admission has accumulated evidence.
            candidate_selector: Optional finite-candidate selector for the
                promotion candidate choice. ``"legacy"`` promotes the
                argmax-utility candidate. ``"hedge"`` or ``"exp3"`` uses a
                bounded-loss selector over candidate slots before the usual
                promotion margin check.
            candidate_selector_learning_rate: Exponentiated-gradient step
                size for the opt-in candidate selector.
            candidate_selector_exploration: Uniform probability floor for the
                opt-in selector. Required to be positive for ``"exp3"``.
            retention_slow_utility_decay: Opt-in slow utility EMA decay for
                hysteretic retention. When positive, active replacement uses
                ``max(fast_utility, slow_utility)`` so mature features are
                deleted only when both timescales are low.
            ancestor_utility_backup_decay: Opt-in descendant-to-ancestor
                retention backup in ``[0, 1]``.  A reverse-topological pass
                gives each valid parent ``decay * child_score`` by max backup;
                repeated application therefore credits all ancestors without
                changing direct learned utilities.
            retention_tanh_min_count: Minimum number of active ``OP_TANH``
                slots to protect from replacement. ``0`` disables this quota.
            retention_product_min_count: Minimum number of active
                ``OP_PRODUCT`` slots to protect from replacement. ``0``
                disables this quota.
            operation_prior: Optional operation probabilities in
                ``[raw, product, sum, tanh, gated]`` order. When supplied,
                generation uses this fixed prior instead of the strategy
                default. The raw probability should be zero for composed
                feature generation.
            learn_generator_resources: If true, use a generator-internal
                meta-resource manager to choose operation/parent mode,
                replacement rate, promotion aggressiveness, candidate refresh
                age, and residual-imprint scale.
            generator_resource_contexts: Number of independent context bins
                for generator-policy allocation.
            generator_resource_learning_rate: Exponentiated-gradient step size
                for generator-policy rewards.
            generator_resource_discount: Preference decay for generator
                policies.
            generator_resource_exploration: Uniform policy-allocation floor.
            generator_resource_advantage_clip: Absolute clip on centered
                generator-policy rewards.
            generator_resource_cost_weight: Optional cost penalty for more
                aggressive generator policies.
            generator_resource_update_rule: ``"hedge"`` uses all finite
                provenance scores; ``"exp3"`` updates only the sampled policy
                with importance weighting.
            generator_resource_promotion_credit: Optional bonus assigned to
                the policy whose delayed candidate is promoted.
            generator_resource_initial_preferences: Optional initial
                log-preferences over generator policies.
        """
        if n_features < 1:
            raise ValueError("n_features must be positive")
        if n_tasks < 1:
            raise ValueError("n_tasks must be positive")
        if candidate_count < 0:
            raise ValueError("candidate_count must be non-negative")
        if not 0.0 <= utility_decay < 1.0:
            raise ValueError("utility_decay must be in [0, 1)")
        for name, value in (
            ("replacement_interval", replacement_interval),
            ("min_feature_age", min_feature_age),
            ("candidate_min_age", candidate_min_age),
        ):
            if type(value) is not int:
                raise TypeError(f"{name} must be an exact Python integer")
            if not 0 <= value <= _INT32_MAX:
                raise ValueError(f"{name} must be inside [0, int32_max]")
        if (
            learn_generator_resources
            and replacement_interval == 1
            and max(DEFAULT_GENERATOR_META_REPLACEMENT_MULTIPLIERS) > 1.0
        ):
            raise ValueError(
                "learned generator resources with replacement_interval=1 create an "
                "unbounded replacement-credit backlog"
            )
        if not 0.0 <= promotion_blend <= 1.0:
            raise ValueError("promotion_blend must be in [0, 1]")
        if promotion_output_mode not in {
            PROMOTION_SCALED_CANDIDATE,
            PROMOTION_BLEND,
        }:
            raise ValueError(
                "promotion_output_mode must be 'scaled_candidate' or 'blend'"
            )
        if max_depth < 1:
            raise ValueError("max_depth must be positive")
        if type(topology_headroom_reserve) is not bool:
            raise TypeError("topology_headroom_reserve must be an exact boolean")
        if type(topology_left_pack_destinations) is not bool:
            raise TypeError(
                "topology_left_pack_destinations must be an exact boolean"
            )
        if generation_strategy not in {
            GENERATION_UNIFORM,
            GENERATION_UTILITY,
            GENERATION_MUTATION,
            GENERATION_RESIDUAL_IMPRINT,
            GENERATION_RECURSIVE_PRODUCT,
            GENERATION_ROBUST_RECURSIVE,
            GENERATION_DOVETAIL_PRODUCT_COVERAGE,
        }:
            raise ValueError(
                "generation_strategy must be one of "
                "'uniform', 'utility', 'mutation', 'residual_imprint', "
                "'recursive_product', 'robust_recursive', or "
                "'dovetail_product_coverage'"
            )
        if parent_temperature <= 0.0:
            raise ValueError("parent_temperature must be positive")
        if parent_novelty_weight < 0.0:
            raise ValueError("parent_novelty_weight must be non-negative")
        if parent_depth_prior < 0.0:
            raise ValueError("parent_depth_prior must be non-negative")
        if retention_depth_bonus < 0.0:
            raise ValueError("retention_depth_bonus must be non-negative")
        if residual_guidance < 0.0:
            raise ValueError("residual_guidance must be non-negative")
        if candidate_imprint_scale < 0.0:
            raise ValueError("candidate_imprint_scale must be non-negative")
        if signed_tanh_scaffold_count < 0:
            raise ValueError("signed_tanh_scaffold_count must be non-negative")
        if not 0.0 <= future_utility_mix <= 1.0:
            raise ValueError("future_utility_mix must be in [0, 1]")
        if not 0.0 <= future_utility_trace_decay < 1.0:
            raise ValueError("future_utility_trace_decay must be in [0, 1)")
        if future_utility_trace_mode not in {"contribution", "marginal"}:
            raise ValueError(
                "future_utility_trace_mode must be 'contribution' or 'marginal'"
            )
        if future_utility_normalization not in {
            "none",
            "age",
            "uncertainty",
            "uncertainty_age",
        }:
            raise ValueError(
                "future_utility_normalization must be one of "
                "'none', 'age', 'uncertainty', or 'uncertainty_age'"
            )
        if not 0.0 <= future_utility_normalization_decay < 1.0:
            raise ValueError("future_utility_normalization_decay must be in [0, 1)")
        if future_utility_rare_task_power < 0.0:
            raise ValueError("future_utility_rare_task_power must be non-negative")
        if not 0.0 <= future_utility_task_activity_decay < 1.0:
            raise ValueError("future_utility_task_activity_decay must be in [0, 1)")
        if candidate_scoring_mode not in {"legacy", "energy_novelty"}:
            raise ValueError(
                "candidate_scoring_mode must be 'legacy' or 'energy_novelty'"
            )
        if not 0.0 <= candidate_score_trace_decay < 1.0:
            raise ValueError("candidate_score_trace_decay must be in [0, 1)")
        if candidate_score_energy_epsilon <= 0.0:
            raise ValueError("candidate_score_energy_epsilon must be positive")
        if not 0.0 <= candidate_novelty_weight <= 1.0:
            raise ValueError("candidate_novelty_weight must be in [0, 1]")
        if candidate_novelty_power <= 0.0:
            raise ValueError("candidate_novelty_power must be positive")
        if not 0.0 <= candidate_novelty_floor <= 1.0:
            raise ValueError("candidate_novelty_floor must be in [0, 1]")
        if type(candidate_novelty_admission_bonus) is not float:
            raise TypeError(
                "candidate_novelty_admission_bonus must be an exact Python float"
            )
        if (
            not math.isfinite(candidate_novelty_admission_bonus)
            or candidate_novelty_admission_bonus < 0.0
        ):
            raise ValueError(
                "candidate_novelty_admission_bonus must be finite and non-negative"
            )
        if candidate_novelty_admission_bonus > 0.0:
            if candidate_count < 1:
                raise ValueError(
                    "candidate_novelty_admission_bonus requires candidate_count > 0"
                )
            if candidate_scoring_mode != "energy_novelty":
                raise ValueError(
                    "candidate_novelty_admission_bonus requires "
                    "candidate_scoring_mode='energy_novelty'"
                )
        if candidate_selector not in {
            CANDIDATE_SELECTOR_LEGACY,
            CANDIDATE_SELECTOR_HEDGE,
            CANDIDATE_SELECTOR_EXP3,
        }:
            raise ValueError("candidate_selector must be 'legacy', 'hedge', or 'exp3'")
        if candidate_selector != CANDIDATE_SELECTOR_LEGACY and candidate_count < 1:
            raise ValueError("candidate_selector requires candidate_count > 0")
        if candidate_selector_learning_rate <= 0.0:
            raise ValueError("candidate_selector_learning_rate must be positive")
        if not 0.0 <= candidate_selector_exploration < 1.0:
            raise ValueError("candidate_selector_exploration must be in [0, 1)")
        if (
            candidate_selector == CANDIDATE_SELECTOR_EXP3
            and candidate_selector_exploration <= 0.0
        ):
            raise ValueError("exp3 candidate_selector requires positive exploration")
        if not 0.0 <= retention_slow_utility_decay < 1.0:
            raise ValueError("retention_slow_utility_decay must be in [0, 1)")
        if type(ancestor_utility_backup_decay) is not float:
            raise TypeError(
                "ancestor_utility_backup_decay must be an exact Python float"
            )
        if (
            not math.isfinite(ancestor_utility_backup_decay)
            or not 0.0 <= ancestor_utility_backup_decay <= 1.0
        ):
            raise ValueError("ancestor_utility_backup_decay must be finite and in [0, 1]")
        if retention_tanh_min_count < 0:
            raise ValueError("retention_tanh_min_count must be non-negative")
        if retention_product_min_count < 0:
            raise ValueError("retention_product_min_count must be non-negative")
        if operation_prior is not None:
            if len(operation_prior) != NUM_OPS:
                raise ValueError("operation_prior must have one entry per op")
            if any(type(prob) is bool for prob in operation_prior):
                raise ValueError("operation_prior entries must be finite real numbers")
            try:
                finite_entries = tuple(math.isfinite(prob) for prob in operation_prior)
            except TypeError as error:
                raise ValueError(
                    "operation_prior entries must be finite real numbers"
                ) from error
            if not all(finite_entries):
                raise ValueError("operation_prior entries must be finite real numbers")
            if any(prob < 0.0 for prob in operation_prior):
                raise ValueError("operation_prior entries must be non-negative")
            if operation_prior[OP_RAW] != 0.0:
                raise ValueError("operation_prior cannot assign mass to OP_RAW")
            composing_mass = sum(float(prob) for prob in operation_prior[1:])
            if not math.isfinite(composing_mass) or composing_mass <= 0.0:
                raise ValueError(
                    "operation_prior must have positive finite composing mass"
                )
        if generation_strategy == GENERATION_DOVETAIL_PRODUCT_COVERAGE:
            if candidate_count < 1:
                raise ValueError(
                    "dovetail_product_coverage requires candidate_count > 0"
                )
            if replacement_interval < 1:
                raise ValueError(
                    "dovetail_product_coverage requires a fixed replacement interval"
                )
            if max_depth < 2:
                raise ValueError("dovetail_product_coverage requires max_depth >= 2")
            if learn_generator_resources:
                raise ValueError(
                    "dovetail_product_coverage does not allow "
                    "learn_generator_resources"
                )
            if operation_prior is not None and (
                operation_prior[OP_PRODUCT] <= 0.0
                or any(
                    operation_prior[op] != 0.0
                    for op in (OP_RAW, OP_SUM, OP_TANH, OP_GATED)
                )
            ):
                raise ValueError(
                    "dovetail_product_coverage is product-only; operation_prior "
                    "cannot support other operations"
                )
        if generator_resource_contexts < 1:
            raise ValueError("generator_resource_contexts must be positive")
        if generator_resource_learning_rate < 0.0:
            raise ValueError("generator_resource_learning_rate must be non-negative")
        if not 0.0 <= generator_resource_discount <= 1.0:
            raise ValueError("generator_resource_discount must be in [0, 1]")
        if not 0.0 <= generator_resource_exploration < 1.0:
            raise ValueError("generator_resource_exploration must be in [0, 1)")
        if generator_resource_advantage_clip <= 0.0:
            raise ValueError("generator_resource_advantage_clip must be positive")
        if generator_resource_cost_weight < 0.0:
            raise ValueError("generator_resource_cost_weight must be non-negative")
        if generator_resource_update_rule not in {"hedge", "exp3"}:
            raise ValueError("generator_resource_update_rule must be 'hedge' or 'exp3'")
        if generator_resource_promotion_credit < 0.0:
            raise ValueError("generator_resource_promotion_credit must be non-negative")
        if (
            generator_resource_initial_preferences is not None
            and len(generator_resource_initial_preferences)
            != len(DEFAULT_GENERATOR_META_POLICY_NAMES)
        ):
            raise ValueError(
                "generator_resource_initial_preferences must match the default "
                "generator policy count"
            )

        self._n_features = n_features
        self._n_tasks = n_tasks
        self._candidate_count = candidate_count
        self._step_size_output = step_size_output
        self._step_size_theta = step_size_theta
        self._utility_decay = utility_decay
        self._replacement_interval = replacement_interval
        self._min_feature_age = min_feature_age
        self._candidate_min_age = candidate_min_age
        self._promotion_margin = promotion_margin
        self._promotion_blend = promotion_blend
        self._promotion_output_mode = promotion_output_mode
        self._max_depth = max_depth
        self._topology_headroom_reserve = topology_headroom_reserve
        self._topology_left_pack_destinations = topology_left_pack_destinations
        self._use_obgd = use_obgd
        self._obgd_kappa = obgd_kappa
        self._generation_strategy = generation_strategy
        self._parent_temperature = parent_temperature
        self._parent_novelty_weight = parent_novelty_weight
        self._parent_depth_prior = parent_depth_prior
        self._retention_depth_bonus = retention_depth_bonus
        self._residual_guidance = residual_guidance
        self._candidate_imprint_scale = candidate_imprint_scale
        self._train_candidate_theta = train_candidate_theta
        self._signed_tanh_scaffold_count = signed_tanh_scaffold_count
        self._future_utility_mix = future_utility_mix
        self._future_utility_trace_decay = future_utility_trace_decay
        self._future_utility_trace_mode = future_utility_trace_mode
        self._future_utility_normalization = future_utility_normalization
        self._future_utility_normalization_decay = future_utility_normalization_decay
        self._future_utility_rare_task_power = future_utility_rare_task_power
        self._future_utility_task_activity_decay = future_utility_task_activity_decay
        self._candidate_scoring_mode = candidate_scoring_mode
        self._candidate_score_trace_decay = candidate_score_trace_decay
        self._candidate_score_energy_epsilon = candidate_score_energy_epsilon
        self._candidate_novelty_weight = candidate_novelty_weight
        self._candidate_novelty_power = candidate_novelty_power
        self._candidate_novelty_floor = candidate_novelty_floor
        self._candidate_novelty_admission_bonus = candidate_novelty_admission_bonus
        self._candidate_selector_mode = candidate_selector
        self._candidate_selector_learning_rate = candidate_selector_learning_rate
        self._candidate_selector_exploration = candidate_selector_exploration
        self._candidate_selector = (
            None
            if candidate_selector == CANDIDATE_SELECTOR_LEGACY
            else FiniteCandidateSelector(
                n_candidates=candidate_count,
                learning_rate=candidate_selector_learning_rate,
                exploration=candidate_selector_exploration,
                update_rule=candidate_selector,
            )
        )
        self._retention_slow_utility_decay = retention_slow_utility_decay
        self._ancestor_utility_backup_decay = ancestor_utility_backup_decay
        self._retention_tanh_min_count = retention_tanh_min_count
        self._retention_product_min_count = retention_product_min_count
        self._operation_prior = operation_prior
        self._learn_generator_resources = learn_generator_resources
        self._generator_resource_contexts = generator_resource_contexts
        self._generator_resource_learning_rate = generator_resource_learning_rate
        self._generator_resource_discount = generator_resource_discount
        self._generator_resource_exploration = generator_resource_exploration
        self._generator_resource_advantage_clip = generator_resource_advantage_clip
        self._generator_resource_cost_weight = generator_resource_cost_weight
        self._generator_resource_update_rule = generator_resource_update_rule
        self._generator_resource_promotion_credit = generator_resource_promotion_credit
        self._generator_resource_initial_preferences = (
            generator_resource_initial_preferences
        )
        self._generator_resource_manager = GeneratorMetaResourceManager(
            policy_names=DEFAULT_GENERATOR_META_POLICY_NAMES,
            op_ids=DEFAULT_GENERATOR_META_OP_IDS,
            parent_modes=DEFAULT_GENERATOR_META_PARENT_MODES,
            replacement_multipliers=DEFAULT_GENERATOR_META_REPLACEMENT_MULTIPLIERS,
            promotion_margin_multipliers=(
                DEFAULT_GENERATOR_META_PROMOTION_MARGIN_MULTIPLIERS
            ),
            candidate_min_age_multipliers=(
                DEFAULT_GENERATOR_META_CANDIDATE_MIN_AGE_MULTIPLIERS
            ),
            imprint_scales=DEFAULT_GENERATOR_META_IMPRINT_SCALES,
            n_contexts=generator_resource_contexts,
            learning_rate=generator_resource_learning_rate,
            discount=generator_resource_discount,
            exploration=generator_resource_exploration,
            cost_weight=generator_resource_cost_weight,
            advantage_clip=generator_resource_advantage_clip,
            update_rule=generator_resource_update_rule,
            initial_preferences=generator_resource_initial_preferences,
        )

    @property
    def n_features(self) -> int:
        """Number of active features."""
        return self._n_features

    @property
    def n_tasks(self) -> int:
        """Number of output tasks."""
        return self._n_tasks

    @property
    def max_depth(self) -> int:
        """Maximum allowed topological depth."""
        return self._max_depth

    def to_config(self) -> dict[str, Any]:
        """Serialize learner configuration."""
        return {
            "type": "CompositionalFeatureLearner",
            "n_features": self._n_features,
            "n_tasks": self._n_tasks,
            "candidate_count": self._candidate_count,
            "step_size_output": self._step_size_output,
            "step_size_theta": self._step_size_theta,
            "utility_decay": self._utility_decay,
            "replacement_interval": self._replacement_interval,
            "min_feature_age": self._min_feature_age,
            "candidate_min_age": self._candidate_min_age,
            "promotion_margin": self._promotion_margin,
            "promotion_blend": self._promotion_blend,
            "promotion_output_mode": self._promotion_output_mode,
            "max_depth": self._max_depth,
            "topology_headroom_reserve": self._topology_headroom_reserve,
            "topology_left_pack_destinations": (
                self._topology_left_pack_destinations
            ),
            "use_obgd": self._use_obgd,
            "obgd_kappa": self._obgd_kappa,
            "generation_strategy": self._generation_strategy,
            "parent_temperature": self._parent_temperature,
            "parent_novelty_weight": self._parent_novelty_weight,
            "parent_depth_prior": self._parent_depth_prior,
            "retention_depth_bonus": self._retention_depth_bonus,
            "residual_guidance": self._residual_guidance,
            "candidate_imprint_scale": self._candidate_imprint_scale,
            "train_candidate_theta": self._train_candidate_theta,
            "signed_tanh_scaffold_count": self._signed_tanh_scaffold_count,
            "future_utility_mix": self._future_utility_mix,
            "future_utility_trace_decay": self._future_utility_trace_decay,
            "future_utility_trace_mode": self._future_utility_trace_mode,
            "future_utility_normalization": self._future_utility_normalization,
            "future_utility_normalization_decay": (
                self._future_utility_normalization_decay
            ),
            "future_utility_rare_task_power": self._future_utility_rare_task_power,
            "future_utility_task_activity_decay": (
                self._future_utility_task_activity_decay
            ),
            "candidate_scoring_mode": self._candidate_scoring_mode,
            "candidate_score_trace_decay": self._candidate_score_trace_decay,
            "candidate_score_energy_epsilon": (
                self._candidate_score_energy_epsilon
            ),
            "candidate_novelty_weight": self._candidate_novelty_weight,
            "candidate_novelty_power": self._candidate_novelty_power,
            "candidate_novelty_floor": self._candidate_novelty_floor,
            "candidate_novelty_admission_bonus": (
                self._candidate_novelty_admission_bonus
            ),
            "candidate_selector": self._candidate_selector_mode,
            "candidate_selector_learning_rate": (
                self._candidate_selector_learning_rate
            ),
            "candidate_selector_exploration": self._candidate_selector_exploration,
            "retention_slow_utility_decay": self._retention_slow_utility_decay,
            "ancestor_utility_backup_decay": self._ancestor_utility_backup_decay,
            "retention_tanh_min_count": self._retention_tanh_min_count,
            "retention_product_min_count": self._retention_product_min_count,
            "operation_prior": (
                None if self._operation_prior is None else list(self._operation_prior)
            ),
            "learn_generator_resources": self._learn_generator_resources,
            "generator_resource_contexts": self._generator_resource_contexts,
            "generator_resource_learning_rate": (
                self._generator_resource_learning_rate
            ),
            "generator_resource_discount": self._generator_resource_discount,
            "generator_resource_exploration": self._generator_resource_exploration,
            "generator_resource_advantage_clip": (
                self._generator_resource_advantage_clip
            ),
            "generator_resource_cost_weight": self._generator_resource_cost_weight,
            "generator_resource_update_rule": self._generator_resource_update_rule,
            "generator_resource_promotion_credit": (
                self._generator_resource_promotion_credit
            ),
            "generator_resource_initial_preferences": (
                None
                if self._generator_resource_initial_preferences is None
                else list(self._generator_resource_initial_preferences)
            ),
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CompositionalFeatureLearner":
        """Reconstruct learner from ``to_config`` output."""
        config = dict(config)
        config.pop("type", None)
        return cls(**config)

    def _init_active_slot(
        self,
        slot: int,
        key: Array,
        feature_dim: int,
    ) -> tuple[int, int, int, Array, int]:
        """Generate Python-side initialization for one composed slot.

        Used by ``init`` to set up randomly composed features above the
        raw-input prefix.  Returns ``(op, parent_a, parent_b, theta, depth)``
        as plain Python / JAX arrays of static shapes.
        """
        # Choose an op uniformly from the composing ops (skip OP_RAW).
        sub_keys = jr.split(key, 3)
        op_key, parent_key, theta_key = sub_keys
        op = int(jr.randint(op_key, (), 1, NUM_OPS))
        # Pick parents uniformly from earlier slots.  At minimum, we have
        # the raw-input prefix [0, feature_dim) and any earlier composed
        # slots [feature_dim, slot).
        max_parent = max(slot, 1)
        parents = jr.randint(parent_key, (2,), 0, max_parent)
        a = int(parents[0])
        b = int(parents[1])
        theta = 0.5 * jr.normal(theta_key, (2,), dtype=jnp.float32)
        # Depth is computed from parents in init below; return a neutral
        # seed value here and let init compute the precise depth array.
        return op, a, b, theta, 1

    def init(self, feature_dim: int, key: Array) -> CompositionalFeatureState:
        """Initialize the active and candidate banks.

        The first ``feature_dim`` slots are ``OP_RAW`` features that simply
        expose raw observation entries.  Remaining active slots are random
        compositions of earlier slots; candidates are similarly random
        compositions of the active raw-input slots.
        """
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        if self._n_features < feature_dim:
            raise ValueError(
                "n_features must be at least feature_dim so raw-input slots fit"
            )
        if self._generation_strategy == GENERATION_DOVETAIL_PRODUCT_COVERAGE:
            cycle = _dovetail_product_coverage_cycle(
                n_features=self._n_features,
                feature_dim=feature_dim,
            )
            if self._replacement_interval * cycle > _UINT16_BASE:
                raise ValueError(
                    "16-bit-limb modulus replacement_interval * cycle must be <= 65536"
                )

        n_features = self._n_features

        # Active slot fields, built in Python with static shapes.
        ops = [OP_RAW] * n_features
        parent_a = list(range(feature_dim)) + [0] * (n_features - feature_dim)
        parent_b = [-1] * n_features
        depth = [0] * n_features
        theta_arr = jnp.zeros((n_features, 2), dtype=jnp.float32)

        key, theta_key = jr.split(key)
        # Pre-allocate randomness for composed slots.
        comp_count = n_features - feature_dim
        if comp_count > 0:
            theta_arr = theta_arr.at[feature_dim:].set(
                0.5 * jr.normal(theta_key, (comp_count, 2), dtype=jnp.float32)
            )

        for slot in range(feature_dim, n_features):
            key, slot_key = jr.split(key)
            op_key, parent_key = jr.split(slot_key)
            if self._generation_strategy in {
                GENERATION_RECURSIVE_PRODUCT,
                GENERATION_ROBUST_RECURSIVE,
                GENERATION_DOVETAIL_PRODUCT_COVERAGE,
            }:
                pair_parents = [
                    (left, right)
                    for left in range(feature_dim)
                    for right in range(left + 1, feature_dim)
                ]
                if self._generation_strategy == GENERATION_ROBUST_RECURSIVE:
                    pair_parents = [
                        *pair_parents,
                        *((idx, idx) for idx in range(feature_dim)),
                    ]
                if pair_parents:
                    offset = slot - feature_dim
                    if offset < len(pair_parents) or self._max_depth < 2:
                        a, b = pair_parents[offset % len(pair_parents)]
                        ops[slot] = OP_PRODUCT
                    elif (
                        self._generation_strategy == GENERATION_ROBUST_RECURSIVE
                        and offset
                        < len(pair_parents) + self._signed_tanh_scaffold_count
                    ):
                        signed_pairs = [
                            (left, right, sign_a, sign_b)
                            for left in range(feature_dim)
                            for right in range(left + 1, feature_dim)
                            for sign_a, sign_b in (
                                (1.0, -1.0),
                                (-1.0, 1.0),
                                (1.0, 1.0),
                                (-1.0, -1.0),
                            )
                        ]
                        tanh_offset = offset - len(pair_parents)
                        left, right, sign_a, sign_b = signed_pairs[
                            tanh_offset % len(signed_pairs)
                        ]
                        a, b = left, right
                        ops[slot] = OP_TANH
                        theta_arr = theta_arr.at[slot].set(
                            jnp.array([sign_a, sign_b], dtype=jnp.float32)
                        )
                    else:
                        depth_offset = (
                            offset
                            - len(pair_parents)
                            - (
                                self._signed_tanh_scaffold_count
                                if self._generation_strategy
                                == GENERATION_ROBUST_RECURSIVE
                                else 0
                            )
                        )
                        pair_slot = feature_dim + depth_offset % len(pair_parents)
                        raw_parent = (
                            depth_offset // len(pair_parents)
                        ) % feature_dim
                        a, b = pair_slot, raw_parent
                        ops[slot] = (
                            OP_PRODUCT
                            if self._generation_strategy
                            in {
                                GENERATION_ROBUST_RECURSIVE,
                                GENERATION_DOVETAIL_PRODUCT_COVERAGE,
                            }
                            else OP_SUM
                        )
                else:
                    a, b = 0, 0
                    ops[slot] = OP_PRODUCT
            else:
                ops[slot] = int(jr.randint(op_key, (), 1, NUM_OPS))
                # Parents must have a strictly smaller slot index; this gives a
                # valid topological order under index iteration.  Restrict to
                # those whose depth + 1 stays within max_depth.
                max_parent_excl = slot
                eligible = [
                    p
                    for p in range(max_parent_excl)
                    if depth[p] + 1 <= self._max_depth
                ]
                if not eligible:
                    # Fall back to a raw-input slot.
                    eligible = list(range(min(feature_dim, max_parent_excl)))
                    if not eligible:
                        eligible = [0]
                choices = jr.randint(parent_key, (2,), 0, len(eligible))
                a = eligible[int(choices[0])]
                b = eligible[int(choices[1])]
            parent_a[slot] = a
            parent_b[slot] = b
            depth[slot] = max(depth[a], depth[b]) + 1

        active_state = {
            "ops": jnp.asarray(ops, dtype=jnp.int32),
            "parent_a": jnp.asarray(parent_a, dtype=jnp.int32),
            "parent_b": jnp.asarray(parent_b, dtype=jnp.int32),
            "theta": theta_arr,
            "depth": jnp.asarray(depth, dtype=jnp.int32),
        }

        # Candidates: each candidate is a random composition referring to
        # active feature slots only.  Their "depth" is recorded as one more
        # than the max of the referenced active depths so promotion can
        # later check the depth budget.
        cand_count = self._candidate_count
        cand_ops = [OP_RAW] * cand_count
        cand_parent_a = [0] * cand_count
        cand_parent_b = [-1] * cand_count
        cand_depth = [0] * cand_count
        cand_theta = jnp.zeros((cand_count, 2), dtype=jnp.float32)
        if cand_count > 0:
            key, cand_theta_key = jr.split(key)
            cand_theta = 0.5 * jr.normal(cand_theta_key, (cand_count, 2), dtype=jnp.float32)
            for i in range(cand_count):
                key, c_key = jr.split(key)
                op_key, parent_key = jr.split(c_key)
                if self._generation_strategy in {
                    GENERATION_RECURSIVE_PRODUCT,
                    GENERATION_ROBUST_RECURSIVE,
                    GENERATION_DOVETAIL_PRODUCT_COVERAGE,
                }:
                    composed_parents = [
                        p
                        for p in range(feature_dim, n_features)
                        if 1 <= depth[p] + 1 <= self._max_depth
                    ]
                    raw_parents = list(range(feature_dim))
                    if composed_parents and raw_parents:
                        a = composed_parents[i % len(composed_parents)]
                        b = raw_parents[(i // len(composed_parents)) % len(raw_parents)]
                    else:
                        a, b = 0, 0
                    cand_ops[i] = OP_PRODUCT
                else:
                    cand_ops[i] = int(jr.randint(op_key, (), 1, NUM_OPS))
                    # Candidates pull parents from active slots only.
                    eligible = [
                        p
                        for p in range(n_features)
                        if depth[p] + 1 <= self._max_depth
                    ]
                    if not eligible:
                        eligible = list(range(feature_dim))
                    choices = jr.randint(parent_key, (2,), 0, len(eligible))
                    a = eligible[int(choices[0])]
                    b = eligible[int(choices[1])]
                cand_parent_a[i] = a
                cand_parent_b[i] = b
                cand_depth[i] = max(depth[a], depth[b]) + 1

        return CompositionalFeatureState(
            key=key,
            ops=active_state["ops"],
            parent_a=active_state["parent_a"],
            parent_b=active_state["parent_b"],
            theta=active_state["theta"],
            depth=active_state["depth"],
            output_weights=jnp.zeros(
                (self._n_tasks, n_features), dtype=jnp.float32
            ),
            output_bias=jnp.zeros(self._n_tasks, dtype=jnp.float32),
            utilities=jnp.zeros(n_features, dtype=jnp.float32),
            utility_contribution_trace=jnp.zeros(
                (self._n_tasks, n_features), dtype=jnp.float32
            ),
            utility_error_trace=jnp.zeros(self._n_tasks, dtype=jnp.float32),
            utility_feature_trace=jnp.zeros(n_features, dtype=jnp.float32),
            utility_feature_energy_trace=jnp.zeros(n_features, dtype=jnp.float32),
            utility_signal_second_moment=jnp.zeros(n_features, dtype=jnp.float32),
            feature_score_residual_trace=jnp.zeros(
                (self._n_tasks, n_features), dtype=jnp.float32
            ),
            feature_score_energy_trace=jnp.zeros(n_features, dtype=jnp.float32),
            retention_slow_utilities=jnp.zeros(n_features, dtype=jnp.float32),
            task_activity_ema=jnp.zeros(self._n_tasks, dtype=jnp.float32),
            ages=jnp.zeros(n_features, dtype=jnp.int32),
            candidate_ops=jnp.asarray(cand_ops, dtype=jnp.int32),
            candidate_parent_a=jnp.asarray(cand_parent_a, dtype=jnp.int32),
            candidate_parent_b=jnp.asarray(cand_parent_b, dtype=jnp.int32),
            candidate_theta=cand_theta,
            candidate_depth=jnp.asarray(cand_depth, dtype=jnp.int32),
            candidate_output_weights=jnp.zeros(
                (self._n_tasks, cand_count), dtype=jnp.float32
            ),
            candidate_utilities=jnp.zeros(cand_count, dtype=jnp.float32),
            candidate_utility_contribution_trace=jnp.zeros(
                (self._n_tasks, cand_count), dtype=jnp.float32
            ),
            candidate_utility_feature_trace=jnp.zeros(
                cand_count, dtype=jnp.float32
            ),
            candidate_utility_feature_energy_trace=jnp.zeros(
                cand_count, dtype=jnp.float32
            ),
            candidate_utility_signal_second_moment=jnp.zeros(
                cand_count, dtype=jnp.float32
            ),
            candidate_score_residual_trace=jnp.zeros(
                (self._n_tasks, cand_count), dtype=jnp.float32
            ),
            candidate_score_energy_trace=jnp.zeros(cand_count, dtype=jnp.float32),
            candidate_retention_slow_utilities=jnp.zeros(
                cand_count, dtype=jnp.float32
            ),
            candidate_active_correlation_trace=jnp.zeros(
                (cand_count, n_features), dtype=jnp.float32
            ),
            candidate_ages=jnp.zeros(cand_count, dtype=jnp.int32),
            candidate_selector_log_weights=jnp.zeros(cand_count, dtype=jnp.float32),
            candidate_selector_cumulative_loss=jnp.zeros(
                cand_count, dtype=jnp.float32
            ),
            candidate_selector_action_counts=jnp.zeros(
                cand_count, dtype=jnp.float32
            ),
            feature_generator_policy=jnp.zeros(n_features, dtype=jnp.int32),
            candidate_generator_policy=jnp.zeros(cand_count, dtype=jnp.int32),
            generator_resource_state=self._generator_resource_manager.init(),
            replacement_accumulator=jnp.array(0.0, dtype=jnp.float32),
            step_count=jnp.array(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
            replacement_phase=jnp.array(0, dtype=jnp.int32),
            birth_timestamp=time.time(),
            uptime_s=0.0,
        )

    def _candidate_features(
        self,
        state: CompositionalFeatureState,
        active_values: Array,
        observation: Array,
    ) -> Array:
        """Compute candidate feature values by referencing active slots.

        Candidate parents always point into the active feature bank, so each
        candidate is evaluated as a single op over those active values plus,
        for ``OP_RAW`` candidates, the raw observation directly.
        """
        feature_dim = observation.shape[0]
        n_features = self._n_features
        cand_count = self._candidate_count
        if cand_count == 0:
            return jnp.zeros((0,), dtype=jnp.float32)

        safe_a_obs = jnp.clip(state.candidate_parent_a, 0, feature_dim - 1)
        safe_a_feat = jnp.clip(state.candidate_parent_a, 0, n_features - 1)
        safe_b_feat = jnp.clip(state.candidate_parent_b, 0, n_features - 1)

        raw = observation[safe_a_obs]
        val_a = active_values[safe_a_feat]
        val_b = jnp.where(
            state.candidate_parent_b >= 0,
            active_values[safe_b_feat],
            0.0,
        )

        product = val_a * val_b
        summ = val_a + val_b
        pre_tanh = (
            state.candidate_theta[:, 0] * val_a
            + state.candidate_theta[:, 1] * val_b
        )
        tanh_val = jnp.tanh(pre_tanh)
        gated = val_a * jax.nn.sigmoid(val_b)

        ops = state.candidate_ops
        values = jnp.select(
            [
                ops == OP_RAW,
                ops == OP_PRODUCT,
                ops == OP_SUM,
                ops == OP_TANH,
                ops == OP_GATED,
            ],
            [raw, product, summ, tanh_val, gated],
            default=jnp.zeros(cand_count, dtype=jnp.float32),
        )
        return jnp.clip(values, -FEATURE_VALUE_CLIP, FEATURE_VALUE_CLIP)

    def _strategy_parent_mode(self) -> Array:
        """Return the parent-selection mode for the fixed generation strategy."""
        if self._generation_strategy == GENERATION_UNIFORM:
            mode = PARENT_MODE_UNIFORM
        elif self._generation_strategy == GENERATION_MUTATION:
            mode = PARENT_MODE_MUTATION
        elif self._generation_strategy in {
            GENERATION_RESIDUAL_IMPRINT,
            GENERATION_ROBUST_RECURSIVE,
            GENERATION_DOVETAIL_PRODUCT_COVERAGE,
        }:
            mode = PARENT_MODE_RESIDUAL_IMPRINT
        else:
            mode = PARENT_MODE_UTILITY
        return jnp.array(mode, dtype=jnp.int32)

    def _op_logits(self, forced_op: Array | None = None) -> Array:
        """Return generation logits for composing op types.

        Probability vectors below are ordered ``[raw, product, sum, tanh,
        gated]``.  ``OP_RAW`` always gets zero mass: raw slots occupy the
        fixed prefix of the bank and are never generated.  The per-strategy
        priors are hand-tuned, not learned; strategies aimed at
        product-structured recursive targets put most or all mass on
        ``OP_PRODUCT``, while ``residual_imprint`` puts equal (0.35) mass on
        ``OP_PRODUCT`` and the parameterized ``OP_TANH``.
        """
        if forced_op is not None:
            op_ids = jnp.arange(NUM_OPS, dtype=jnp.int32)
            return jnp.where(op_ids == forced_op, 0.0, -jnp.inf)
        if self._operation_prior is not None:
            probs = jnp.asarray(self._operation_prior, dtype=jnp.float32)
            probs = probs / jnp.sum(probs)
            return jnp.where(probs > 0.0, jnp.log(probs), -jnp.inf)
        if self._generation_strategy == GENERATION_RECURSIVE_PRODUCT:
            probs = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
            return jnp.where(probs > 0.0, jnp.log(probs), -jnp.inf)
        if self._generation_strategy == GENERATION_DOVETAIL_PRODUCT_COVERAGE:
            probs = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
            return jnp.where(probs > 0.0, jnp.log(probs), -jnp.inf)
        if self._generation_strategy == GENERATION_ROBUST_RECURSIVE:
            probs = jnp.array([0.0, 0.5, 0.1, 0.3, 0.1], dtype=jnp.float32)
            return jnp.where(probs > 0.0, jnp.log(probs), -jnp.inf)
        if self._generation_strategy == GENERATION_MUTATION:
            probs = jnp.array([0.0, 0.55, 0.15, 0.15, 0.15], dtype=jnp.float32)
        elif self._generation_strategy == GENERATION_RESIDUAL_IMPRINT:
            probs = jnp.array([0.0, 0.35, 0.15, 0.35, 0.15], dtype=jnp.float32)
        else:
            probs = jnp.array([0.0, 0.4, 0.2, 0.2, 0.2], dtype=jnp.float32)
        return jnp.where(probs > 0.0, jnp.log(probs), -jnp.inf)

    def _parent_logits(
        self,
        eligible: Array,
        utilities: Array,
        feature_values: Array | None = None,
        feature_credit: Array | None = None,
        depth: Array | None = None,
        ages: Array | None = None,
        parent_mode: Array | None = None,
    ) -> Array:
        """Return masked parent logits for the configured search strategy."""
        mode = self._strategy_parent_mode() if parent_mode is None else parent_mode
        uniform_logits = jnp.zeros_like(utilities, dtype=jnp.float32)
        utility_scores = utilities + 1e-3
        residual_scores = jnp.zeros_like(utilities, dtype=jnp.float32)
        if feature_values is not None and feature_credit is not None:
            residual_scores = jnp.abs(feature_credit) + 0.05 * jnp.abs(feature_values)
        novelty_scores = jnp.zeros_like(utilities, dtype=jnp.float32)
        if self._parent_novelty_weight > 0.0:
            inverse_utility = 1.0 / jnp.sqrt(jnp.maximum(utility_scores, 1e-6))
            age_bonus = jnp.zeros_like(utilities, dtype=jnp.float32)
            if ages is not None:
                age_bonus = 1.0 / jnp.sqrt(ages.astype(jnp.float32) + 1.0)
            novelty_scores = self._parent_novelty_weight * (
                0.5 * inverse_utility + age_bonus
            )
        depth_scores = jnp.zeros_like(utilities, dtype=jnp.float32)
        if self._parent_depth_prior > 0.0 and depth is not None:
            depth_scores = self._parent_depth_prior * jnp.log1p(
                depth.astype(jnp.float32)
            )
        guided_scores = (
            utility_scores
            + self._residual_guidance * residual_scores
            + novelty_scores
            + depth_scores
        )
        utility_logits = (
            jnp.log(jnp.maximum(utility_scores, 1e-6)) / self._parent_temperature
        )
        residual_logits = (
            jnp.log(jnp.maximum(guided_scores, 1e-6)) / self._parent_temperature
        )
        logits = jnp.select(
            [
                mode == PARENT_MODE_UNIFORM,
                mode == PARENT_MODE_RESIDUAL_IMPRINT,
            ],
            [uniform_logits, residual_logits],
            default=utility_logits,
        )
        return jnp.where(eligible, logits, -1e9)

    def _partner_logits(
        self,
        eligible: Array,
        depth: Array,
        utilities: Array,
        ages: Array | None = None,
        parent_mode: Array | None = None,
    ) -> Array:
        """Return logits for the second parent in mutation-like strategies."""
        mode = self._strategy_parent_mode() if parent_mode is None else parent_mode
        shallow_logits = jnp.where(eligible, -0.25 * depth.astype(jnp.float32), -1e9)
        default_logits = self._parent_logits(
            eligible,
            utilities,
            depth=depth,
            ages=ages,
            parent_mode=mode,
        )
        return jnp.where(
            (mode == PARENT_MODE_MUTATION) | (mode == PARENT_MODE_RESIDUAL_IMPRINT),
            shallow_logits,
            default_logits,
        )

    def _candidate_value_from_parts(
        self,
        op: Array,
        parent_a: Array,
        parent_b: Array,
        theta: Array,
        active_values: Array,
        observation: Array,
    ) -> Array:
        """Evaluate one generated candidate against the current observation."""
        feature_dim = observation.shape[0]
        safe_a_obs = jnp.clip(parent_a, 0, feature_dim - 1)
        safe_a_feat = jnp.clip(parent_a, 0, self._n_features - 1)
        safe_b_feat = jnp.clip(parent_b, 0, self._n_features - 1)

        raw = observation[safe_a_obs]
        val_a = active_values[safe_a_feat]
        val_b = jnp.where(parent_b >= 0, active_values[safe_b_feat], 0.0)
        product = val_a * val_b
        summ = val_a + val_b
        tanh_val = jnp.tanh(theta[0] * val_a + theta[1] * val_b)
        gated = val_a * jax.nn.sigmoid(val_b)

        value = jnp.select(
            [
                op == OP_RAW,
                op == OP_PRODUCT,
                op == OP_SUM,
                op == OP_TANH,
                op == OP_GATED,
            ],
            [raw, product, summ, tanh_val, gated],
            default=jnp.array(0.0, dtype=jnp.float32),
        )
        return jnp.clip(value, -FEATURE_VALUE_CLIP, FEATURE_VALUE_CLIP)

    def _initial_candidate_output_weights(
        self,
        op: Array,
        parent_a: Array,
        parent_b: Array,
        theta: Array,
        active_values: Array,
        observation: Array,
        errors: Array,
        active_count: Array,
        imprint_scale: Array | None = None,
    ) -> Array:
        """Initialize fresh candidate output weights from the current residual.

        Delegates to :func:`_imprint_candidate_output_weights` for the damped
        one-sample least-squares imprint formula.
        """
        scale = (
            jnp.asarray(self._candidate_imprint_scale, dtype=jnp.float32)
            if imprint_scale is None
            else imprint_scale
        )
        if self._candidate_imprint_scale == 0.0 and imprint_scale is None:
            return jnp.zeros((self._n_tasks,), dtype=jnp.float32)
        candidate_value = self._candidate_value_from_parts(
            op,
            parent_a,
            parent_b,
            theta,
            active_values,
            observation,
        )
        return _imprint_candidate_output_weights(
            errors,
            candidate_value,
            active_count,
            scale,
        )

    def _promoted_output_weights(
        self,
        active_weights: Array,
        candidate_weights: Array,
    ) -> Array:
        """Compute output weights for a promoted candidate slot."""
        if self._promotion_output_mode == PROMOTION_BLEND:
            return (
                (1.0 - self._promotion_blend) * active_weights
                + self._promotion_blend * candidate_weights
            )
        return self._promotion_blend * candidate_weights

    def _future_utility_signal(
        self,
        errors: Array,
        feature_values: Array,
        active_mask: Array,
        active_count: Array,
        task_activity_ema: Array,
        contribution_trace: Array,
        error_trace: Array,
        feature_trace: Array,
        feature_energy_trace: Array,
    ) -> tuple[Array, Array, Array, Array, Array]:
        """Predict traced output-loss reduction for each feature slot."""
        if self._future_utility_trace_mode == "marginal":
            (
                reductions,
                new_error_trace,
                new_feature_trace,
                new_feature_energy_trace,
            ) = trace_output_loss_reduction(
                errors,
                feature_values,
                active_mask,
                self._step_size_output,
                active_count,
                error_trace,
                feature_trace,
                feature_energy_trace,
                self._future_utility_trace_decay,
            )
            new_contribution_trace = contribution_trace
        else:
            reductions, new_contribution_trace, new_feature_energy_trace = (
                contribution_trace_output_loss_reduction(
                    errors,
                    feature_values,
                    active_mask,
                    self._step_size_output,
                    active_count,
                    contribution_trace,
                    feature_energy_trace,
                    self._future_utility_trace_decay,
                )
            )
            new_error_trace = error_trace
            new_feature_trace = feature_trace
        if self._future_utility_rare_task_power > 0.0:
            frequency_floor = jnp.array(
                1.0 - self._future_utility_task_activity_decay,
                dtype=jnp.float32,
            )
            rare_weights = jnp.power(
                1.0 / jnp.maximum(task_activity_ema, frequency_floor),
                self._future_utility_rare_task_power,
            )
            reductions = reductions * rare_weights[:, None]
        return (
            jnp.mean(reductions, axis=0),
            new_contribution_trace,
            new_error_trace,
            new_feature_trace,
            new_feature_energy_trace,
        )

    def _candidate_future_utility_signal(
        self,
        errors: Array,
        feature_values: Array,
        active_mask: Array,
        active_count: Array,
        task_activity_ema: Array,
        contribution_trace: Array,
        error_trace: Array,
        feature_trace: Array,
        feature_energy_trace: Array,
    ) -> tuple[Array, Array, Array, Array]:
        """Predict traced output-loss reduction for candidate slots."""
        if self._future_utility_trace_mode == "marginal":
            reductions, _, new_feature_trace, new_feature_energy_trace = (
                trace_output_loss_reduction(
                    errors,
                    feature_values,
                    active_mask,
                    self._step_size_output,
                    active_count,
                    error_trace,
                    feature_trace,
                    feature_energy_trace,
                    self._future_utility_trace_decay,
                )
            )
            new_contribution_trace = contribution_trace
        else:
            reductions, new_contribution_trace, new_feature_energy_trace = (
                contribution_trace_output_loss_reduction(
                    errors,
                    feature_values,
                    active_mask,
                    self._step_size_output,
                    active_count,
                    contribution_trace,
                    feature_energy_trace,
                    self._future_utility_trace_decay,
                )
            )
            new_feature_trace = feature_trace
        if self._future_utility_rare_task_power > 0.0:
            frequency_floor = jnp.array(
                1.0 - self._future_utility_task_activity_decay,
                dtype=jnp.float32,
            )
            rare_weights = jnp.power(
                1.0 / jnp.maximum(task_activity_ema, frequency_floor),
                self._future_utility_rare_task_power,
            )
            reductions = reductions * rare_weights[:, None]
        return (
            jnp.mean(reductions, axis=0),
            new_contribution_trace,
            new_feature_trace,
            new_feature_energy_trace,
        )

    def _mixed_utility_signal(
        self,
        current_signal: Array,
        future_signal: Array,
    ) -> Array:
        """Blend the backward-looking utility with predicted future utility."""
        if self._future_utility_mix == 0.0:
            return current_signal
        return (
            (1.0 - self._future_utility_mix) * current_signal
            + self._future_utility_mix * future_signal
        )

    def _retention_slow_utility(
        self,
        previous_slow_utility: Array,
        utility_signal: Array,
    ) -> Array:
        """Update opt-in slow utility for hysteretic deletion."""
        if self._retention_slow_utility_decay == 0.0:
            return previous_slow_utility
        decay = jnp.asarray(self._retention_slow_utility_decay, dtype=jnp.float32)
        return decay * previous_slow_utility + (1.0 - decay) * utility_signal

    def _retention_score(
        self,
        fast_utility: Array,
        slow_utility: Array,
    ) -> Array:
        """Return the utility score used for opt-in delayed deletion."""
        if self._retention_slow_utility_decay == 0.0:
            return fast_utility
        return jnp.maximum(fast_utility, slow_utility)

    def _ancestor_backed_retention_score(
        self,
        direct_score: Array,
        ops: Array,
        parent_a: Array,
        parent_b: Array,
    ) -> Array:
        """Apply deterministic transitive max backup over the active DAG."""

        if self._ancestor_utility_backup_decay == 0.0:
            return direct_score
        decay = jnp.asarray(self._ancestor_utility_backup_decay, dtype=jnp.float32)

        def back_up_child(reverse_index: int, backed: Array) -> Array:
            child = self._n_features - 1 - reverse_index
            child_credit = decay * backed[child]
            composed = ops[child] != OP_RAW
            pa = parent_a[child]
            pb = parent_b[child]
            safe_pa = jnp.clip(pa, 0, self._n_features - 1)
            safe_pb = jnp.clip(pb, 0, self._n_features - 1)
            valid_pa = composed & (pa >= 0) & (pa < child)
            valid_pb = composed & (pb >= 0) & (pb < child)
            backed = backed.at[safe_pa].set(
                jnp.where(
                    valid_pa,
                    jnp.maximum(backed[safe_pa], child_credit),
                    backed[safe_pa],
                )
            )
            return backed.at[safe_pb].set(
                jnp.where(
                    valid_pb,
                    jnp.maximum(backed[safe_pb], child_credit),
                    backed[safe_pb],
                )
            )

        return jax.lax.fori_loop(0, self._n_features, back_up_child, direct_score)

    def _candidate_novelty_scores_from_statistics(
        self,
        candidate_energy_trace: Array,
        active_energy_trace: Array,
        candidate_active_correlation_trace: Array,
    ) -> Array:
        """Return evidence-gated correlation novelty in ``[0, 1]``."""

        if self._candidate_count == 0:
            return jnp.zeros((0,), dtype=jnp.float32)
        denom = jnp.sqrt(
            candidate_energy_trace[:, None] * active_energy_trace[None, :]
            + self._candidate_score_energy_epsilon
        )
        correlations = jnp.clip(
            jnp.abs(candidate_active_correlation_trace) / denom,
            0.0,
            1.0,
        )
        novelty = 1.0 - jnp.max(correlations, axis=1)
        has_candidate_evidence = (
            candidate_energy_trace > self._candidate_score_energy_epsilon
        )
        has_active_evidence = jnp.any(
            active_energy_trace > self._candidate_score_energy_epsilon
        )
        return jnp.where(
            has_candidate_evidence & has_active_evidence,
            jnp.power(jnp.clip(novelty, 0.0, 1.0), self._candidate_novelty_power),
            0.0,
        )

    def _ranking_topology_valid(
        self,
        state: CompositionalFeatureState,
        feature_dim: int,
    ) -> Bool[Array, ""]:
        """Validate active/candidate descriptor topology for ranking updates."""

        indices = jnp.arange(self._n_features, dtype=jnp.int32)
        reserved_raw = indices < feature_dim
        active_pa = jnp.clip(state.parent_a, 0, self._n_features - 1)
        active_pb = jnp.clip(state.parent_b, 0, self._n_features - 1)
        active_expected_depth = (
            jnp.maximum(state.depth[active_pa], state.depth[active_pb]) + 1
        )
        active_raw_valid = (
            (state.ops == OP_RAW)
            & (state.parent_a == indices)
            & (state.parent_b == -1)
            & (state.depth == 0)
        )
        active_composed_valid = (
            (state.ops > OP_RAW)
            & (state.ops < NUM_OPS)
            & (state.parent_a >= 0)
            & (state.parent_a < indices)
            & (state.parent_b >= 0)
            & (state.parent_b < indices)
            & (state.depth == active_expected_depth)
            & (state.depth >= 1)
            & (state.depth <= self._max_depth)
        )
        active_valid = jnp.all(
            jnp.where(reserved_raw, active_raw_valid, active_composed_valid)
        )

        candidate_pa = jnp.clip(
            state.candidate_parent_a, 0, self._n_features - 1
        )
        candidate_pb = jnp.clip(
            state.candidate_parent_b, 0, self._n_features - 1
        )
        candidate_expected_depth = (
            jnp.maximum(state.depth[candidate_pa], state.depth[candidate_pb]) + 1
        )
        candidate_composed_valid = (
            (state.candidate_ops > OP_RAW)
            & (state.candidate_ops < NUM_OPS)
            & (state.candidate_parent_a >= 0)
            & (state.candidate_parent_a < self._n_features)
            & (state.candidate_parent_b >= 0)
            & (state.candidate_parent_b < self._n_features)
            & (state.candidate_depth == candidate_expected_depth)
            & (state.candidate_depth >= 1)
            & (state.candidate_depth <= self._max_depth)
        )
        candidate_valid = jnp.all(candidate_composed_valid)
        return jnp.asarray(
            (0 < feature_dim <= self._n_features), dtype=jnp.bool_
        ) & active_valid & candidate_valid

    def _ranking_state_finite(
        self,
        state: CompositionalFeatureState,
    ) -> Bool[Array, ""]:
        """Validate all floating learner state consulted by an update."""

        finite = jnp.asarray(True, dtype=jnp.bool_)
        for value in (
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
            state.birth_timestamp,
            state.uptime_s,
        ):
            finite = finite & jnp.all(jnp.isfinite(jnp.asarray(value)))
        return finite

    @functools.partial(jax.jit, static_argnums=(0, 2))
    def ranking_diagnostics(
        self,
        state: CompositionalFeatureState,
        feature_dim: int,
    ) -> CompositionalRankingDiagnostics:
        """Return pre-update direct, backed, and novelty-augmented scores."""

        direct_active = self._retention_score(
            state.utilities,
            state.retention_slow_utilities,
        )
        backed_active = self._ancestor_backed_retention_score(
            direct_active,
            state.ops,
            state.parent_a,
            state.parent_b,
        )
        direct_candidate = self._retention_score(
            state.candidate_utilities,
            state.candidate_retention_slow_utilities,
        )
        novelty = self._candidate_novelty_scores_from_statistics(
            state.candidate_score_energy_trace,
            state.feature_score_energy_trace,
            state.candidate_active_correlation_trace,
        )
        augmented_candidate = direct_candidate + jnp.asarray(
            self._candidate_novelty_admission_bonus, dtype=jnp.float32
        ) * novelty
        return CompositionalRankingDiagnostics(
            contract_valid=self._ranking_topology_valid(state, feature_dim)
            & self._ranking_state_finite(state),
            direct_active_scores=direct_active,
            backed_active_scores=backed_active,
            direct_candidate_scores=direct_candidate,
            candidate_novelty_scores=novelty,
            augmented_candidate_scores=augmented_candidate,
            candidate_mature=state.candidate_ages >= self._candidate_min_age,
        )

    def _energy_normalized_residual_score(
        self,
        errors: Array,
        feature_values: Array,
        residual_trace: Array,
        energy_trace: Array,
    ) -> tuple[Array, Array, Array]:
        """Return online matching-pursuit residual scores."""
        trace_decay = jnp.asarray(
            self._candidate_score_trace_decay, dtype=jnp.float32
        )
        new_residual_trace = (
            trace_decay * residual_trace + errors[:, None] * feature_values[None, :]
        )
        new_energy_trace = trace_decay * energy_trace + feature_values * feature_values
        score = jnp.mean(jnp.abs(new_residual_trace), axis=0) / jnp.sqrt(
            new_energy_trace + self._candidate_score_energy_epsilon
        )
        return score, new_residual_trace, new_energy_trace

    def _candidate_novelty_gate(
        self,
        candidate_feature_values: Array,
        active_feature_values: Array,
        candidate_energy_trace: Array,
        active_energy_trace: Array,
        candidate_active_correlation_trace: Array,
    ) -> tuple[Array, Array, Array]:
        """Return direct-score gate, admission novelty, and new correlations."""
        trace_decay = jnp.asarray(
            self._candidate_score_trace_decay, dtype=jnp.float32
        )
        new_correlation_trace = (
            trace_decay * candidate_active_correlation_trace
            + candidate_feature_values[:, None] * active_feature_values[None, :]
        )
        denom = jnp.sqrt(
            candidate_energy_trace[:, None] * active_energy_trace[None, :]
            + self._candidate_score_energy_epsilon
        )
        correlations = jnp.clip(jnp.abs(new_correlation_trace) / denom, 0.0, 1.0)
        max_correlation = jnp.max(correlations, axis=1)
        novelty = 1.0 - max_correlation
        novelty_gate = jnp.power(
            jnp.clip(
                novelty,
                self._candidate_novelty_floor,
                1.0,
            ),
            self._candidate_novelty_power,
        )
        gate = (
            (1.0 - self._candidate_novelty_weight)
            + self._candidate_novelty_weight * novelty_gate
        )
        if self._candidate_novelty_admission_bonus > 0.0:
            admission_novelty = self._candidate_novelty_scores_from_statistics(
                candidate_energy_trace,
                active_energy_trace,
                new_correlation_trace,
            )
        else:
            admission_novelty = jnp.zeros(
                (self._candidate_count,), dtype=jnp.float32
            )
        return gate, admission_novelty, new_correlation_trace

    def _generator_policy_scores(
        self,
        utilities: Array,
        feature_generator_policy: Array,
        candidate_utilities: Array,
        candidate_generator_policy: Array,
    ) -> tuple[Array, Array]:
        """Return mean utility and availability mask per generator policy."""
        policy_ids = jnp.arange(
            self._generator_resource_manager.n_policies,
            dtype=jnp.int32,
        )
        active_matches = feature_generator_policy[None, :] == policy_ids[:, None]
        active_sums = jnp.sum(
            jnp.where(active_matches, utilities[None, :], 0.0),
            axis=1,
        )
        active_counts = jnp.sum(active_matches.astype(jnp.float32), axis=1)
        candidate_matches = (
            candidate_generator_policy[None, :] == policy_ids[:, None]
        )
        candidate_sums = jnp.sum(
            jnp.where(candidate_matches, candidate_utilities[None, :], 0.0),
            axis=1,
        )
        candidate_counts = jnp.sum(candidate_matches.astype(jnp.float32), axis=1)
        counts = active_counts + candidate_counts
        scores = (active_sums + candidate_sums) / jnp.maximum(counts, 1.0)
        return scores, counts > 0.0

    @functools.partial(jax.jit, static_argnums=(0,))
    def constructed_features(
        self,
        state: CompositionalFeatureState,
        observation: Array,
    ) -> Array:
        """Return active compositional feature values for ``observation``.

        These literal compositions are the Step 2 hand-off representation:
        downstream Horde or SARSA learners can consume them as fixed features.
        """
        return _compute_feature_values(
            state.ops,
            state.parent_a,
            state.parent_b,
            state.theta,
            observation,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def augmented_observation(
        self,
        state: CompositionalFeatureState,
        observation: Array,
    ) -> Array:
        """Concatenate raw observation with active compositional features."""
        return jnp.concatenate(
            [observation, self.constructed_features(state, observation)]
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(
        self,
        state: CompositionalFeatureState,
        observation: Array,
    ) -> Array:
        """Predict all tasks from active compositional features."""
        features = self.constructed_features(state, observation)
        result: Array = state.output_weights @ features + state.output_bias
        return result

    def _generate_one(
        self,
        key: Array,
        existing_depth: Array,
        existing_utilities: Array | None = None,
        existing_ages: Array | None = None,
        feature_values: Array | None = None,
        feature_credit: Array | None = None,
        forced_op: Array | None = None,
        parent_mode: Array | None = None,
        coverage_cursor: Array | None = None,
        feature_dim: int | None = None,
        preferred_depth1_parent: Array | None = None,
    ) -> tuple[Array, Array, Array, Array, Array]:
        """Sample a fresh candidate composition.

        The op type is biased toward cheap/non-trivial compositional
        primitives.  Parent selection is controlled by ``generation_strategy``:
        ``"utility"`` biases parents toward high-utility slots; mutation and
        imprint variants anchor one parent on high-score features and sample
        the other from shallow eligible features to encourage local variants
        of useful parents.

        Args:
            key: PRNG key.
            existing_depth: Depths of the active feature bank, shape
                ``(n_features,)``.
            existing_utilities: Optional utility array used to bias parent
                selection; when ``None`` parents are drawn uniformly over
                eligible slots.
            existing_ages: Optional age array used by novelty-biased parent
                selection.
            feature_values: Optional active feature values for one-step
                residual-imprint parent scoring.
            feature_credit: Optional active feature residual credit for
                one-step residual-imprint parent scoring.
            forced_op: Optional op id supplied by a meta-resource policy.
            parent_mode: Optional parent-selection mode supplied by a
                meta-resource policy.
            coverage_cursor: Optional exact curation cursor for
                ``dovetail_product_coverage``. Corrective regeneration omits
                it and retains ordinary recursive product generation.
            feature_dim: Static raw-input dimension required with a coverage
                cursor.
            preferred_depth1_parent: Optional newly admitted depth-1 slot to
                extend on an odd coverage cursor. Invalid preferences use the
                deterministic cyclic next-valid fallback.

        Returns:
            ``(op, parent_a, parent_b, theta, depth)`` as scalar / shape-2
            JAX arrays.
        """
        op_key, pa_key, pb_key, fallback_pa_key, fallback_pb_key, theta_key = jr.split(
            key, 6
        )
        if (
            self._generation_strategy == GENERATION_DOVETAIL_PRODUCT_COVERAGE
            and coverage_cursor is not None
        ):
            if feature_dim is None:
                raise ValueError("coverage generation requires static feature_dim")
            if not 2 <= feature_dim < self._n_features:
                raise ValueError(
                    "coverage generation requires 2 <= feature_dim < n_features"
                )
            raw_pairs = tuple(
                (left, right)
                for left in range(feature_dim)
                for right in range(left + 1, feature_dim)
            )
            pair_left = jnp.asarray(
                tuple(left for left, _ in raw_pairs), dtype=jnp.int32
            )
            pair_right = jnp.asarray(
                tuple(right for _, right in raw_pairs), dtype=jnp.int32
            )
            cursor = jnp.asarray(coverage_cursor, dtype=jnp.int32)
            dovetail_index = cursor // jnp.asarray(2, dtype=jnp.int32)
            pair_index = dovetail_index % len(raw_pairs)
            raw_pair_a = pair_left[pair_index]
            raw_pair_b = pair_right[pair_index]

            generated_slots = self._n_features - feature_dim
            requested_slot = jnp.asarray(feature_dim, dtype=jnp.int32) + (
                dovetail_index % generated_slots
            )
            if preferred_depth1_parent is not None:
                requested_slot = jnp.asarray(
                    preferred_depth1_parent, dtype=jnp.int32
                )
            slot_indices = jnp.arange(self._n_features, dtype=jnp.int32)
            valid_depth1 = (
                (slot_indices >= feature_dim)
                & (existing_depth == 1)
                & (existing_depth + 1 <= self._max_depth)
            )
            cyclic_distance = jnp.mod(
                slot_indices - requested_slot,
                jnp.asarray(self._n_features, dtype=jnp.int32),
            )
            extension_parent = jnp.argmin(
                jnp.where(valid_depth1, cyclic_distance, self._n_features + 1)
            ).astype(jnp.int32)
            has_depth1 = jnp.any(valid_depth1)
            extension_raw = (
                dovetail_index // generated_slots
            ) % jnp.asarray(feature_dim, dtype=jnp.int32)
            odd_cursor = (cursor % jnp.asarray(2, dtype=jnp.int32)) == 1
            use_extension = odd_cursor & has_depth1
            parent_a = jnp.where(use_extension, extension_parent, raw_pair_a)
            parent_b = jnp.where(use_extension, extension_raw, raw_pair_b)
            theta = 0.5 * jr.normal(theta_key, (2,), dtype=jnp.float32)
            proposal_depth = (
                jnp.maximum(existing_depth[parent_a], existing_depth[parent_b]) + 1
            ).astype(jnp.int32)
            return (
                jnp.asarray(OP_PRODUCT, dtype=jnp.int32),
                parent_a,
                parent_b,
                theta,
                proposal_depth,
            )
        recursive_product = (
            self._generation_strategy
            in {
                GENERATION_RECURSIVE_PRODUCT,
                GENERATION_ROBUST_RECURSIVE,
                GENERATION_DOVETAIL_PRODUCT_COVERAGE,
            }
            and forced_op is None
        )
        op = jr.categorical(op_key, self._op_logits(forced_op)).astype(jnp.int32)
        # Eligibility mask: parent depth + 1 <= max_depth.
        eligible = existing_depth + 1 <= self._max_depth
        if existing_utilities is None:
            utilities = jnp.ones_like(existing_depth, dtype=jnp.float32)
        else:
            utilities = existing_utilities
        parent_logits = self._parent_logits(
            eligible,
            utilities,
            feature_values=feature_values,
            feature_credit=feature_credit,
            depth=existing_depth,
            ages=existing_ages,
            parent_mode=parent_mode,
        )
        partner_logits = self._partner_logits(
            eligible,
            existing_depth,
            utilities,
            ages=existing_ages,
            parent_mode=parent_mode,
        )
        a_idx = jr.categorical(pa_key, parent_logits).astype(jnp.int32)
        b_idx = jr.categorical(pb_key, partner_logits).astype(jnp.int32)
        if recursive_product:
            recursive_parent = eligible & (existing_depth >= 1)
            shallow_parent = eligible & (existing_depth == 0)
            has_recursive_parent = jnp.any(recursive_parent)
            has_shallow_parent = jnp.any(shallow_parent)
            recursive_logits = self._parent_logits(
                recursive_parent,
                utilities,
                feature_values=feature_values,
                feature_credit=feature_credit,
                depth=existing_depth,
                ages=existing_ages,
                parent_mode=jnp.array(PARENT_MODE_RESIDUAL_IMPRINT, dtype=jnp.int32),
            )
            recursive_logits = jnp.where(
                has_recursive_parent,
                recursive_logits,
                parent_logits,
            )
            shallow_logits = jnp.where(
                shallow_parent,
                jnp.zeros_like(utilities, dtype=jnp.float32),
                -1e9,
            )
            shallow_logits = jnp.where(
                has_shallow_parent,
                shallow_logits,
                partner_logits,
            )
            recursive_a = jr.categorical(
                fallback_pa_key, recursive_logits
            ).astype(jnp.int32)
            recursive_b = jr.categorical(
                fallback_pb_key, shallow_logits
            ).astype(jnp.int32)
            a_idx = jnp.where(has_recursive_parent, recursive_a, a_idx)
            b_idx = jnp.where(
                has_recursive_parent & has_shallow_parent,
                recursive_b,
                b_idx,
            )
        new_theta = 0.5 * jr.normal(theta_key, (2,), dtype=jnp.float32)
        new_depth = (
            jnp.maximum(existing_depth[a_idx], existing_depth[b_idx]) + 1
        ).astype(jnp.int32)
        return op, a_idx, b_idx, new_theta, new_depth

    def _curation_stage_guidance(
        self,
        ops: Array,
        parent_a: Array,
        parent_b: Array,
        theta: Array,
        output_weights: Array,
        observation: Array,
        errors: Array,
        active_count: Array,
    ) -> tuple[Array, Array]:
        """Evaluate residual guidance against one exact curation-stage bank.

        Callers invoke this only from an applied curation branch.  The residual
        remains the current sample's pre-update prediction error, while feature
        values and direct output-weight credit reflect the structural bank,
        parameters, and output weights visible at that write stage.
        """

        stage_values = _compute_feature_values(
            ops,
            parent_a,
            parent_b,
            theta,
            observation,
        )
        stage_credit = (errors @ output_weights) / active_count
        return stage_values, stage_credit

    def _cascade_replace_with_mask(
        self,
        ops: Array,
        parent_a: Array,
        parent_b: Array,
        theta: Array,
        depth: Array,
        utilities: Array,
        ages: Array,
        output_weights: Array,
        replaced_mask: Array,
        observation: Array,
        key: Array,
        feature_values: Array | None = None,
        feature_credit: Array | None = None,
        forced_op: Array | None = None,
        parent_mode: Array | None = None,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]:
        """Apply cascade replacement: every descendant of a replaced slot is also replaced.

        Iterates over slots in topological order; on each pass, a slot is
        marked for replacement if it is currently in ``replaced_mask`` or if
        it is non-raw and references a parent that has been marked.  Each
        replaced slot is filled with a fresh raw-input passthrough or, when
        possible, a fresh random composition that respects the depth budget
        and the topological invariant.
        """
        n_features = self._n_features
        # The raw-input dim is static under JIT: it is the observation length.
        # Using it (rather than ``n_features``) keeps OP_RAW fallback indices
        # inside the raw-input range so they never alias a wrong raw feature.
        feature_dim = observation.shape[0]
        del observation  # only the static shape is needed by the refresh path

        # Descendant cascade: scan through slots and propagate the mask.
        def cascade_step(
            carry_mask: Array, i: Array
        ) -> tuple[Array, None]:
            a = parent_a[i]
            b = parent_b[i]
            is_composed = ops[i] != OP_RAW
            # Safe gathers; for raw ops the parents reference different
            # spaces, but `is_composed` masks out their effect.
            safe_a = jnp.clip(a, 0, n_features - 1)
            safe_b = jnp.clip(b, 0, n_features - 1)
            parent_replaced = jnp.where(
                is_composed,
                carry_mask[safe_a]
                | jnp.where(b >= 0, carry_mask[safe_b], jnp.bool_(False)),
                jnp.bool_(False),
            )
            new_mark = carry_mask[i] | parent_replaced
            new_carry = carry_mask.at[i].set(new_mark)
            return new_carry, None

        cascaded_mask, _ = jax.lax.scan(
            cascade_step, replaced_mask, jnp.arange(n_features)
        )

        # Generate fresh slot contents, respecting the strict-less-than
        # parent invariant.  Each replaced slot becomes a passthrough of a
        # randomly chosen still-alive earlier slot, or a raw input if none
        # earlier slots survive.
        def refill_step(
            carry: tuple[Array, Array, Array, Array, Array, Array, Array, Array],
            i: Array,
        ) -> tuple[
            tuple[Array, Array, Array, Array, Array, Array, Array, Array], None
        ]:
            (
                ops_c,
                pa_c,
                pb_c,
                theta_c,
                depth_c,
                utils_c,
                ages_c,
                ow_c,
            ) = carry
            do_replace = cascaded_mask[i]
            # Give each slot a stable random substream so an unrelated earlier
            # slot entering or leaving the cascade cannot perturb this refill.
            slot_key = jr.fold_in(key, i.astype(jnp.uint32))
            op_key, pa_key, pb_key, theta_key = jr.split(slot_key, 4)

            # Determine the eligible parent set: indices < i whose slot is
            # NOT being replaced.  Bias parent selection by utility so
            # productive surviving features are more likely to become
            # parents of replacements.
            slot_indices = jnp.arange(n_features)
            in_range = slot_indices < i
            alive = in_range & (~cascaded_mask)
            depth_ok = depth_c + 1 <= self._max_depth
            eligible = alive & depth_ok
            any_eligible = jnp.any(eligible)
            logits = self._parent_logits(
                eligible,
                utils_c,
                feature_values=feature_values,
                feature_credit=feature_credit,
                depth=depth_c,
                ages=ages_c,
                parent_mode=parent_mode,
            )
            partner_logits = self._partner_logits(
                eligible,
                depth_c,
                utils_c,
                ages=ages_c,
                parent_mode=parent_mode,
            )
            a_idx = jnp.where(
                any_eligible,
                jr.categorical(pa_key, logits).astype(jnp.int32),
                jnp.array(0, dtype=jnp.int32),
            )
            b_idx = jnp.where(
                any_eligible,
                jr.categorical(pb_key, partner_logits).astype(jnp.int32),
                jnp.array(0, dtype=jnp.int32),
            )
            new_op = jnp.where(
                any_eligible,
                jr.categorical(op_key, self._op_logits(forced_op)).astype(jnp.int32),
                jnp.array(OP_RAW, dtype=jnp.int32),
            )
            # For OP_RAW fallback, parent_a is a raw-input index (clamp to
            # feature_dim-1) and parent_b is -1.  This keeps the slot valid.
            raw_a_idx = jnp.clip(jnp.minimum(i, feature_dim - 1), 0, feature_dim - 1)
            new_pa = jnp.where(any_eligible, a_idx, raw_a_idx)
            new_pb = jnp.where(
                any_eligible, b_idx, jnp.array(-1, dtype=jnp.int32)
            )
            new_theta = 0.5 * jr.normal(theta_key, (2,), dtype=jnp.float32)
            new_depth = jnp.where(
                any_eligible,
                jnp.maximum(depth_c[a_idx], depth_c[b_idx]) + 1,
                jnp.array(0, dtype=jnp.int32),
            ).astype(jnp.int32)

            ops_n = jnp.where(do_replace, ops_c.at[i].set(new_op), ops_c)
            pa_n = jnp.where(do_replace, pa_c.at[i].set(new_pa), pa_c)
            pb_n = jnp.where(do_replace, pb_c.at[i].set(new_pb), pb_c)
            theta_n = jnp.where(
                do_replace, theta_c.at[i].set(new_theta), theta_c
            )
            depth_n = jnp.where(
                do_replace, depth_c.at[i].set(new_depth), depth_c
            )
            utils_n = jnp.where(do_replace, utils_c.at[i].set(0.0), utils_c)
            ages_n = jnp.where(do_replace, ages_c.at[i].set(0), ages_c)
            ow_n = jnp.where(
                do_replace, ow_c.at[:, i].set(0.0), ow_c
            )
            return (ops_n, pa_n, pb_n, theta_n, depth_n, utils_n, ages_n, ow_n), None

        (ops_f, pa_f, pb_f, theta_f, depth_f, utils_f, ages_f, ow_f), _ = (
            jax.lax.scan(
                refill_step,
                (ops, parent_a, parent_b, theta, depth, utilities, ages, output_weights),
                jnp.arange(n_features),
            )
        )
        return (
            ops_f,
            pa_f,
            pb_f,
            theta_f,
            depth_f,
            utils_f,
            ages_f,
            ow_f,
            cascaded_mask,
        )

    def _cascade_replace(
        self,
        ops: Array,
        parent_a: Array,
        parent_b: Array,
        theta: Array,
        depth: Array,
        utilities: Array,
        ages: Array,
        output_weights: Array,
        replaced_mask: Array,
        observation: Array,
        key: Array,
        feature_values: Array | None = None,
        feature_credit: Array | None = None,
        forced_op: Array | None = None,
        parent_mode: Array | None = None,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
        """Retain the private eight-value surface for compatibility callers."""
        result = self._cascade_replace_with_mask(
            ops,
            parent_a,
            parent_b,
            theta,
            depth,
            utilities,
            ages,
            output_weights,
            replaced_mask,
            observation,
            key,
            feature_values=feature_values,
            feature_credit=feature_credit,
            forced_op=forced_op,
            parent_mode=parent_mode,
        )
        return result[:8]

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: CompositionalFeatureState,
        observation: Array,
        targets: Array,
        context_id: Array | int = 0,
        curation_allowed: Array | bool = True,
    ) -> CompositionalFeatureUpdateResult:
        """Perform one temporally-uniform compositional-feature update.

        ``curation_allowed=False`` consumes any due curation opportunity while
        retaining ordinary learning, RNG, clocks, and cadence advancement.  A
        consumed opportunity is never queued for immediate retry.
        """
        context = jnp.asarray(context_id, dtype=jnp.int32)
        allow_curation = _dynamic_bool_scalar(
            curation_allowed,
            name="curation_allowed",
        )
        lifetime_counter_valid = _compositional_counter_state_valid(
            state,
            self._replacement_interval,
        )
        proposed_step_words, lifetime_capacity_available = (
            _checked_step_words_increment(state.step_words)
        )
        update_available = lifetime_counter_valid & lifetime_capacity_available
        if (
            self._candidate_novelty_admission_bonus > 0.0
            or self._ancestor_utility_backup_decay > 0.0
        ):
            topology_valid = self._ranking_topology_valid(
                state,
                observation.shape[0],
            )
            state_finite = self._ranking_state_finite(state)
            observation_finite = jnp.all(jnp.isfinite(observation))
            targets_valid = jnp.all(jnp.isnan(targets) | jnp.isfinite(targets))
            context_valid = jnp.all(
                (context >= 0) & (context < self._generator_resource_contexts)
            )
            update_available = (
                update_available
                & topology_valid
                & state_finite
                & observation_finite
                & targets_valid
                & context_valid
            )
        active_mask = ~jnp.isnan(targets)
        safe_targets = jnp.where(active_mask, targets, 0.0)
        active_count = jnp.maximum(jnp.sum(active_mask.astype(jnp.float32)), 1.0)
        task_activity_ema = (
            self._future_utility_task_activity_decay * state.task_activity_ema
            + (1.0 - self._future_utility_task_activity_decay)
            * active_mask.astype(jnp.float32)
        )

        feature_values = _compute_feature_values(
            state.ops,
            state.parent_a,
            state.parent_b,
            state.theta,
            observation,
        )
        predictions = state.output_weights @ feature_values + state.output_bias
        errors = jnp.where(active_mask, safe_targets - predictions, 0.0)
        reported_errors = jnp.where(active_mask, errors, jnp.nan)

        # Output-weight update.
        output_delta = (
            self._step_size_output
            * errors[:, None]
            * feature_values[None, :]
            / active_count
        )
        output_bias_delta = self._step_size_output * errors / active_count

        # Per-feature credit and theta update via local linearization.
        feature_credit = (errors @ state.output_weights) / active_count
        d_theta0, d_theta1 = _theta_local_grads(
            state.ops,
            state.parent_a,
            state.parent_b,
            state.theta,
            feature_values,
        )
        theta_delta = self._step_size_theta * jnp.stack(
            [feature_credit * d_theta0, feature_credit * d_theta1], axis=-1
        )

        # Backward-looking utility: an equal blend of contribution magnitude
        # (mean |w| * |f|, how much slot i currently moves the predictions)
        # and gradient credit (|errors @ W|, how much the residual would move
        # it).  Matches ``FixedBudgetFeatureLearner``'s default (mean-aggregated)
        # signal in feature_discovery.
        current_utility_signal = (
            0.5 * jnp.mean(jnp.abs(state.output_weights), axis=0) * jnp.abs(feature_values)
            + 0.5 * jnp.abs(feature_credit)
        )
        (
            future_utility_signal,
            utility_contribution_trace,
            utility_error_trace,
            utility_feature_trace,
            utility_feature_energy_trace,
        ) = self._future_utility_signal(
            errors,
            feature_values,
            active_mask,
            active_count,
            task_activity_ema,
            state.utility_contribution_trace,
            state.utility_error_trace,
            state.utility_feature_trace,
            state.utility_feature_energy_trace,
        )
        if (
            self._future_utility_mix > 0.0
            and self._future_utility_normalization != "none"
        ):
            future_utility_signal, utility_signal_second_moment = (
                normalize_future_utility_signal(
                    future_utility_signal,
                    state.ages,
                    state.utility_signal_second_moment,
                    self._future_utility_normalization_decay,
                    self._utility_decay,
                    self._future_utility_normalization,
                )
            )
        else:
            utility_signal_second_moment = state.utility_signal_second_moment
        utility_signal = self._mixed_utility_signal(
            current_utility_signal,
            future_utility_signal,
        )
        feature_score_residual_trace = state.feature_score_residual_trace
        feature_score_energy_trace = state.feature_score_energy_trace
        if self._candidate_scoring_mode == "energy_novelty":
            (
                utility_signal,
                feature_score_residual_trace,
                feature_score_energy_trace,
            ) = self._energy_normalized_residual_score(
                errors,
                feature_values,
                state.feature_score_residual_trace,
                state.feature_score_energy_trace,
            )
        new_utilities = (
            self._utility_decay * state.utilities
            + (1.0 - self._utility_decay) * utility_signal
        )
        retention_slow_utilities = self._retention_slow_utility(
            state.retention_slow_utilities,
            utility_signal,
        )

        # Candidate forward + utility (candidates contribute to training of
        # their own output weights/parameters but not to predictions).
        candidate_output_delta = jnp.zeros_like(state.candidate_output_weights)
        candidate_theta_delta = jnp.zeros_like(state.candidate_theta)
        new_candidate_utilities = state.candidate_utilities
        candidate_utility_contribution_trace = (
            state.candidate_utility_contribution_trace
        )
        candidate_utility_feature_trace = state.candidate_utility_feature_trace
        candidate_utility_feature_energy_trace = (
            state.candidate_utility_feature_energy_trace
        )
        candidate_utility_signal_second_moment = (
            state.candidate_utility_signal_second_moment
        )
        candidate_score_residual_trace = state.candidate_score_residual_trace
        candidate_score_energy_trace = state.candidate_score_energy_trace
        candidate_active_correlation_trace = state.candidate_active_correlation_trace
        candidate_admission_novelty = jnp.zeros(
            (self._candidate_count,), dtype=jnp.float32
        )
        candidate_feature_values = jnp.zeros(
            (self._candidate_count,), dtype=jnp.float32
        )
        if self._candidate_count > 0:
            candidate_feature_values = self._candidate_features(
                state, feature_values, observation
            )
            candidate_output_delta = (
                self._step_size_output
                * errors[:, None]
                * candidate_feature_values[None, :]
                / active_count
            )
            candidate_credit = (
                errors @ state.candidate_output_weights
            ) / active_count
            candidate_d_theta0, candidate_d_theta1 = _candidate_theta_local_grads(
                state.candidate_ops,
                state.candidate_parent_a,
                state.candidate_parent_b,
                state.candidate_theta,
                candidate_feature_values,
                feature_values,
            )
            candidate_theta_delta = self._step_size_theta * jnp.stack(
                [
                    candidate_credit * candidate_d_theta0,
                    candidate_credit * candidate_d_theta1,
                ],
                axis=-1,
            )
            if not self._train_candidate_theta:
                candidate_theta_delta = jnp.zeros_like(candidate_theta_delta)
            candidate_signal = (
                0.5
                * jnp.mean(jnp.abs(state.candidate_output_weights), axis=0)
                * jnp.abs(candidate_feature_values)
                + 0.5 * jnp.abs(candidate_credit)
            )
            (
                candidate_future_signal,
                candidate_utility_contribution_trace,
                candidate_utility_feature_trace,
                candidate_utility_feature_energy_trace,
            ) = self._candidate_future_utility_signal(
                errors,
                candidate_feature_values,
                active_mask,
                active_count,
                task_activity_ema,
                state.candidate_utility_contribution_trace,
                state.utility_error_trace,
                state.candidate_utility_feature_trace,
                state.candidate_utility_feature_energy_trace,
            )
            if (
                self._future_utility_mix > 0.0
                and self._future_utility_normalization != "none"
            ):
                candidate_future_signal, candidate_utility_signal_second_moment = (
                    normalize_future_utility_signal(
                        candidate_future_signal,
                        state.candidate_ages,
                        state.candidate_utility_signal_second_moment,
                        self._future_utility_normalization_decay,
                        self._utility_decay,
                        self._future_utility_normalization,
                    )
                )
            candidate_signal = self._mixed_utility_signal(
                candidate_signal,
                candidate_future_signal,
            )
            if self._candidate_scoring_mode == "energy_novelty":
                (
                    candidate_signal,
                    candidate_score_residual_trace,
                    candidate_score_energy_trace,
                ) = self._energy_normalized_residual_score(
                    errors,
                    candidate_feature_values,
                    state.candidate_score_residual_trace,
                    state.candidate_score_energy_trace,
                )
                (
                    novelty_gate,
                    candidate_admission_novelty,
                    candidate_active_correlation_trace,
                ) = (
                    self._candidate_novelty_gate(
                        candidate_feature_values,
                        feature_values,
                        candidate_score_energy_trace,
                        feature_score_energy_trace,
                        state.candidate_active_correlation_trace,
                    )
                )
                candidate_signal = candidate_signal * novelty_gate
            new_candidate_utilities = (
                self._utility_decay * state.candidate_utilities
                + (1.0 - self._utility_decay) * candidate_signal
            )
        candidate_retention_slow_utilities = self._retention_slow_utility(
            state.candidate_retention_slow_utilities,
            candidate_signal if self._candidate_count > 0 else new_candidate_utilities,
        )

        # ObGD-style global update bounding (Elsayed et al. 2024, "Streaming
        # Deep Reinforcement Learning Finally Works").  One shared scale keeps
        # the total L1 update below ``1 / (kappa * max(||errors||, 1))``, so a
        # rare large-error sample cannot overshoot and wreck the whole bank;
        # higher kappa is more conservative.
        bounding_scale = jnp.array(1.0, dtype=jnp.float32)
        if self._use_obgd:
            total_step = (
                jnp.sum(jnp.abs(output_delta))
                + jnp.sum(jnp.abs(output_bias_delta))
                + jnp.sum(jnp.abs(theta_delta))
                + jnp.sum(jnp.abs(candidate_output_delta))
                + jnp.sum(jnp.abs(candidate_theta_delta))
            )
            err_norm = jnp.linalg.norm(errors)
            bound_magnitude = self._obgd_kappa * jnp.maximum(err_norm, 1.0) * total_step
            bounding_scale = 1.0 / jnp.maximum(bound_magnitude, 1.0)
            output_delta = bounding_scale * output_delta
            output_bias_delta = bounding_scale * output_bias_delta
            theta_delta = bounding_scale * theta_delta
            candidate_output_delta = bounding_scale * candidate_output_delta
            candidate_theta_delta = bounding_scale * candidate_theta_delta

        output_weights = state.output_weights + output_delta
        output_bias = state.output_bias + output_bias_delta
        theta = state.theta + theta_delta
        candidate_theta = state.candidate_theta + candidate_theta_delta
        candidate_output_weights = (
            state.candidate_output_weights + candidate_output_delta
        )
        ages = _saturating_nonnegative_int32_increment(state.ages)
        candidate_ages = _saturating_nonnegative_int32_increment(
            state.candidate_ages
        )
        step_count = _saturating_nonnegative_int32_increment(state.step_count)
        step_words = jnp.where(
            update_available,
            proposed_step_words,
            state.step_words,
        ).astype(jnp.uint32)
        replacement_phase = state.replacement_phase
        key, decision_key, curation_key = jr.split(state.key, 3)
        proposal_key, cascade_key = compositional_curation_keys(curation_key)
        candidate_overdepth_regeneration_key = jr.fold_in(
            curation_key,
            jnp.uint32(COMPOSITIONAL_CURATION_OVERDEPTH_REGENERATION_CHANNEL),
        )
        coverage_cursor: Array | None = None
        if self._generation_strategy == GENERATION_DOVETAIL_PRODUCT_COVERAGE:
            coverage_cursor = _dovetail_product_coverage_cursor(
                state.step_words,
                replacement_interval=self._replacement_interval,
                n_features=self._n_features,
                feature_dim=observation.shape[0],
            )

        replaced_slot = jnp.array(-1, dtype=jnp.int32)
        promoted_candidate = jnp.array(-1, dtype=jnp.int32)

        forced_op: Array | None = None
        parent_mode: Array | None = None
        generator_policy = jnp.asarray(
            FIXED_GENERATOR_POLICY_PLACEHOLDER,
            dtype=jnp.int32,
        )
        imprint_scale = jnp.asarray(self._candidate_imprint_scale, dtype=jnp.float32)
        promotion_margin = jnp.asarray(self._promotion_margin, dtype=jnp.float32)
        candidate_min_age = jnp.asarray(self._candidate_min_age, dtype=jnp.float32)
        replacement_accumulator = state.replacement_accumulator
        if self._learn_generator_resources:
            decision = self._generator_resource_manager.select(
                state.generator_resource_state,
                decision_key,
                context,
            )
            generator_policy = decision.action
            forced_op = decision.op_id
            parent_mode = decision.parent_mode
            imprint_scale = decision.imprint_scale
            promotion_margin = promotion_margin * decision.promotion_margin_multiplier
            candidate_min_age = candidate_min_age * decision.candidate_min_age_multiplier
            replacement_rate = (
                jnp.array(0.0, dtype=jnp.float32)
                if self._replacement_interval == 0
                else decision.replacement_multiplier
                / float(self._replacement_interval)
            )
            replacement_accumulator = replacement_accumulator + replacement_rate
            # Consume a due credit even when the caller suppresses curation.
            # The configured per-step rate is at most one, so this cannot
            # accumulate an unbounded deferred-replacement backlog.
            curation_due = (
                update_available
                & (self._replacement_interval > 0)
                & (replacement_accumulator >= 1.0)
            )
            replacement_accumulator = jnp.where(
                curation_due,
                replacement_accumulator - 1.0,
                replacement_accumulator,
            )
            should_try_replace = curation_due & allow_curation
        else:
            proposed_replacement_phase, fixed_curation_due = (
                _advance_replacement_phase(
                    state.replacement_phase,
                    self._replacement_interval,
                )
            )
            replacement_phase = jnp.where(
                update_available,
                proposed_replacement_phase,
                state.replacement_phase,
            ).astype(jnp.int32)
            should_try_replace = (
                update_available & fixed_curation_due & allow_curation
            )

        # Identify the worst eligible active slot.  Raw-input slots
        # (depth == 0) are protected from replacement.
        is_raw = state.ops == OP_RAW
        recursive_product_scaffold = (
            (state.ops == OP_PRODUCT)
            & (state.depth == 1)
            & (self._generation_strategy == GENERATION_RECURSIVE_PRODUCT)
        )
        tanh_quota_protected = (
            (state.ops == OP_TANH)
            & (
                jnp.sum((state.ops == OP_TANH).astype(jnp.int32))
                <= self._retention_tanh_min_count
            )
        )
        product_quota_protected = (
            (state.ops == OP_PRODUCT)
            & (
                jnp.sum((state.ops == OP_PRODUCT).astype(jnp.int32))
                <= self._retention_product_min_count
            )
        )
        eligible_active = (
            (ages >= self._min_feature_age)
            & (~is_raw)
            & (~recursive_product_scaffold)
            & (~tanh_quota_protected)
            & (~product_quota_protected)
        )
        active_direct_replacement_score = self._retention_score(
            new_utilities,
            retention_slow_utilities,
        )
        active_replacement_score = self._ancestor_backed_retention_score(
            active_direct_replacement_score,
            state.ops,
            state.parent_a,
            state.parent_b,
        )
        retention_bonus = (
            jnp.asarray(self._retention_depth_bonus, dtype=jnp.float32)
            * state.depth.astype(jnp.float32)
            / jnp.maximum(float(self._max_depth), 1.0)
        )
        active_scores = jnp.where(
            eligible_active,
            active_replacement_score + retention_bonus,
            jnp.inf,
        )
        worst_active = jnp.argmin(active_scores).astype(jnp.int32)
        has_active_slot = jnp.any(eligible_active)

        ops = state.ops
        parent_a = state.parent_a
        parent_b = state.parent_b
        depth = state.depth
        candidate_ops = state.candidate_ops
        candidate_parent_a = state.candidate_parent_a
        candidate_parent_b = state.candidate_parent_b
        candidate_depth = state.candidate_depth
        feature_generator_policy = state.feature_generator_policy
        candidate_generator_policy = state.candidate_generator_policy
        should_promote_for_trace = jnp.array(False)
        best_candidate_for_trace = jnp.array(0, dtype=jnp.int32)
        promoted_slot_for_trace = worst_active
        no_event = jnp.asarray(False, dtype=jnp.bool_)
        absent_index = jnp.asarray(-1, dtype=jnp.int32)
        absent_theta = jnp.zeros((2,), dtype=jnp.float32)
        proposal_formed = no_event
        proposal_destination_bank = jnp.asarray(
            CURATION_DESTINATION_NONE, dtype=jnp.int32
        )
        proposal_destination_slot = absent_index
        proposal_op = absent_index
        proposal_parent_a = absent_index
        proposal_parent_b = absent_index
        proposal_theta = absent_theta
        proposal_depth = absent_index
        proposal_generator_policy = absent_index
        root_change_applied = no_event
        post_root_pre_cascade_slot = absent_index
        post_root_pre_cascade_op = absent_index
        post_root_pre_cascade_parent_a = absent_index
        post_root_pre_cascade_parent_b = absent_index
        post_root_pre_cascade_theta = absent_theta
        post_root_pre_cascade_depth = absent_index
        post_root_pre_cascade_generator_policy = absent_index
        promotion_source_candidate = absent_index
        promotion_destination_active = absent_index
        promoted_pre_refresh_op = absent_index
        promoted_pre_refresh_parent_a = absent_index
        promoted_pre_refresh_parent_b = absent_index
        promoted_pre_refresh_theta = absent_theta
        promoted_pre_refresh_depth = absent_index
        promoted_pre_refresh_generator_policy = absent_index
        root_change_mask = jnp.zeros((self._n_features,), dtype=jnp.bool_)
        cascade_change_mask = jnp.zeros((self._n_features,), dtype=jnp.bool_)
        active_change_mask = jnp.zeros((self._n_features,), dtype=jnp.bool_)
        ordinary_candidate_refresh_mask = jnp.zeros(
            (self._candidate_count,), dtype=jnp.bool_
        )
        post_promotion_candidate_refresh_mask = jnp.zeros(
            (self._candidate_count,), dtype=jnp.bool_
        )
        candidate_rebound_mask = jnp.zeros((self._candidate_count,), dtype=jnp.bool_)
        candidate_overdepth_regeneration_mask = jnp.zeros(
            (self._candidate_count,), dtype=jnp.bool_
        )
        candidate_selector_log_weights = state.candidate_selector_log_weights
        candidate_selector_cumulative_loss = state.candidate_selector_cumulative_loss
        candidate_selector_action_counts = state.candidate_selector_action_counts

        # Candidate decision defaults preserve fixed shapes when C == 0.  For
        # C > 0 the branch below overwrites every value with the exact arrays
        # used by admission and ordinary refresh selection.
        candidate_direct_promotion_scores = jnp.zeros(
            (self._candidate_count,), dtype=jnp.float32
        )
        candidate_promotion_scores = jnp.zeros(
            (self._candidate_count,), dtype=jnp.float32
        )
        eligible_candidates = jnp.zeros(
            (self._candidate_count,), dtype=jnp.bool_
        )
        candidate_depth_after_all = jnp.zeros(
            (self._candidate_count,), dtype=jnp.int32
        )
        candidate_topology_compatible = jnp.zeros(
            (self._candidate_count, self._n_features), dtype=jnp.bool_
        )
        candidate_depth_compatible = jnp.zeros(
            (self._candidate_count, self._n_features), dtype=jnp.bool_
        )
        candidate_headroom_compatible = jnp.ones(
            (self._candidate_count, self._n_features), dtype=jnp.bool_
        )
        candidate_margin_eligible = jnp.zeros(
            (self._candidate_count, self._n_features), dtype=jnp.bool_
        )
        compatible_active_by_candidate = jnp.zeros(
            (self._candidate_count, self._n_features), dtype=jnp.bool_
        )
        candidate_has_destination = jnp.zeros(
            (self._candidate_count,), dtype=jnp.bool_
        )
        candidate_ranking_scores = jnp.zeros(
            (self._candidate_count,), dtype=jnp.float32
        )
        candidate_refresh_utilities = jnp.zeros(
            (self._candidate_count,), dtype=jnp.float32
        )
        has_candidate = jnp.asarray(False, dtype=jnp.bool_)
        has_refresh_candidate = jnp.asarray(False, dtype=jnp.bool_)
        left_pack_destination_available = jnp.asarray(False, dtype=jnp.bool_)
        selected_candidate_for_audit = absent_index
        selected_destination_for_audit = absent_index
        selected_refresh_candidate_for_audit = absent_index
        selected_candidate_score_for_audit = jnp.asarray(0.0, dtype=jnp.float32)
        selected_destination_score_for_audit = jnp.asarray(
            0.0, dtype=jnp.float32
        )
        margin_rhs_for_audit = jnp.asarray(0.0, dtype=jnp.float32)
        margin_passed_for_audit = jnp.asarray(False, dtype=jnp.bool_)
        selected_topology_ok_for_audit = jnp.asarray(False, dtype=jnp.bool_)
        selected_depth_ok_for_audit = jnp.asarray(False, dtype=jnp.bool_)
        selected_headroom_ok_for_audit = jnp.asarray(False, dtype=jnp.bool_)
        selected_can_promote_for_audit = jnp.asarray(False, dtype=jnp.bool_)
        should_promote_for_audit = jnp.asarray(False, dtype=jnp.bool_)
        should_refresh_for_audit = jnp.asarray(False, dtype=jnp.bool_)

        # Hold immutable references to the post-learning/pre-curation arrays;
        # later structural branches may rebind the local state variables.
        decision_active_theta = theta
        decision_active_ages = ages
        decision_active_fast_utilities = new_utilities
        decision_active_slow_utilities = retention_slow_utilities
        decision_candidate_theta = candidate_theta
        decision_candidate_ages = candidate_ages
        decision_candidate_fast_utilities = new_candidate_utilities
        decision_candidate_slow_utilities = candidate_retention_slow_utilities

        if self._candidate_count > 0:
            eligible_candidates = candidate_ages.astype(jnp.float32) >= candidate_min_age
            slot_indices = jnp.arange(self._n_features)
            safe_candidate_pa = jnp.clip(candidate_parent_a, 0, self._n_features - 1)
            safe_candidate_pb = jnp.clip(candidate_parent_b, 0, self._n_features - 1)
            candidate_depth_after_all = (
                jnp.maximum(
                    depth[safe_candidate_pa],
                    jnp.where(
                        candidate_parent_b >= 0,
                        depth[safe_candidate_pb],
                        0,
                    ),
                )
                + 1
            )
            candidate_parent_max = jnp.maximum(
                candidate_parent_a,
                jnp.where(candidate_parent_b >= 0, candidate_parent_b, -1),
            )
            candidate_topology_compatible = (
                slot_indices[None, :] > candidate_parent_max[:, None]
            )
            candidate_depth_compatible = jnp.broadcast_to(
                (candidate_depth_after_all <= self._max_depth)[:, None],
                (self._candidate_count, self._n_features),
            )
            if self._topology_headroom_reserve:
                remaining_depth = (
                    jnp.asarray(self._max_depth, dtype=jnp.int32)
                    - candidate_depth_after_all
                )
                candidate_headroom_compatible = (
                    slot_indices[None, :] + remaining_depth[:, None]
                    < self._n_features
                )
            compatible_active_by_candidate = (
                eligible_candidates[:, None]
                & eligible_active[None, :]
                & candidate_topology_compatible
                & candidate_depth_compatible
                & candidate_headroom_compatible
            )
            candidate_has_destination = jnp.any(
                compatible_active_by_candidate, axis=1
            )
            candidate_direct_promotion_scores = self._retention_score(
                new_candidate_utilities,
                candidate_retention_slow_utilities,
            )
            candidate_promotion_scores = candidate_direct_promotion_scores
            if self._candidate_novelty_admission_bonus > 0.0:
                candidate_promotion_scores = (
                    candidate_promotion_scores
                    + jnp.asarray(
                        self._candidate_novelty_admission_bonus,
                        dtype=jnp.float32,
                    )
                    * candidate_admission_novelty
                )
            candidate_margin_eligible = (
                compatible_active_by_candidate
                & (
                    candidate_promotion_scores[:, None]
                    > promotion_margin * active_replacement_score[None, :]
                )
            )
            candidate_selector_state = FiniteCandidateSelectorState(
                log_weights=candidate_selector_log_weights,
                cumulative_loss=candidate_selector_cumulative_loss,
                action_counts=candidate_selector_action_counts,
                step_count=state.step_count,
            )
            if self._candidate_selector is not None:
                selector_probabilities = self._candidate_selector.probabilities(
                    candidate_selector_state
                )
                candidate_ranking_scores = selector_probabilities
                candidate_scores = jnp.where(
                    candidate_has_destination, selector_probabilities, -jnp.inf
                )
            else:
                candidate_ranking_scores = candidate_promotion_scores
                candidate_scores = jnp.where(
                    candidate_has_destination, candidate_promotion_scores, -jnp.inf
                )
            best_candidate = jnp.argmax(candidate_scores).astype(jnp.int32)
            has_candidate = jnp.any(candidate_has_destination)
            if self._candidate_selector is not None:
                selector_losses = _candidate_scores_to_unit_losses(
                    candidate_promotion_scores,
                    candidate_has_destination,
                )
                selector_result = self._candidate_selector.update(
                    candidate_selector_state,
                    selector_losses,
                    selected_action=best_candidate,
                )
                candidate_selector_log_weights = selector_result.state.log_weights
                candidate_selector_cumulative_loss = (
                    selector_result.state.cumulative_loss
                )
                candidate_selector_action_counts = selector_result.state.action_counts
            has_refresh_candidate = jnp.any(eligible_candidates)
            candidate_refresh_utilities = (
                candidate_promotion_scores
                if self._candidate_novelty_admission_bonus > 0.0
                else new_candidate_utilities
            )
            refresh_scores = jnp.where(
                eligible_candidates, candidate_refresh_utilities, jnp.inf
            )
            worst_candidate = jnp.argmin(refresh_scores).astype(jnp.int32)
            compatible_active = compatible_active_by_candidate[best_candidate]
            if self._topology_left_pack_destinations:
                selected_margin_eligible = candidate_margin_eligible[best_candidate]
                left_pack_destination_available = jnp.any(
                    selected_margin_eligible
                )
                promotion_slot = jnp.argmin(
                    jnp.where(
                        selected_margin_eligible,
                        slot_indices,
                        self._n_features,
                    )
                ).astype(jnp.int32)
                should_promote = (
                    should_try_replace
                    & has_active_slot
                    & has_candidate
                    & left_pack_destination_available
                )
            else:
                promotion_slot_scores = jnp.where(
                    compatible_active, active_replacement_score, jnp.inf
                )
                promotion_slot = jnp.argmin(promotion_slot_scores).astype(jnp.int32)
                should_promote = (
                    should_try_replace
                    & has_active_slot
                    & has_candidate
                    & (
                        candidate_promotion_scores[best_candidate]
                        > promotion_margin
                        * active_replacement_score[promotion_slot]
                    )
                )
            ordinary_candidate_refresh_applied = (
                (~should_promote) & should_try_replace & has_refresh_candidate
            )
            selected_candidate_for_audit = jnp.where(
                has_candidate, best_candidate, absent_index
            ).astype(jnp.int32)
            has_selected_destination = has_candidate & has_active_slot
            if self._topology_left_pack_destinations:
                has_selected_destination = (
                    has_selected_destination & left_pack_destination_available
                )
            selected_destination_for_audit = jnp.where(
                has_selected_destination, promotion_slot, absent_index
            ).astype(jnp.int32)
            selected_refresh_candidate_for_audit = jnp.where(
                has_refresh_candidate, worst_candidate, absent_index
            ).astype(jnp.int32)
            selected_candidate_score_for_audit = jnp.where(
                has_candidate,
                candidate_promotion_scores[best_candidate],
                0.0,
            ).astype(jnp.float32)
            selected_destination_score_for_audit = jnp.where(
                has_selected_destination,
                active_replacement_score[promotion_slot],
                0.0,
            ).astype(jnp.float32)
            margin_rhs_for_audit = (
                promotion_margin * selected_destination_score_for_audit
            ).astype(jnp.float32)
            margin_passed_for_audit = has_selected_destination & (
                selected_candidate_score_for_audit > margin_rhs_for_audit
            )
            safe_selected_candidate_pa = candidate_parent_a[best_candidate]
            safe_selected_candidate_pb = candidate_parent_b[best_candidate]
            selected_topology_ok_for_audit = (
                has_selected_destination
                & (safe_selected_candidate_pa < promotion_slot)
                & (
                    (safe_selected_candidate_pb < 0)
                    | (safe_selected_candidate_pb < promotion_slot)
                )
            )
            selected_depth_ok_for_audit = (
                has_selected_destination
                & (candidate_depth_after_all[best_candidate] <= self._max_depth)
            )
            selected_headroom_ok_for_audit = (
                has_selected_destination
                & candidate_headroom_compatible[
                    best_candidate, promotion_slot
                ]
            )
            selected_can_promote_for_audit = (
                selected_topology_ok_for_audit
                & selected_depth_ok_for_audit
                & selected_headroom_ok_for_audit
            )
            should_promote_for_audit = should_promote
            should_refresh_for_audit = ordinary_candidate_refresh_applied
            pre_curation_candidate_ops = candidate_ops
            pre_curation_candidate_parent_a = candidate_parent_a
            pre_curation_candidate_parent_b = candidate_parent_b
            pre_curation_candidate_theta = candidate_theta
            pre_curation_candidate_depth = candidate_depth
            pre_curation_candidate_generator_policy = candidate_generator_policy

            def promote_branch(args: tuple[Array, ...]) -> tuple[Array, ...]:
                (
                    ops_a,
                    pa_a,
                    pb_a,
                    theta_a,
                    depth_a,
                    util_a,
                    age_a,
                    ow_a,
                    co_a,
                    cpa_a,
                    cpb_a,
                    ctheta_a,
                    cdepth_a,
                    cow_a,
                    cutil_a,
                    cage_a,
                    fgp_a,
                    cgp_a,
                    _stage_values_a,
                    _stage_credit_a,
                ) = args
                # Build a candidate that is "promotable": its parents must be
                # strictly smaller than the destination index ``promotion_slot``
                # to preserve the topological invariant.  We only promote when
                # both candidate parents are < promotion_slot.  Otherwise we
                # fall back to a refresh.
                cand_pa = cpa_a[best_candidate]
                cand_pb = cpb_a[best_candidate]
                cand_op = co_a[best_candidate]
                # Also ensure the resulting depth remains within budget.
                cand_depth_after = jnp.maximum(
                    depth_a[jnp.clip(cand_pa, 0, self._n_features - 1)],
                    jnp.where(
                        cand_pb >= 0,
                        depth_a[jnp.clip(cand_pb, 0, self._n_features - 1)],
                        0,
                    ),
                ) + 1
                topo_ok = (cand_pa < promotion_slot) & (
                    (cand_pb < 0) | (cand_pb < promotion_slot)
                )
                depth_ok = cand_depth_after <= self._max_depth
                headroom_ok = (
                    (not self._topology_headroom_reserve)
                    | (
                        promotion_slot
                        + (self._max_depth - cand_depth_after)
                        < self._n_features
                    )
                )
                margin_ok = (
                    (not self._topology_left_pack_destinations)
                    | candidate_margin_eligible[
                        best_candidate, promotion_slot
                    ]
                )
                can_promote = topo_ok & depth_ok & headroom_ok & margin_ok

                ops_b = ops_a.at[promotion_slot].set(
                    jnp.where(can_promote, cand_op, ops_a[promotion_slot])
                )
                pa_b = pa_a.at[promotion_slot].set(
                    jnp.where(can_promote, cand_pa, pa_a[promotion_slot])
                )
                pb_b = pb_a.at[promotion_slot].set(
                    jnp.where(can_promote, cand_pb, pb_a[promotion_slot])
                )
                theta_b = theta_a.at[promotion_slot].set(
                    jnp.where(
                        can_promote, ctheta_a[best_candidate], theta_a[promotion_slot]
                    )
                )
                depth_b = depth_a.at[promotion_slot].set(
                    jnp.where(
                        can_promote, cand_depth_after, depth_a[promotion_slot]
                    ).astype(jnp.int32)
                )
                util_b = util_a.at[promotion_slot].set(
                    jnp.where(
                        can_promote, cutil_a[best_candidate], util_a[promotion_slot]
                    )
                )
                age_b = age_a.at[promotion_slot].set(
                    jnp.where(can_promote, 0, age_a[promotion_slot]).astype(jnp.int32)
                )
                ow_b = ow_a.at[:, promotion_slot].set(
                    jnp.where(
                        can_promote,
                        self._promoted_output_weights(
                            ow_a[:, promotion_slot],
                            cow_a[:, best_candidate],
                        ),
                        ow_a[:, promotion_slot],
                    )
                )
                fgp_b = fgp_a.at[promotion_slot].set(
                    jnp.where(
                        can_promote,
                        cgp_a[best_candidate],
                        fgp_a[promotion_slot],
                    )
                )

                # Refresh the promoted candidate slot with a fresh
                # composition (parents drawn over ALL active slots, biased
                # by utility).
                promoted_feature_values, promoted_feature_credit = (
                    self._curation_stage_guidance(
                        ops_b,
                        pa_b,
                        pb_b,
                        theta_b,
                        ow_b,
                        observation,
                        errors,
                        active_count,
                    )
                )
                if (
                    self._generation_strategy
                    == GENERATION_DOVETAIL_PRODUCT_COVERAGE
                ):
                    gen_op, gen_pa, gen_pb, gen_theta, gen_depth = self._generate_one(
                        proposal_key,
                        depth_b,
                        util_b,
                        existing_ages=age_b,
                        feature_values=promoted_feature_values,
                        feature_credit=promoted_feature_credit,
                        forced_op=forced_op,
                        parent_mode=parent_mode,
                        coverage_cursor=coverage_cursor,
                        feature_dim=observation.shape[0],
                        preferred_depth1_parent=promotion_slot,
                    )
                else:
                    gen_op, gen_pa, gen_pb, gen_theta, gen_depth = self._generate_one(
                        proposal_key,
                        depth_b,
                        util_b,
                        existing_ages=age_b,
                        feature_values=promoted_feature_values,
                        feature_credit=promoted_feature_credit,
                        forced_op=forced_op,
                        parent_mode=parent_mode,
                    )
                gen_weights = self._initial_candidate_output_weights(
                    gen_op,
                    gen_pa,
                    gen_pb,
                    gen_theta,
                    promoted_feature_values,
                    observation,
                    errors,
                    active_count,
                    imprint_scale=imprint_scale,
                )
                co_b = co_a.at[best_candidate].set(gen_op)
                cpa_b = cpa_a.at[best_candidate].set(gen_pa)
                cpb_b = cpb_a.at[best_candidate].set(gen_pb)
                ctheta_b = ctheta_a.at[best_candidate].set(gen_theta)
                cdepth_b = cdepth_a.at[best_candidate].set(gen_depth)
                cow_b = cow_a.at[:, best_candidate].set(gen_weights)
                cutil_b = cutil_a.at[best_candidate].set(0.0)
                cage_b = cage_a.at[best_candidate].set(0)
                cgp_b = cgp_a.at[best_candidate].set(generator_policy)

                return (
                    ops_b,
                    pa_b,
                    pb_b,
                    theta_b,
                    depth_b,
                    util_b,
                    age_b,
                    ow_b,
                    co_b,
                    cpa_b,
                    cpb_b,
                    ctheta_b,
                    cdepth_b,
                    cow_b,
                    cutil_b,
                    cage_b,
                    fgp_b,
                    cgp_b,
                    promoted_feature_values,
                    promoted_feature_credit,
                )

            def refresh_branch(args: tuple[Array, ...]) -> tuple[Array, ...]:
                (
                    ops_a,
                    pa_a,
                    pb_a,
                    theta_a,
                    depth_a,
                    util_a,
                    age_a,
                    ow_a,
                    co_a,
                    cpa_a,
                    cpb_a,
                    ctheta_a,
                    cdepth_a,
                    cow_a,
                    cutil_a,
                    cage_a,
                    fgp_a,
                    cgp_a,
                    stage_values_a,
                    stage_credit_a,
                ) = args
                do_refresh = ordinary_candidate_refresh_applied

                def apply_refresh(
                    candidate_args: tuple[Array, ...],
                ) -> tuple[Array, ...]:
                    (
                        co_x,
                        cpa_x,
                        cpb_x,
                        ctheta_x,
                        cdepth_x,
                        cow_x,
                        cutil_x,
                        cage_x,
                        cgp_x,
                        _stage_values_x,
                        _stage_credit_x,
                    ) = candidate_args
                    refresh_feature_values, refresh_feature_credit = (
                        self._curation_stage_guidance(
                            ops_a,
                            pa_a,
                            pb_a,
                            theta_a,
                            ow_a,
                            observation,
                            errors,
                            active_count,
                        )
                    )
                    if (
                        self._generation_strategy
                        == GENERATION_DOVETAIL_PRODUCT_COVERAGE
                    ):
                        gen_op, gen_pa, gen_pb, gen_theta, gen_depth = (
                            self._generate_one(
                                proposal_key,
                                depth_a,
                                util_a,
                                existing_ages=age_a,
                                feature_values=refresh_feature_values,
                                feature_credit=refresh_feature_credit,
                                forced_op=forced_op,
                                parent_mode=parent_mode,
                                coverage_cursor=coverage_cursor,
                                feature_dim=observation.shape[0],
                            )
                        )
                    else:
                        gen_op, gen_pa, gen_pb, gen_theta, gen_depth = (
                            self._generate_one(
                                proposal_key,
                                depth_a,
                                util_a,
                                existing_ages=age_a,
                                feature_values=refresh_feature_values,
                                feature_credit=refresh_feature_credit,
                                forced_op=forced_op,
                                parent_mode=parent_mode,
                            )
                        )
                    gen_weights = self._initial_candidate_output_weights(
                        gen_op,
                        gen_pa,
                        gen_pb,
                        gen_theta,
                        refresh_feature_values,
                        observation,
                        errors,
                        active_count,
                        imprint_scale=imprint_scale,
                    )
                    return (
                        co_x.at[worst_candidate].set(gen_op),
                        cpa_x.at[worst_candidate].set(gen_pa),
                        cpb_x.at[worst_candidate].set(gen_pb),
                        ctheta_x.at[worst_candidate].set(gen_theta),
                        cdepth_x.at[worst_candidate].set(gen_depth),
                        cow_x.at[:, worst_candidate].set(gen_weights),
                        cutil_x.at[worst_candidate].set(0.0),
                        cage_x.at[worst_candidate].set(0),
                        cgp_x.at[worst_candidate].set(generator_policy),
                        refresh_feature_values,
                        refresh_feature_credit,
                    )

                def keep_candidate(
                    candidate_args: tuple[Array, ...],
                ) -> tuple[Array, ...]:
                    return candidate_args

                (
                    co_b,
                    cpa_b,
                    cpb_b,
                    ctheta_b,
                    cdepth_b,
                    cow_b,
                    cutil_b,
                    cage_b,
                    cgp_b,
                    stage_values_b,
                    stage_credit_b,
                ) = jax.lax.cond(
                    do_refresh,
                    apply_refresh,
                    keep_candidate,
                    (
                        co_a,
                        cpa_a,
                        cpb_a,
                        ctheta_a,
                        cdepth_a,
                        cow_a,
                        cutil_a,
                        cage_a,
                        cgp_a,
                        stage_values_a,
                        stage_credit_a,
                    ),
                )
                return (
                    ops_a,
                    pa_a,
                    pb_a,
                    theta_a,
                    depth_a,
                    util_a,
                    age_a,
                    ow_a,
                    co_b,
                    cpa_b,
                    cpb_b,
                    ctheta_b,
                    cdepth_b,
                    cow_b,
                    cutil_b,
                    cage_b,
                    fgp_a,
                    cgp_b,
                    stage_values_b,
                    stage_credit_b,
                )

            candidate_curation_values = jnp.zeros_like(feature_values)
            candidate_curation_credit = jnp.zeros_like(feature_credit)
            carry = (
                ops,
                parent_a,
                parent_b,
                theta,
                depth,
                new_utilities,
                ages,
                output_weights,
                candidate_ops,
                candidate_parent_a,
                candidate_parent_b,
                candidate_theta,
                candidate_depth,
                candidate_output_weights,
                new_candidate_utilities,
                candidate_ages,
                feature_generator_policy,
                candidate_generator_policy,
                candidate_curation_values,
                candidate_curation_credit,
            )
            (
                ops,
                parent_a,
                parent_b,
                theta,
                depth,
                new_utilities,
                ages,
                output_weights,
                candidate_ops,
                candidate_parent_a,
                candidate_parent_b,
                candidate_theta,
                candidate_depth,
                candidate_output_weights,
                new_candidate_utilities,
                candidate_ages,
                feature_generator_policy,
                candidate_generator_policy,
                candidate_curation_values,
                candidate_curation_credit,
            ) = jax.lax.cond(
                should_promote, promote_branch, refresh_branch, carry
            )
            replaced_slot = jnp.where(should_promote, promotion_slot, replaced_slot)
            promoted_candidate = jnp.where(
                should_promote, best_candidate, promoted_candidate
            )
            should_promote_for_trace = should_promote
            best_candidate_for_trace = best_candidate
            promoted_slot_for_trace = promotion_slot
            ordinary_candidate_refresh_mask = ordinary_candidate_refresh_mask.at[
                worst_candidate
            ].set(ordinary_candidate_refresh_applied)
            post_promotion_candidate_refresh_mask = (
                post_promotion_candidate_refresh_mask.at[best_candidate].set(
                    should_promote
                )
            )
            proposal_formed = should_promote | ordinary_candidate_refresh_applied
            proposal_destination_bank = jnp.where(
                proposal_formed,
                jnp.asarray(CURATION_DESTINATION_CANDIDATE, dtype=jnp.int32),
                proposal_destination_bank,
            )
            proposed_candidate_slot = jnp.where(
                should_promote, best_candidate, worst_candidate
            )
            proposal_destination_slot = jnp.where(
                proposal_formed,
                proposed_candidate_slot,
                proposal_destination_slot,
            )
            proposal_op = jnp.where(
                proposal_formed,
                candidate_ops[proposed_candidate_slot],
                proposal_op,
            )
            proposal_parent_a = jnp.where(
                proposal_formed,
                candidate_parent_a[proposed_candidate_slot],
                proposal_parent_a,
            )
            proposal_parent_b = jnp.where(
                proposal_formed,
                candidate_parent_b[proposed_candidate_slot],
                proposal_parent_b,
            )
            proposal_theta = jnp.where(
                proposal_formed,
                candidate_theta[proposed_candidate_slot],
                proposal_theta,
            )
            proposal_depth = jnp.where(
                proposal_formed,
                candidate_depth[proposed_candidate_slot],
                proposal_depth,
            )
            proposal_generator_policy = jnp.where(
                proposal_formed,
                candidate_generator_policy[proposed_candidate_slot],
                proposal_generator_policy,
            )

            root_change_applied = should_promote
            root_change_mask = root_change_mask.at[promotion_slot].set(
                should_promote
            )
            post_root_pre_cascade_slot = jnp.where(
                should_promote,
                promotion_slot,
                post_root_pre_cascade_slot,
            )
            post_root_pre_cascade_op = jnp.where(
                should_promote,
                ops[promotion_slot],
                post_root_pre_cascade_op,
            )
            post_root_pre_cascade_parent_a = jnp.where(
                should_promote,
                parent_a[promotion_slot],
                post_root_pre_cascade_parent_a,
            )
            post_root_pre_cascade_parent_b = jnp.where(
                should_promote,
                parent_b[promotion_slot],
                post_root_pre_cascade_parent_b,
            )
            post_root_pre_cascade_theta = jnp.where(
                should_promote,
                theta[promotion_slot],
                post_root_pre_cascade_theta,
            )
            post_root_pre_cascade_depth = jnp.where(
                should_promote,
                depth[promotion_slot],
                post_root_pre_cascade_depth,
            )
            post_root_pre_cascade_generator_policy = jnp.where(
                should_promote,
                feature_generator_policy[promotion_slot],
                post_root_pre_cascade_generator_policy,
            )

            promotion_source_candidate = jnp.where(
                should_promote,
                best_candidate,
                promotion_source_candidate,
            )
            promotion_destination_active = jnp.where(
                should_promote,
                promotion_slot,
                promotion_destination_active,
            )
            promoted_pre_refresh_op = jnp.where(
                should_promote,
                pre_curation_candidate_ops[best_candidate],
                promoted_pre_refresh_op,
            )
            promoted_pre_refresh_parent_a = jnp.where(
                should_promote,
                pre_curation_candidate_parent_a[best_candidate],
                promoted_pre_refresh_parent_a,
            )
            promoted_pre_refresh_parent_b = jnp.where(
                should_promote,
                pre_curation_candidate_parent_b[best_candidate],
                promoted_pre_refresh_parent_b,
            )
            promoted_pre_refresh_theta = jnp.where(
                should_promote,
                pre_curation_candidate_theta[best_candidate],
                promoted_pre_refresh_theta,
            )
            promoted_pre_refresh_depth = jnp.where(
                should_promote,
                pre_curation_candidate_depth[best_candidate],
                promoted_pre_refresh_depth,
            )
            promoted_pre_refresh_generator_policy = jnp.where(
                should_promote,
                pre_curation_candidate_generator_policy[best_candidate],
                promoted_pre_refresh_generator_policy,
            )

            # If we promoted, cascade-replace any active descendants of
            # ``promotion_slot`` (slots that referenced it as a parent).
            # The promotion branch computed guidance after installing the root
            # and its current output weights.  Candidate refresh does not alter
            # the active bank, so the same stage is exact for cascade parents.
            def cascade_after_promote(
                args: tuple[
                    Array,
                    Array,
                    Array,
                    Array,
                    Array,
                    Array,
                    Array,
                    Array,
                    Array,
                    Array,
                ],
            ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]:
                (
                    ops_x,
                    pa_x,
                    pb_x,
                    theta_x,
                    depth_x,
                    util_x,
                    age_x,
                    ow_x,
                    stage_values_x,
                    stage_credit_x,
                ) = args
                slot_indices = jnp.arange(self._n_features)
                # Mark direct descendants as needing replacement; the cascade
                # routine will then propagate further.
                composed = ops_x != OP_RAW
                refs_a = (
                    composed
                    & (pa_x == promotion_slot)
                    & (slot_indices > promotion_slot)
                )
                refs_b = (
                    composed
                    & (pb_x >= 0)
                    & (pb_x == promotion_slot)
                    & (slot_indices > promotion_slot)
                )
                replaced_mask = refs_a | refs_b
                return self._cascade_replace_with_mask(
                    ops_x,
                    pa_x,
                    pb_x,
                    theta_x,
                    depth_x,
                    util_x,
                    age_x,
                    ow_x,
                    replaced_mask,
                    observation,
                    cascade_key,
                    feature_values=stage_values_x,
                    feature_credit=stage_credit_x,
                    forced_op=forced_op,
                    parent_mode=parent_mode,
                )

            def no_cascade(
                args: tuple[
                    Array,
                    Array,
                    Array,
                    Array,
                    Array,
                    Array,
                    Array,
                    Array,
                    Array,
                    Array,
                ],
            ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]:
                return (
                    args[0],
                    args[1],
                    args[2],
                    args[3],
                    args[4],
                    args[5],
                    args[6],
                    args[7],
                    jnp.zeros((self._n_features,), dtype=jnp.bool_),
                )

            (
                ops,
                parent_a,
                parent_b,
                theta,
                depth,
                new_utilities,
                ages,
                output_weights,
                cascade_change_mask,
            ) = jax.lax.cond(
                should_promote,
                cascade_after_promote,
                no_cascade,
                (
                    ops,
                    parent_a,
                    parent_b,
                    theta,
                    depth,
                    new_utilities,
                    ages,
                    output_weights,
                    candidate_curation_values,
                    candidate_curation_credit,
                ),
            )
            feature_generator_policy = jnp.where(
                cascade_change_mask,
                generator_policy,
                feature_generator_policy,
            )
            active_change_mask = root_change_mask | cascade_change_mask
        else:

            def replace_active_branch(
                args: tuple[
                    Array, Array, Array, Array, Array, Array, Array, Array, Array, Array
                ],
            ) -> tuple[
                Array, Array, Array, Array, Array, Array, Array, Array, Array, Array
            ]:
                (
                    ops_x,
                    pa_x,
                    pb_x,
                    theta_x,
                    depth_x,
                    util_x,
                    age_x,
                    ow_x,
                    fgp_x,
                    _change_mask_x,
                ) = args
                # Build a fresh composition whose parents are < worst_active.
                # Mask out slots >= worst_active and bias selection by the
                # current utility estimate so productive features become
                # parents more often.  Guidance is evaluated after the gradient
                # update and before the structural root write.
                pre_root_values, pre_root_credit = self._curation_stage_guidance(
                    ops_x,
                    pa_x,
                    pb_x,
                    theta_x,
                    ow_x,
                    observation,
                    errors,
                    active_count,
                )
                op_key, pa_key, pb_key, theta_key = jr.split(proposal_key, 4)
                slot_indices = jnp.arange(self._n_features)
                in_range = slot_indices < worst_active
                depth_ok = depth_x + 1 <= self._max_depth
                eligible = in_range & depth_ok
                logits = self._parent_logits(
                    eligible,
                    util_x,
                    feature_values=pre_root_values,
                    feature_credit=pre_root_credit,
                    depth=depth_x,
                    ages=age_x,
                    parent_mode=parent_mode,
                )
                partner_logits = self._partner_logits(
                    eligible,
                    depth_x,
                    util_x,
                    ages=age_x,
                    parent_mode=parent_mode,
                )
                a_idx = jr.categorical(pa_key, logits).astype(jnp.int32)
                b_idx = jr.categorical(pb_key, partner_logits).astype(jnp.int32)
                new_op = jr.categorical(op_key, self._op_logits(forced_op)).astype(
                    jnp.int32
                )
                new_theta = 0.5 * jr.normal(theta_key, (2,), dtype=jnp.float32)
                new_depth = (
                    jnp.maximum(depth_x[a_idx], depth_x[b_idx]) + 1
                ).astype(jnp.int32)

                ops_n = ops_x.at[worst_active].set(new_op)
                pa_n = pa_x.at[worst_active].set(a_idx)
                pb_n = pb_x.at[worst_active].set(b_idx)
                theta_n = theta_x.at[worst_active].set(new_theta)
                depth_n = depth_x.at[worst_active].set(new_depth)
                util_n = util_x.at[worst_active].set(0.0)
                age_n = age_x.at[worst_active].set(0)
                ow_n = ow_x.at[:, worst_active].set(0.0)
                fgp_n = fgp_x.at[worst_active].set(generator_policy)

                # Cascade guidance sees the installed root and its reset output
                # weights.  Descendant slots are excluded from the eligible
                # parent set, so evaluating this post-root bank is sufficient
                # for every parent the cascade may sample.
                post_root_values, post_root_credit = self._curation_stage_guidance(
                    ops_n,
                    pa_n,
                    pb_n,
                    theta_n,
                    ow_n,
                    observation,
                    errors,
                    active_count,
                )
                composed = ops_n != OP_RAW
                refs_a = composed & (pa_n == worst_active) & (slot_indices > worst_active)
                refs_b = (
                    composed
                    & (pb_n >= 0)
                    & (pb_n == worst_active)
                    & (slot_indices > worst_active)
                )
                replaced_mask = refs_a | refs_b
                (
                    ops_f,
                    pa_f,
                    pb_f,
                    theta_f,
                    depth_f,
                    util_f,
                    age_f,
                    ow_f,
                    cascade_change_mask,
                ) = self._cascade_replace_with_mask(
                    ops_n,
                    pa_n,
                    pb_n,
                    theta_n,
                    depth_n,
                    util_n,
                    age_n,
                    ow_n,
                    replaced_mask,
                    observation,
                    cascade_key,
                    feature_values=post_root_values,
                    feature_credit=post_root_credit,
                    forced_op=forced_op,
                    parent_mode=parent_mode,
                )
                fgp_f = jnp.where(cascade_change_mask, generator_policy, fgp_n)
                applied_change_mask = cascade_change_mask.at[worst_active].set(True)
                return (
                    ops_f,
                    pa_f,
                    pb_f,
                    theta_f,
                    depth_f,
                    util_f,
                    age_f,
                    ow_f,
                    fgp_f,
                    applied_change_mask,
                )

            def keep_active_branch(
                args: tuple[
                    Array, Array, Array, Array, Array, Array, Array, Array, Array, Array
                ],
            ) -> tuple[
                Array, Array, Array, Array, Array, Array, Array, Array, Array, Array
            ]:
                return args

            do_replace = should_try_replace & has_active_slot
            (
                ops,
                parent_a,
                parent_b,
                theta,
                depth,
                new_utilities,
                ages,
                output_weights,
                feature_generator_policy,
                active_change_mask,
            ) = jax.lax.cond(
                do_replace,
                replace_active_branch,
                keep_active_branch,
                (
                    ops,
                    parent_a,
                    parent_b,
                    theta,
                    depth,
                    new_utilities,
                    ages,
                    output_weights,
                    feature_generator_policy,
                    active_change_mask,
                ),
            )
            root_change_applied = do_replace
            root_change_mask = root_change_mask.at[worst_active].set(do_replace)
            cascade_change_mask = active_change_mask & (~root_change_mask)
            proposal_formed = do_replace
            proposal_destination_bank = jnp.where(
                do_replace,
                jnp.asarray(CURATION_DESTINATION_ACTIVE, dtype=jnp.int32),
                proposal_destination_bank,
            )
            proposal_destination_slot = jnp.where(
                do_replace,
                worst_active,
                proposal_destination_slot,
            )
            proposal_op = jnp.where(do_replace, ops[worst_active], proposal_op)
            proposal_parent_a = jnp.where(
                do_replace,
                parent_a[worst_active],
                proposal_parent_a,
            )
            proposal_parent_b = jnp.where(
                do_replace,
                parent_b[worst_active],
                proposal_parent_b,
            )
            proposal_theta = jnp.where(
                do_replace,
                theta[worst_active],
                proposal_theta,
            )
            proposal_depth = jnp.where(
                do_replace,
                depth[worst_active],
                proposal_depth,
            )
            proposal_generator_policy = jnp.where(
                do_replace,
                feature_generator_policy[worst_active],
                proposal_generator_policy,
            )
            post_root_pre_cascade_slot = jnp.where(
                do_replace,
                worst_active,
                post_root_pre_cascade_slot,
            )
            post_root_pre_cascade_op = jnp.where(
                do_replace,
                ops[worst_active],
                post_root_pre_cascade_op,
            )
            post_root_pre_cascade_parent_a = jnp.where(
                do_replace,
                parent_a[worst_active],
                post_root_pre_cascade_parent_a,
            )
            post_root_pre_cascade_parent_b = jnp.where(
                do_replace,
                parent_b[worst_active],
                post_root_pre_cascade_parent_b,
            )
            post_root_pre_cascade_theta = jnp.where(
                do_replace,
                theta[worst_active],
                post_root_pre_cascade_theta,
            )
            post_root_pre_cascade_depth = jnp.where(
                do_replace,
                depth[worst_active],
                post_root_pre_cascade_depth,
            )
            post_root_pre_cascade_generator_policy = jnp.where(
                do_replace,
                feature_generator_policy[worst_active],
                post_root_pre_cascade_generator_policy,
            )
            replaced_slot = jnp.where(do_replace, worst_active, replaced_slot)

        if self._candidate_count > 0:
            # Candidate descriptors are slot-relative.  Replacing an active
            # parent rebinds such a candidate to a new signal and therefore a
            # new structural lifetime identity.  The external event ledger
            # must allocate that identity.  When the rebound remains within the
            # depth budget, preserve its op, parents, and provenance, recompute
            # derived depth, and restart learned state.  If the final parent is
            # now too deep, preserving the descriptor would violate max_depth;
            # regenerate that slot in a separate, trace-visible key domain.
            # A trainable pre-existing TANH theta is learned state and cold
            # resets on a plain rebound.  The source slot refreshed by a
            # promotion is born after the root mutation, so only a subsequent
            # cascade can rebound or regenerate that fresh proposal.
            safe_candidate_pa = jnp.clip(
                candidate_parent_a, 0, self._n_features - 1
            )
            safe_candidate_pb = jnp.clip(
                candidate_parent_b, 0, self._n_features - 1
            )
            references_active_change = (candidate_ops != OP_RAW) & (
                active_change_mask[safe_candidate_pa]
                | (
                    (candidate_parent_b >= 0)
                    & active_change_mask[safe_candidate_pb]
                )
            )
            references_later_cascade = (candidate_ops != OP_RAW) & (
                cascade_change_mask[safe_candidate_pa]
                | (
                    (candidate_parent_b >= 0)
                    & cascade_change_mask[safe_candidate_pb]
                )
            )
            candidate_indices = jnp.arange(self._candidate_count, dtype=jnp.int32)
            refreshed_after_root = should_promote_for_trace & (
                candidate_indices == best_candidate_for_trace
            )
            candidate_parent_change_mask = jnp.where(
                refreshed_after_root,
                references_later_cascade,
                references_active_change,
            )
            rebound_depth = (
                jnp.maximum(
                    depth[safe_candidate_pa],
                    jnp.where(
                        candidate_parent_b >= 0,
                        depth[safe_candidate_pb],
                        jnp.asarray(0, dtype=jnp.int32),
                    ),
                )
                + 1
            ).astype(jnp.int32)
            candidate_overdepth_regeneration_mask = (
                candidate_parent_change_mask & (rebound_depth > self._max_depth)
            )
            candidate_rebound_mask = (
                candidate_parent_change_mask
                & (~candidate_overdepth_regeneration_mask)
            )
            candidate_depth = jnp.where(
                candidate_parent_change_mask,
                rebound_depth,
                candidate_depth,
            )

            def regenerate_overdepth_candidates(
                candidate_args: tuple[
                    Array, Array, Array, Array, Array, Array, Array
                ],
            ) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
                final_feature_values, final_feature_credit = (
                    self._curation_stage_guidance(
                        ops,
                        parent_a,
                        parent_b,
                        theta,
                        output_weights,
                        observation,
                        errors,
                        active_count,
                    )
                )

                def regenerate_one(
                    candidate_index: int,
                    loop_args: tuple[
                        Array, Array, Array, Array, Array, Array, Array
                    ],
                ) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
                    def regenerate_slot(
                        slot_args: tuple[
                            Array, Array, Array, Array, Array, Array, Array
                        ],
                    ) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
                        (
                            candidate_ops_x,
                            candidate_parent_a_x,
                            candidate_parent_b_x,
                            candidate_theta_x,
                            candidate_depth_x,
                            candidate_output_weights_x,
                            candidate_generator_policy_x,
                        ) = slot_args
                        slot_key = jr.fold_in(
                            candidate_overdepth_regeneration_key,
                            jnp.asarray(candidate_index, dtype=jnp.uint32),
                        )
                        gen_op, gen_pa, gen_pb, gen_theta, gen_depth = (
                            self._generate_one(
                                slot_key,
                                depth,
                                new_utilities,
                                existing_ages=ages,
                                feature_values=final_feature_values,
                                feature_credit=final_feature_credit,
                                forced_op=forced_op,
                                parent_mode=parent_mode,
                            )
                        )
                        gen_weights = self._initial_candidate_output_weights(
                            gen_op,
                            gen_pa,
                            gen_pb,
                            gen_theta,
                            final_feature_values,
                            observation,
                            errors,
                            active_count,
                            imprint_scale=imprint_scale,
                        )
                        return (
                            candidate_ops_x.at[candidate_index].set(gen_op),
                            candidate_parent_a_x.at[candidate_index].set(gen_pa),
                            candidate_parent_b_x.at[candidate_index].set(gen_pb),
                            candidate_theta_x.at[candidate_index].set(gen_theta),
                            candidate_depth_x.at[candidate_index].set(gen_depth),
                            candidate_output_weights_x.at[:, candidate_index].set(
                                gen_weights
                            ),
                            candidate_generator_policy_x.at[candidate_index].set(
                                generator_policy
                            ),
                        )

                    def keep_slot(
                        slot_args: tuple[
                            Array, Array, Array, Array, Array, Array, Array
                        ],
                    ) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
                        return slot_args

                    return jax.lax.cond(
                        candidate_overdepth_regeneration_mask[candidate_index],
                        regenerate_slot,
                        keep_slot,
                        loop_args,
                    )

                return jax.lax.fori_loop(
                    0,
                    self._candidate_count,
                    regenerate_one,
                    candidate_args,
                )

            def keep_rebound_candidates(
                candidate_args: tuple[
                    Array, Array, Array, Array, Array, Array, Array
                ],
            ) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
                return candidate_args

            (
                candidate_ops,
                candidate_parent_a,
                candidate_parent_b,
                candidate_theta,
                candidate_depth,
                candidate_output_weights,
                candidate_generator_policy,
            ) = jax.lax.cond(
                jnp.any(candidate_overdepth_regeneration_mask),
                regenerate_overdepth_candidates,
                keep_rebound_candidates,
                (
                    candidate_ops,
                    candidate_parent_a,
                    candidate_parent_b,
                    candidate_theta,
                    candidate_depth,
                    candidate_output_weights,
                    candidate_generator_policy,
                ),
            )
            if self._train_candidate_theta:
                preexisting_trainable_rebound = (
                    candidate_rebound_mask
                    & (~refreshed_after_root)
                    & (candidate_ops == OP_TANH)
                )
                candidate_theta = jnp.where(
                    preexisting_trainable_rebound[:, None],
                    jnp.zeros_like(candidate_theta),
                    candidate_theta,
                )
            candidate_structure_reset_mask = (
                candidate_rebound_mask | candidate_overdepth_regeneration_mask
            )
            candidate_output_weights = jnp.where(
                candidate_rebound_mask[None, :],
                jnp.zeros_like(candidate_output_weights),
                candidate_output_weights,
            )
            new_candidate_utilities = jnp.where(
                candidate_structure_reset_mask,
                jnp.zeros_like(new_candidate_utilities),
                new_candidate_utilities,
            )
            candidate_ages = jnp.where(
                candidate_structure_reset_mask,
                jnp.zeros_like(candidate_ages),
                candidate_ages,
            )

        reset_active_traces = active_change_mask
        if self._candidate_count > 0:
            # Snapshot promotion evidence before the candidate-local reset
            # below.  A freshly generated promoted-candidate slot and any
            # rebound candidates have age zero, but the promoted active root
            # still inherits the source candidate's pre-refresh traces.
            safe_best_candidate = jnp.clip(
                best_candidate_for_trace, 0, self._candidate_count - 1
            )
            promoted_contribution_trace = candidate_utility_contribution_trace[
                :, safe_best_candidate
            ]
            promoted_feature_trace = candidate_utility_feature_trace[
                safe_best_candidate
            ]
            promoted_feature_energy_trace = candidate_utility_feature_energy_trace[
                safe_best_candidate
            ]
            promoted_signal_second_moment = candidate_utility_signal_second_moment[
                safe_best_candidate
            ]
            promoted_score_residual_trace = candidate_score_residual_trace[
                :, safe_best_candidate
            ]
            promoted_score_energy_trace = candidate_score_energy_trace[
                safe_best_candidate
            ]
            promoted_retention_slow_utility = candidate_retention_slow_utilities[
                safe_best_candidate
            ]
        else:
            promoted_contribution_trace = jnp.zeros(
                (self._n_tasks,), dtype=jnp.float32
            )
            promoted_feature_trace = jnp.array(0.0, dtype=jnp.float32)
            promoted_feature_energy_trace = jnp.array(0.0, dtype=jnp.float32)
            promoted_signal_second_moment = jnp.array(0.0, dtype=jnp.float32)
            promoted_score_residual_trace = jnp.zeros(
                (self._n_tasks,), dtype=jnp.float32
            )
            promoted_score_energy_trace = jnp.array(0.0, dtype=jnp.float32)
            promoted_retention_slow_utility = jnp.array(0.0, dtype=jnp.float32)
        utility_contribution_trace = jnp.where(
            reset_active_traces[None, :], 0.0, utility_contribution_trace
        )
        utility_feature_trace = jnp.where(
            reset_active_traces, 0.0, utility_feature_trace
        )
        utility_feature_energy_trace = jnp.where(
            reset_active_traces, 0.0, utility_feature_energy_trace
        )
        utility_signal_second_moment = jnp.where(
            reset_active_traces, 0.0, utility_signal_second_moment
        )
        feature_score_residual_trace = jnp.where(
            reset_active_traces[None, :], 0.0, feature_score_residual_trace
        )
        feature_score_energy_trace = jnp.where(
            reset_active_traces, 0.0, feature_score_energy_trace
        )
        retention_slow_utilities = jnp.where(
            reset_active_traces, 0.0, retention_slow_utilities
        )
        utility_contribution_trace = utility_contribution_trace.at[
            :, promoted_slot_for_trace
        ].set(
            jnp.where(
                should_promote_for_trace,
                promoted_contribution_trace,
                utility_contribution_trace[:, promoted_slot_for_trace],
            )
        )
        utility_feature_trace = utility_feature_trace.at[
            promoted_slot_for_trace
        ].set(
            jnp.where(
                should_promote_for_trace,
                promoted_feature_trace,
                utility_feature_trace[promoted_slot_for_trace],
            )
        )
        utility_feature_energy_trace = utility_feature_energy_trace.at[
            promoted_slot_for_trace
        ].set(
            jnp.where(
                should_promote_for_trace,
                promoted_feature_energy_trace,
                utility_feature_energy_trace[promoted_slot_for_trace],
            )
        )
        utility_signal_second_moment = utility_signal_second_moment.at[
            promoted_slot_for_trace
        ].set(
            jnp.where(
                should_promote_for_trace,
                promoted_signal_second_moment,
                utility_signal_second_moment[promoted_slot_for_trace],
            )
        )
        feature_score_residual_trace = feature_score_residual_trace.at[
            :, promoted_slot_for_trace
        ].set(
            jnp.where(
                should_promote_for_trace,
                promoted_score_residual_trace,
                feature_score_residual_trace[:, promoted_slot_for_trace],
            )
        )
        feature_score_energy_trace = feature_score_energy_trace.at[
            promoted_slot_for_trace
        ].set(
            jnp.where(
                should_promote_for_trace,
                promoted_score_energy_trace,
                feature_score_energy_trace[promoted_slot_for_trace],
            )
        )
        retention_slow_utilities = retention_slow_utilities.at[
            promoted_slot_for_trace
        ].set(
            jnp.where(
                should_promote_for_trace,
                promoted_retention_slow_utility,
                retention_slow_utilities[promoted_slot_for_trace],
            )
        )
        reset_candidate_traces = (candidate_ages == 0) | candidate_rebound_mask
        candidate_utility_contribution_trace = jnp.where(
            reset_candidate_traces[None, :],
            0.0,
            candidate_utility_contribution_trace,
        )
        candidate_utility_feature_trace = jnp.where(
            reset_candidate_traces, 0.0, candidate_utility_feature_trace
        )
        candidate_utility_feature_energy_trace = jnp.where(
            reset_candidate_traces, 0.0, candidate_utility_feature_energy_trace
        )
        candidate_utility_signal_second_moment = jnp.where(
            reset_candidate_traces, 0.0, candidate_utility_signal_second_moment
        )
        candidate_score_residual_trace = jnp.where(
            reset_candidate_traces[None, :], 0.0, candidate_score_residual_trace
        )
        candidate_score_energy_trace = jnp.where(
            reset_candidate_traces, 0.0, candidate_score_energy_trace
        )
        candidate_retention_slow_utilities = jnp.where(
            reset_candidate_traces, 0.0, candidate_retention_slow_utilities
        )
        candidate_active_correlation_trace = jnp.where(
            reset_candidate_traces[:, None] | reset_active_traces[None, :],
            0.0,
            candidate_active_correlation_trace,
        )
        candidate_selector_log_weights = jnp.where(
            reset_candidate_traces, 0.0, candidate_selector_log_weights
        )
        candidate_selector_cumulative_loss = jnp.where(
            reset_candidate_traces, 0.0, candidate_selector_cumulative_loss
        )
        candidate_selector_action_counts = jnp.where(
            reset_candidate_traces, 0.0, candidate_selector_action_counts
        )

        generator_resource_state = state.generator_resource_state
        if self._learn_generator_resources:
            policy_scores, policy_finite = self._generator_policy_scores(
                new_utilities,
                feature_generator_policy,
                new_candidate_utilities,
                candidate_generator_policy,
            )
            if self._generator_resource_promotion_credit > 0.0:
                promoted_policy = feature_generator_policy[promoted_slot_for_trace]
                promotion_bonus = (
                    jnp.asarray(
                        self._generator_resource_promotion_credit,
                        dtype=jnp.float32,
                    )
                    * jnp.maximum(jnp.max(new_candidate_utilities), 0.0)
                )
                policy_ids = jnp.arange(
                    self._generator_resource_manager.n_policies,
                    dtype=jnp.int32,
                )
                promotion_mask = policy_ids == promoted_policy
                policy_scores = policy_scores + jnp.where(
                    promotion_mask,
                    jnp.where(should_promote_for_trace, promotion_bonus, 0.0),
                    0.0,
                )
                policy_finite = policy_finite | (
                    promotion_mask & should_promote_for_trace
                )
            replacement_cost = jnp.asarray(
                DEFAULT_GENERATOR_META_REPLACEMENT_MULTIPLIERS,
                dtype=jnp.float32,
            )
            imprint_cost = jnp.asarray(
                DEFAULT_GENERATOR_META_IMPRINT_SCALES,
                dtype=jnp.float32,
            )
            margin_cost = 1.0 / jnp.asarray(
                DEFAULT_GENERATOR_META_PROMOTION_MARGIN_MULTIPLIERS,
                dtype=jnp.float32,
            )
            age_cost = 1.0 / jnp.asarray(
                DEFAULT_GENERATOR_META_CANDIDATE_MIN_AGE_MULTIPLIERS,
                dtype=jnp.float32,
            )
            # Composite churn proxy per policy, charged via the manager's
            # ``cost_weight`` (a no-op when ``generator_resource_cost_weight``
            # is 0).  Looser margins and shorter trials enter as reciprocals
            # so that more aggressive settings cost more.  The hand-set
            # weights rank replacement churn as the dominant cost, imprint
            # and margin secondary, and trial age smallest.
            policy_costs = (
                replacement_cost
                + 0.25 * imprint_cost
                + 0.25 * margin_cost
                + 0.1 * age_cost
            )
            generator_resource_state = self._generator_resource_manager.update(
                generator_resource_state,
                policy_scores,
                context_id=context,
                finite_mask=policy_finite,
                resource_costs=policy_costs,
                selected_action=decision.action,
                selected_probability=decision.weights[decision.action],
            ).state.replace(
                step_count=_saturating_nonnegative_int32_increment(
                    state.generator_resource_state.step_count
                )
            )

        proposed_state = CompositionalFeatureState(
            key=key,
            ops=ops,
            parent_a=parent_a,
            parent_b=parent_b,
            theta=theta,
            depth=depth,
            output_weights=output_weights,
            output_bias=output_bias,
            utilities=new_utilities,
            utility_contribution_trace=utility_contribution_trace,
            utility_error_trace=utility_error_trace,
            utility_feature_trace=utility_feature_trace,
            utility_feature_energy_trace=utility_feature_energy_trace,
            utility_signal_second_moment=utility_signal_second_moment,
            feature_score_residual_trace=feature_score_residual_trace,
            feature_score_energy_trace=feature_score_energy_trace,
            retention_slow_utilities=retention_slow_utilities,
            task_activity_ema=task_activity_ema,
            ages=ages,
            candidate_ops=candidate_ops,
            candidate_parent_a=candidate_parent_a,
            candidate_parent_b=candidate_parent_b,
            candidate_theta=candidate_theta,
            candidate_depth=candidate_depth,
            candidate_output_weights=candidate_output_weights,
            candidate_utilities=new_candidate_utilities,
            candidate_utility_contribution_trace=candidate_utility_contribution_trace,
            candidate_utility_feature_trace=candidate_utility_feature_trace,
            candidate_utility_feature_energy_trace=(
                candidate_utility_feature_energy_trace
            ),
            candidate_utility_signal_second_moment=(
                candidate_utility_signal_second_moment
            ),
            candidate_score_residual_trace=candidate_score_residual_trace,
            candidate_score_energy_trace=candidate_score_energy_trace,
            candidate_retention_slow_utilities=candidate_retention_slow_utilities,
            candidate_active_correlation_trace=candidate_active_correlation_trace,
            candidate_ages=candidate_ages,
            candidate_selector_log_weights=candidate_selector_log_weights,
            candidate_selector_cumulative_loss=candidate_selector_cumulative_loss,
            candidate_selector_action_counts=candidate_selector_action_counts,
            feature_generator_policy=feature_generator_policy,
            candidate_generator_policy=candidate_generator_policy,
            generator_resource_state=generator_resource_state,
            replacement_accumulator=replacement_accumulator,
            step_count=step_count,
            step_words=step_words,
            replacement_phase=replacement_phase,
            birth_timestamp=state.birth_timestamp,
            uptime_s=state.uptime_s,
        )
        commit_available = update_available
        if (
            self._candidate_novelty_admission_bonus > 0.0
            or self._ancestor_utility_backup_decay > 0.0
        ):
            commit_available = (
                commit_available
                & self._ranking_topology_valid(
                    proposed_state,
                    observation.shape[0],
                )
                & self._ranking_state_finite(proposed_state)
            )
        new_state = jax.lax.cond(
            commit_available,
            lambda: proposed_state,
            lambda: state,
        )

        if (
            self._candidate_novelty_admission_bonus > 0.0
            or self._ancestor_utility_backup_decay > 0.0
        ):
            # A rejected proposal is not a curation event.  Mask every event
            # identity back to the established no-event representation while
            # preserving score/prediction diagnostics for debugging.
            replaced_slot = jnp.where(commit_available, replaced_slot, -1)
            promoted_candidate = jnp.where(commit_available, promoted_candidate, -1)
            should_try_replace = should_try_replace & commit_available
            proposal_formed = proposal_formed & commit_available
            root_change_applied = root_change_applied & commit_available
            should_promote_for_trace = should_promote_for_trace & commit_available
            root_change_mask = root_change_mask & commit_available
            cascade_change_mask = cascade_change_mask & commit_available
            active_change_mask = active_change_mask & commit_available
            ordinary_candidate_refresh_mask = (
                ordinary_candidate_refresh_mask & commit_available
            )
            post_promotion_candidate_refresh_mask = (
                post_promotion_candidate_refresh_mask & commit_available
            )
            candidate_rebound_mask = candidate_rebound_mask & commit_available
            candidate_overdepth_regeneration_mask = (
                candidate_overdepth_regeneration_mask & commit_available
            )
            proposal_destination_bank = jnp.where(
                commit_available,
                proposal_destination_bank,
                CURATION_DESTINATION_NONE,
            )
            for_no_event = (
                proposal_destination_slot,
                proposal_op,
                proposal_parent_a,
                proposal_parent_b,
                proposal_depth,
                proposal_generator_policy,
                post_root_pre_cascade_slot,
                post_root_pre_cascade_op,
                post_root_pre_cascade_parent_a,
                post_root_pre_cascade_parent_b,
                post_root_pre_cascade_depth,
                post_root_pre_cascade_generator_policy,
                promotion_source_candidate,
                promotion_destination_active,
                promoted_pre_refresh_op,
                promoted_pre_refresh_parent_a,
                promoted_pre_refresh_parent_b,
                promoted_pre_refresh_depth,
                promoted_pre_refresh_generator_policy,
            )
            (
                proposal_destination_slot,
                proposal_op,
                proposal_parent_a,
                proposal_parent_b,
                proposal_depth,
                proposal_generator_policy,
                post_root_pre_cascade_slot,
                post_root_pre_cascade_op,
                post_root_pre_cascade_parent_a,
                post_root_pre_cascade_parent_b,
                post_root_pre_cascade_depth,
                post_root_pre_cascade_generator_policy,
                promotion_source_candidate,
                promotion_destination_active,
                promoted_pre_refresh_op,
                promoted_pre_refresh_parent_a,
                promoted_pre_refresh_parent_b,
                promoted_pre_refresh_depth,
                promoted_pre_refresh_generator_policy,
            ) = tuple(
                jnp.where(commit_available, value, absent_index)
                for value in for_no_event
            )
            proposal_theta = jnp.where(
                commit_available, proposal_theta, absent_theta
            )
            post_root_pre_cascade_theta = jnp.where(
                commit_available, post_root_pre_cascade_theta, absent_theta
            )
            promoted_pre_refresh_theta = jnp.where(
                commit_available, promoted_pre_refresh_theta, absent_theta
            )

        loss = jnp.sum(errors**2) / active_count
        mean_abs_error = jnp.sum(jnp.abs(errors)) / active_count
        max_candidate_utility = (
            jnp.max(new_candidate_utilities)
            if self._candidate_count > 0
            else jnp.array(0.0, dtype=jnp.float32)
        )
        replacement_flag = (replaced_slot >= 0).astype(jnp.float32)
        metrics = jnp.array(
            [
                loss,
                mean_abs_error,
                jnp.mean(new_utilities),
                jnp.min(new_utilities),
                max_candidate_utility,
                replacement_flag,
                bounding_scale,
            ],
            dtype=jnp.float32,
        )

        candidate_refresh_mask = (
            ordinary_candidate_refresh_mask | post_promotion_candidate_refresh_mask
        )
        proposal_count = proposal_formed.astype(jnp.int32)
        root_change_count = jnp.sum(root_change_mask.astype(jnp.int32))
        promotion_count = should_promote_for_trace.astype(jnp.int32)
        cascade_refill_count = jnp.sum(cascade_change_mask.astype(jnp.int32))
        ordinary_candidate_refresh_count = jnp.sum(
            ordinary_candidate_refresh_mask.astype(jnp.int32)
        )
        post_promotion_candidate_refresh_count = jnp.sum(
            post_promotion_candidate_refresh_mask.astype(jnp.int32)
        )
        candidate_refresh_count = jnp.sum(candidate_refresh_mask.astype(jnp.int32))
        candidate_rebound_count = jnp.sum(candidate_rebound_mask.astype(jnp.int32))
        candidate_overdepth_regeneration_count = jnp.sum(
            candidate_overdepth_regeneration_mask.astype(jnp.int32)
        )
        # Proposal formation is the cause of an active-root or candidate-slot
        # birth, not a second ledger event.  Promotion classifies the root
        # change.  Cascades, plain rebounds, and over-depth regenerations each
        # create additional structural lifetimes and therefore contribute
        # independently.  A repaired post-promotion proposal contributes both
        # its transient refresh birth and its final regeneration birth.
        logical_event_count = (
            root_change_count
            + candidate_refresh_count
            + cascade_refill_count
            + candidate_rebound_count
            + candidate_overdepth_regeneration_count
        )
        curation_trace = CompositionalCurationTrace(
            pre_step=state.step_count,
            post_step=new_state.step_count,
            pre_step_words=state.step_words,
            post_step_words=new_state.step_words,
            pre_replacement_phase=state.replacement_phase,
            post_replacement_phase=new_state.replacement_phase,
            lifetime_counter_valid=lifetime_counter_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            decision_key=decision_key,
            curation_key=curation_key,
            proposal_key=proposal_key,
            cascade_key=cascade_key,
            candidate_overdepth_regeneration_key=(
                candidate_overdepth_regeneration_key
            ),
            should_try_replace=jnp.asarray(should_try_replace, dtype=jnp.bool_),
            has_event=logical_event_count > 0,
            generator_policy_sampled=jnp.asarray(
                self._learn_generator_resources, dtype=jnp.bool_
            ),
            generator_policy_id=generator_policy,
            decision_update_available=update_available,
            decision_commit_available=commit_available,
            decision_active_ops=state.ops,
            decision_active_parent_a=state.parent_a,
            decision_active_parent_b=state.parent_b,
            decision_active_theta=decision_active_theta,
            decision_active_depth=state.depth,
            decision_active_generator_policy=state.feature_generator_policy,
            decision_active_ages=decision_active_ages,
            decision_active_fast_utilities=decision_active_fast_utilities,
            decision_active_slow_utilities=decision_active_slow_utilities,
            decision_active_direct_scores=active_direct_replacement_score,
            decision_active_backed_scores=active_replacement_score,
            decision_active_eligible=eligible_active,
            decision_active_selection_scores=active_scores,
            decision_worst_active=jnp.where(
                has_active_slot, worst_active, absent_index
            ).astype(jnp.int32),
            decision_has_active_slot=has_active_slot,
            decision_candidate_ops=state.candidate_ops,
            decision_candidate_parent_a=state.candidate_parent_a,
            decision_candidate_parent_b=state.candidate_parent_b,
            decision_candidate_theta=decision_candidate_theta,
            decision_candidate_depth=state.candidate_depth,
            decision_candidate_generator_policy=(
                state.candidate_generator_policy
            ),
            decision_candidate_ages=decision_candidate_ages,
            decision_candidate_fast_utilities=(
                decision_candidate_fast_utilities
            ),
            decision_candidate_slow_utilities=(
                decision_candidate_slow_utilities
            ),
            decision_candidate_direct_scores=(
                candidate_direct_promotion_scores
            ),
            decision_candidate_novelty_scores=candidate_admission_novelty,
            decision_candidate_augmented_scores=candidate_promotion_scores,
            decision_candidate_mature=eligible_candidates,
            decision_candidate_recomputed_depth=candidate_depth_after_all,
            decision_candidate_topology_compatible=(
                candidate_topology_compatible
            ),
            decision_candidate_depth_compatible=candidate_depth_compatible,
            decision_candidate_headroom_compatible=(
                candidate_headroom_compatible
            ),
            decision_candidate_margin_eligible=candidate_margin_eligible,
            decision_candidate_destination_compatible=(
                compatible_active_by_candidate
            ),
            decision_candidate_has_destination=candidate_has_destination,
            decision_candidate_ranking_scores=candidate_ranking_scores,
            decision_candidate_refresh_utilities=(
                candidate_refresh_utilities
            ),
            decision_selected_candidate=selected_candidate_for_audit,
            decision_has_candidate=has_candidate,
            decision_selected_destination=selected_destination_for_audit,
            decision_selected_refresh_candidate=(
                selected_refresh_candidate_for_audit
            ),
            decision_has_refresh_candidate=has_refresh_candidate,
            decision_left_pack_destinations_enabled=jnp.asarray(
                self._topology_left_pack_destinations, dtype=jnp.bool_
            ),
            decision_left_pack_destination_available=(
                left_pack_destination_available
            ),
            decision_effective_promotion_margin=promotion_margin,
            decision_selected_candidate_score=(
                selected_candidate_score_for_audit
            ),
            decision_selected_destination_backed_score=(
                selected_destination_score_for_audit
            ),
            decision_margin_rhs=margin_rhs_for_audit,
            decision_margin_passed=margin_passed_for_audit,
            decision_selected_topology_ok=selected_topology_ok_for_audit,
            decision_selected_depth_ok=selected_depth_ok_for_audit,
            decision_selected_headroom_ok=selected_headroom_ok_for_audit,
            decision_selected_can_promote=selected_can_promote_for_audit,
            decision_should_promote=should_promote_for_audit,
            decision_should_refresh=should_refresh_for_audit,
            proposal_formed=proposal_formed,
            proposal_destination_bank=proposal_destination_bank,
            proposal_destination_slot=proposal_destination_slot,
            proposal_op=proposal_op,
            proposal_parent_a=proposal_parent_a,
            proposal_parent_b=proposal_parent_b,
            proposal_theta=proposal_theta,
            proposal_depth=proposal_depth,
            proposal_generator_policy=proposal_generator_policy,
            root_change_mask=root_change_mask,
            root_change_applied=root_change_applied,
            post_root_pre_cascade_slot=post_root_pre_cascade_slot,
            post_root_pre_cascade_op=post_root_pre_cascade_op,
            post_root_pre_cascade_parent_a=post_root_pre_cascade_parent_a,
            post_root_pre_cascade_parent_b=post_root_pre_cascade_parent_b,
            post_root_pre_cascade_theta=post_root_pre_cascade_theta,
            post_root_pre_cascade_depth=post_root_pre_cascade_depth,
            post_root_pre_cascade_generator_policy=(
                post_root_pre_cascade_generator_policy
            ),
            promotion_applied=should_promote_for_trace,
            promotion_source_candidate=promotion_source_candidate,
            promotion_destination_active=promotion_destination_active,
            promoted_pre_refresh_op=promoted_pre_refresh_op,
            promoted_pre_refresh_parent_a=promoted_pre_refresh_parent_a,
            promoted_pre_refresh_parent_b=promoted_pre_refresh_parent_b,
            promoted_pre_refresh_theta=promoted_pre_refresh_theta,
            promoted_pre_refresh_depth=promoted_pre_refresh_depth,
            promoted_pre_refresh_generator_policy=(
                promoted_pre_refresh_generator_policy
            ),
            cascade_refill_mask=cascade_change_mask,
            cascade_final_ops=new_state.ops,
            cascade_final_parent_a=new_state.parent_a,
            cascade_final_parent_b=new_state.parent_b,
            cascade_final_theta=new_state.theta,
            cascade_final_depth=new_state.depth,
            cascade_final_generator_policy=new_state.feature_generator_policy,
            active_change_mask=active_change_mask,
            ordinary_candidate_refresh_mask=ordinary_candidate_refresh_mask,
            post_promotion_candidate_refresh_mask=(
                post_promotion_candidate_refresh_mask
            ),
            candidate_refresh_mask=candidate_refresh_mask,
            candidate_rebound_mask=candidate_rebound_mask,
            candidate_overdepth_regeneration_mask=(
                candidate_overdepth_regeneration_mask
            ),
            candidate_final_ops=new_state.candidate_ops,
            candidate_final_parent_a=new_state.candidate_parent_a,
            candidate_final_parent_b=new_state.candidate_parent_b,
            candidate_final_theta=new_state.candidate_theta,
            candidate_final_depth=new_state.candidate_depth,
            candidate_final_generator_policy=new_state.candidate_generator_policy,
            proposal_count=proposal_count,
            root_change_count=root_change_count,
            promotion_count=promotion_count,
            cascade_refill_count=cascade_refill_count,
            ordinary_candidate_refresh_count=ordinary_candidate_refresh_count,
            post_promotion_candidate_refresh_count=(
                post_promotion_candidate_refresh_count
            ),
            candidate_refresh_count=candidate_refresh_count,
            candidate_rebound_count=candidate_rebound_count,
            candidate_overdepth_regeneration_count=(
                candidate_overdepth_regeneration_count
            ),
            logical_event_count=logical_event_count,
        )

        return CompositionalFeatureUpdateResult(
            state=new_state,
            predictions=predictions,
            errors=reported_errors,
            metrics=metrics,
            replaced_slot=replaced_slot,
            promoted_candidate=promoted_candidate,
            curation_trace=curation_trace,
        )


def run_compositional_arrays(
    learner: CompositionalFeatureLearner,
    state: CompositionalFeatureState,
    observations: Array,
    targets: Array,
) -> CompositionalFeatureLearningResult:
    """Run a compositional learner over pre-collected stream arrays."""

    def step_fn(
        carry: CompositionalFeatureState,
        inputs: tuple[Array, Array],
    ) -> tuple[CompositionalFeatureState, Array]:
        observation, target = inputs
        result = learner.update(carry, observation, target)
        return result.state, result.metrics

    t0 = time.time()
    final_state, metrics = jax.lax.scan(step_fn, state, (observations, targets))
    elapsed = time.time() - t0
    final_state = final_state.replace(  # type: ignore[attr-defined]
        uptime_s=final_state.uptime_s + elapsed
    )
    return CompositionalFeatureLearningResult(state=final_state, metrics=metrics)


def run_compositional_loop(
    learner: CompositionalFeatureLearner,
    stream: Any,
    num_steps: int,
    key: Array,
    learner_state: CompositionalFeatureState | None = None,
) -> CompositionalFeatureLearningResult:
    """Run compositional feature discovery directly from a scan-compatible stream."""
    stream_key, learner_key = jr.split(key)
    stream_state = stream.init(stream_key)
    if learner_state is None:
        learner_state = learner.init(stream.feature_dim, learner_key)

    def step_fn(
        carry: tuple[CompositionalFeatureState, Any],
        idx: Array,
    ) -> tuple[tuple[CompositionalFeatureState, Any], Array]:
        l_state, s_state = carry
        timestep, new_s_state = stream.step(s_state, idx)
        result = learner.update(l_state, timestep.observation, timestep.target)
        return (result.state, new_s_state), result.metrics

    t0 = time.time()
    (final_state, _), metrics = jax.lax.scan(
        step_fn, (learner_state, stream_state), jnp.arange(num_steps)
    )
    elapsed = time.time() - t0
    final_state = final_state.replace(  # type: ignore[attr-defined]
        uptime_s=final_state.uptime_s + elapsed
    )
    return CompositionalFeatureLearningResult(state=final_state, metrics=metrics)
