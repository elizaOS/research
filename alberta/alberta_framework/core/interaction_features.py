"""Fixed-budget pairwise interaction feature discovery.

This module is a concrete Step 2 probe.  It restricts feature construction to
pairwise products of existing scalar observations, then studies whether an
online learner can manage a bounded set of those constructed features by
testing, scoring, promoting, and replacing them.
"""

import functools
import math
import time
from numbers import Real
from pathlib import Path
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Bool, Float, Int, PRNGKeyArray

from alberta_framework.core.checkpoints import (
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)
from alberta_framework.core.feature_discovery import (
    GENERATOR_IMPRINT,
    GENERATOR_MUTATE_PARENT,
    GENERATOR_RANDOM,
)
from alberta_framework.core.future_utility import one_step_output_loss_reduction

_INT32_MAX = 2**31 - 1

RELEVANCE_PROBE_MODE_CONDITIONAL_V1 = "conditional_v1"
RELEVANCE_PROBE_MODE_TARGET_ONLY_V1 = "target_only_v1"
RELEVANCE_PROBE_MODES = frozenset(
    {
        RELEVANCE_PROBE_MODE_CONDITIONAL_V1,
        RELEVANCE_PROBE_MODE_TARGET_ONLY_V1,
    }
)
"""Versioned semantics supported by the fixed-shape relevance-probe state."""

INTERACTION_FEATURE_CHECKPOINT_SCHEMA = "alberta.interaction-feature-checkpoint.v1"


def _require_array_contract(
    value: Array,
    *,
    name: str,
    shape: tuple[int, ...] | None,
    dtype: jnp.dtype,
) -> Array:
    """Return an array after trace-time shape and effective-dtype validation."""

    array = jnp.asarray(value)
    if shape is None:
        if array.ndim != 1:
            raise ValueError(f"{name} must be rank one, got shape {array.shape}")
    elif array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {array.dtype}")
    return array


def _saturating_int32_increment(value: Array) -> Array:
    """Increment a non-negative int32 counter without wraparound."""

    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    counter = jnp.asarray(value, dtype=jnp.int32)
    return jnp.minimum(jnp.maximum(counter, 0), maximum - 1) + 1


@chex.dataclass(frozen=True)
class InteractionFeatureState:
    """State for ``FixedBudgetInteractionLearner``."""

    key: PRNGKeyArray
    feature_left: Int[Array, " n_features"]
    feature_right: Int[Array, " n_features"]
    output_weights: Float[Array, "n_tasks n_features"]
    relevance_probe_weights: Float[Array, "n_tasks n_features"]
    relevance_probe_biases: Float[Array, " n_tasks"]
    output_biases: Float[Array, " n_tasks"]
    utilities: Float[Array, " n_features"]
    evidence_idle_steps: Int[Array, " n_features"]
    utility_evidence_streak: Int[Array, " n_features"]
    active_output_memory_committed: Bool[Array, " n_features"]
    task_activity_ema: Float[Array, " n_tasks"]
    ages: Int[Array, " n_features"]
    candidate_left: Int[Array, " n_candidates"]
    candidate_right: Int[Array, " n_candidates"]
    candidate_output_weights: Float[Array, "n_tasks n_candidates"]
    candidate_utilities: Float[Array, " n_candidates"]
    candidate_ages: Int[Array, " n_candidates"]
    candidate_promotion_evidence_streak: Int[Array, " n_candidates"]
    candidate_reacquisition_required: Bool[Array, " n_candidates"]
    feature_second_moments: Float[Array, " n_feature_normalizers"]
    candidate_second_moments: Float[Array, " n_candidate_normalizers"]
    target_second_moments: Float[Array, " n_target_normalizers"]
    feature_parent_a: Int[Array, " n_features"]
    feature_parent_b: Int[Array, " n_features"]
    feature_generator: Int[Array, " n_features"]
    candidate_parent_a: Int[Array, " n_candidates"]
    candidate_parent_b: Int[Array, " n_candidates"]
    candidate_generator: Int[Array, " n_candidates"]
    step_count: Int[Array, ""]
    birth_timestamp: Float[Array, ""]
    uptime_s: Float[Array, ""]


@chex.dataclass(frozen=True)
class InteractionCurationPriorityOverride:
    """Transient fixed-shape ranks used only at the curation boundary."""

    enabled: Bool[Array, ""]
    active_ranks: Float[Array, " n_features"]
    candidate_ranks: Float[Array, " n_candidates"]


@chex.dataclass(frozen=True)
class InteractionFeatureUpdateResult:
    """Result of one pairwise interaction feature update."""

    state: InteractionFeatureState
    pre_curation_state: InteractionFeatureState
    predictions: Float[Array, " n_tasks"]
    errors: Float[Array, " n_tasks"]
    metrics: Float[Array, " 7"]
    replaced_slot: Int[Array, ""]
    promoted_candidate: Int[Array, ""]
    refreshed_candidate: Int[Array, ""]
    retired_slot: Int[Array, ""]
    retired_left: Int[Array, ""]
    retired_right: Int[Array, ""]
    evidence_refreshed: Bool[Array, " n_features"]
    retention_evidence_refreshed: Bool[Array, " n_features"]
    relevance_probe_scores: Float[Array, " n_features"]
    relevance_probe_errors: Float[Array, "n_tasks n_features"]
    candidate_promotion_signal: Float[Array, " n_candidates"]
    candidate_promotion_raw_evidence: Bool[Array, " n_candidates"]
    candidate_promotion_evidence_streak_pre: Int[Array, " n_candidates"]
    candidate_promotion_evidence_streak_updated: Int[Array, " n_candidates"]
    candidate_promotion_confirmed: Bool[Array, " n_candidates"]
    candidate_reacquisition_required_pre: Bool[Array, " n_candidates"]
    candidate_reacquisition_required_post: Bool[Array, " n_candidates"]
    candidate_reacquisition_confirmed: Bool[Array, " n_candidates"]
    durable_read_mask: Bool[Array, " n_features"]
    matching_candidate_reset_mask: Bool[Array, " n_candidates"]
    live_feature_count: Int[Array, ""]
    vacancy_count: Int[Array, ""]
    promoted_into_vacancy: Bool[Array, ""]
    curation_priority_override_enabled: Bool[Array, ""]
    curation_attempted: Bool[Array, ""]
    curation_priority_override_applied: Bool[Array, ""]
    curation_selected_active_worst_slot: Int[Array, ""]
    curation_selected_promotion_candidate: Int[Array, ""]
    curation_selected_refresh_candidate: Int[Array, ""]
    update_rejected: Bool[Array, ""]


@chex.dataclass(frozen=True)
class InteractionFeatureLearningResult:
    """Result from a scan-based interaction feature run."""

    state: InteractionFeatureState
    metrics: Float[Array, "num_steps 7"]


class FixedBudgetInteractionLearner:
    """Fixed-budget learner over constructed pairwise product features.

    Active features are products ``x[left] * x[right]``.  Optional candidates
    are trained against residual error without contributing to predictions.
    Periodic replacement either promotes the best candidate into the worst
    active slot or refreshes the worst candidate.

    ``scale_robust=True`` opts into per-coordinate normalized output updates
    and signed loss-intervention utilities.  The default remains the original
    LMS/OBGD contribution-utility path.  Robust mode adds one second moment
    per active feature and candidate plus one per task; this fixed cost is
    exposed by :meth:`memory_accounting`.

    With ``independent_relevance_probe=True``, ``conditional_v1`` (the default)
    makes candidate heads learn the residual of the currently readable durable
    bank and active head ``i`` learn its leave-one-out residual.  The explicit
    ``target_only_v1`` alternative instead gives every isolated one-coordinate
    head the same target-minus-probe-bias baseline, so its evidence is invariant
    to deployed durable-bank weights.  Both modes use the same fixed state and
    linear compute. Public updates enforce trace-time array contracts;
    nonfinite dynamic input is an explicit atomic no-op reported by
    ``update_rejected``.
    """

    def __init__(
        self,
        n_features: int,
        n_tasks: int,
        step_size_output: float = 0.03,
        utility_decay: float = 0.995,
        replacement_interval: int = 100,
        min_feature_age: int = 50,
        candidate_count: int = 0,
        candidate_min_age: int = 25,
        promotion_margin: float = 1.05,
        promotion_blend: float = 1.0,
        generator_mix: tuple[float, float, float] = (1.0, 0.0, 0.0),
        candidate_strategy: str = "random",
        utility_aggregation: str = "mean",
        utility_top_k: int = 1,
        utility_task_balancing: str = "none",
        task_activity_decay: float = 0.995,
        future_utility_mix: float = 0.0,
        utility_retention_decay: float | None = None,
        utility_retention_grace_steps: int | None = None,
        utility_evidence_threshold: float = 0.0,
        evidence_gated_active_output_memory: bool = False,
        utility_evidence_confirmation_steps: int = 1,
        independent_relevance_probe: bool = False,
        relevance_probe_mode: str = RELEVANCE_PROBE_MODE_CONDITIONAL_V1,
        retire_stale_features: bool = False,
        candidate_promotion_floor: float = 0.0,
        candidate_promotion_confirmation_steps: int = 1,
        candidate_reacquisition_confirmation_steps: int = 1,
        candidate_utility_retention_decay: float | None = None,
        refresh_candidates: bool = True,
        refresh_promoted_candidate: bool = True,
        include_squares: bool = False,
        use_obgd: bool = True,
        obgd_kappa: float = 2.0,
        scale_robust: bool = False,
        scale_normalizer_decay: float = 0.99,
        scale_normalizer_epsilon: float = 1e-6,
    ):
        if n_features < 1:
            raise ValueError("n_features must be positive")
        if n_tasks < 1:
            raise ValueError("n_tasks must be positive")
        if candidate_count < 0:
            raise ValueError("candidate_count must be non-negative")
        if not 0.0 <= utility_decay < 1.0:
            raise ValueError("utility_decay must be in [0, 1)")
        if replacement_interval < 0:
            raise ValueError("replacement_interval must be non-negative")
        if (
            isinstance(promotion_margin, bool)
            or not isinstance(cast(object, promotion_margin), Real)
            or not math.isfinite(float(promotion_margin))
            or promotion_margin <= 0.0
        ):
            raise ValueError("promotion_margin must be finite and positive")
        if not 0.0 <= promotion_blend <= 1.0:
            raise ValueError("promotion_blend must be in [0, 1]")
        if candidate_strategy not in {"random", "all_pairs"}:
            raise ValueError("candidate_strategy must be 'random' or 'all_pairs'")
        if utility_aggregation not in {"mean", "max", "topk"}:
            raise ValueError("utility_aggregation must be 'mean', 'max', or 'topk'")
        if utility_top_k < 1:
            raise ValueError("utility_top_k must be positive")
        if utility_task_balancing not in {"none", "active", "active_inverse_frequency"}:
            raise ValueError(
                "utility_task_balancing must be 'none', 'active', or 'active_inverse_frequency'"
            )
        if not 0.0 <= task_activity_decay < 1.0:
            raise ValueError("task_activity_decay must be in [0, 1)")
        if not 0.0 <= future_utility_mix <= 1.0:
            raise ValueError("future_utility_mix must be in [0, 1]")
        if utility_retention_decay is not None and not (
            utility_decay <= utility_retention_decay < 1.0
        ):
            raise ValueError("utility_retention_decay must be in [utility_decay, 1) when set")
        if utility_retention_grace_steps is not None and (
            isinstance(utility_retention_grace_steps, bool)
            or not isinstance(utility_retention_grace_steps, int)
            or not 0 <= utility_retention_grace_steps < 2**31 - 1
        ):
            raise ValueError(
                "utility_retention_grace_steps must be None or an int32-safe non-negative integer"
            )
        if (
            isinstance(utility_evidence_threshold, bool)
            or not isinstance(cast(object, utility_evidence_threshold), Real)
            or not math.isfinite(float(utility_evidence_threshold))
            or utility_evidence_threshold < 0.0
        ):
            raise ValueError("utility_evidence_threshold must be finite and non-negative")
        if utility_retention_grace_steps is not None and utility_evidence_threshold <= 0.0:
            raise ValueError(
                "utility_evidence_threshold must be positive when an evidence lease is enabled"
            )
        if not isinstance(evidence_gated_active_output_memory, bool):
            raise ValueError("evidence_gated_active_output_memory must be boolean")
        if (
            isinstance(utility_evidence_confirmation_steps, bool)
            or not isinstance(utility_evidence_confirmation_steps, int)
            or not 0 <= utility_evidence_confirmation_steps < 2**31 - 1
        ):
            raise ValueError(
                "utility_evidence_confirmation_steps must be a non-negative "
                "int32-safe integer"
            )
        if evidence_gated_active_output_memory and utility_retention_grace_steps is None:
            raise ValueError(
                "evidence_gated_active_output_memory requires utility_retention_grace_steps"
            )
        if evidence_gated_active_output_memory and utility_evidence_threshold <= 0.0:
            raise ValueError(
                "evidence_gated_active_output_memory requires a positive "
                "utility_evidence_threshold"
            )
        if evidence_gated_active_output_memory and utility_evidence_confirmation_steps <= 0:
            raise ValueError(
                "utility_evidence_confirmation_steps must be positive when "
                "evidence_gated_active_output_memory is enabled"
            )
        if not isinstance(independent_relevance_probe, bool):
            raise ValueError("independent_relevance_probe must be boolean")
        if (
            not isinstance(relevance_probe_mode, str)
            or relevance_probe_mode not in RELEVANCE_PROBE_MODES
        ):
            raise ValueError(
                "relevance_probe_mode must be 'conditional_v1' or "
                "'target_only_v1'"
            )
        if independent_relevance_probe and not evidence_gated_active_output_memory:
            raise ValueError(
                "independent_relevance_probe requires "
                "evidence_gated_active_output_memory"
            )
        if not isinstance(retire_stale_features, bool):
            raise ValueError("retire_stale_features must be boolean")
        if retire_stale_features and utility_retention_grace_steps is None:
            raise ValueError("retire_stale_features requires utility_retention_grace_steps")
        if (
            isinstance(candidate_promotion_floor, bool)
            or not isinstance(cast(object, candidate_promotion_floor), Real)
            or not math.isfinite(float(candidate_promotion_floor))
            or candidate_promotion_floor < 0.0
        ):
            raise ValueError("candidate_promotion_floor must be finite and non-negative")
        if retire_stale_features and replacement_interval <= 0:
            raise ValueError("retire_stale_features requires a positive replacement_interval")
        if retire_stale_features and candidate_promotion_floor <= 0.0:
            raise ValueError("retire_stale_features requires a positive candidate_promotion_floor")
        if (
            isinstance(candidate_promotion_confirmation_steps, bool)
            or not isinstance(candidate_promotion_confirmation_steps, int)
            or not 1 <= candidate_promotion_confirmation_steps < 2**31 - 1
        ):
            raise ValueError(
                "candidate_promotion_confirmation_steps must be a positive "
                "int32-safe integer"
            )
        if (
            isinstance(candidate_reacquisition_confirmation_steps, bool)
            or not isinstance(candidate_reacquisition_confirmation_steps, int)
            or not 1 <= candidate_reacquisition_confirmation_steps < 2**31 - 1
        ):
            raise ValueError(
                "candidate_reacquisition_confirmation_steps must be a positive "
                "int32-safe integer"
            )
        # Only a retirement raises candidate_reacquisition_required. A value
        # above one is intentionally allowed in the matched no-retirement arm,
        # where the threshold remains inert and all fixed-shape state is kept.
        if candidate_reacquisition_confirmation_steps > 1 and not independent_relevance_probe:
            raise ValueError(
                "candidate_reacquisition_confirmation_steps greater than one "
                "requires independent_relevance_probe"
            )
        if candidate_utility_retention_decay is not None and not (
            utility_decay <= candidate_utility_retention_decay < 1.0
        ):
            raise ValueError(
                "candidate_utility_retention_decay must be in [utility_decay, 1) when set"
            )
        if not 0.0 <= scale_normalizer_decay < 1.0:
            raise ValueError("scale_normalizer_decay must be in [0, 1)")
        if scale_normalizer_epsilon <= 0.0:
            raise ValueError("scale_normalizer_epsilon must be positive")
        if scale_robust and future_utility_mix != 0.0:
            raise ValueError(
                "future_utility_mix must be 0 when scale_robust=True; "
                "the robust loss proxy replaces the raw LMS counterfactual"
            )

        mix = jnp.array(generator_mix, dtype=jnp.float32)
        if mix.shape != (3,):
            raise ValueError("generator_mix must have three entries")
        mix_sum = float(jnp.sum(mix))
        if mix_sum <= 0.0:
            raise ValueError("generator_mix must contain positive mass")

        self._n_features = n_features
        self._n_tasks = n_tasks
        self._step_size_output = step_size_output
        self._utility_decay = utility_decay
        self._replacement_interval = replacement_interval
        self._min_feature_age = min_feature_age
        self._candidate_count = candidate_count
        self._candidate_min_age = candidate_min_age
        self._promotion_margin = promotion_margin
        self._promotion_blend = promotion_blend
        self._generator_mix = tuple(float(v) / mix_sum for v in generator_mix)
        self._candidate_strategy = candidate_strategy
        self._utility_aggregation = utility_aggregation
        self._utility_top_k = utility_top_k
        self._utility_task_balancing = utility_task_balancing
        self._task_activity_decay = task_activity_decay
        self._future_utility_mix = future_utility_mix
        self._utility_retention_decay = utility_retention_decay
        self._utility_retention_grace_steps = utility_retention_grace_steps
        self._utility_evidence_threshold = utility_evidence_threshold
        self._evidence_gated_active_output_memory = evidence_gated_active_output_memory
        self._utility_evidence_confirmation_steps = utility_evidence_confirmation_steps
        self._independent_relevance_probe = independent_relevance_probe
        self._relevance_probe_mode = relevance_probe_mode
        self._retire_stale_features = retire_stale_features
        self._candidate_promotion_floor = candidate_promotion_floor
        self._candidate_promotion_confirmation_steps = (
            candidate_promotion_confirmation_steps
        )
        self._candidate_reacquisition_confirmation_steps = (
            candidate_reacquisition_confirmation_steps
        )
        self._candidate_utility_retention_decay = candidate_utility_retention_decay
        self._refresh_candidates = refresh_candidates
        self._refresh_promoted_candidate = refresh_promoted_candidate
        self._include_squares = include_squares
        self._use_obgd = use_obgd
        self._obgd_kappa = obgd_kappa
        self._scale_robust = scale_robust
        self._scale_normalizer_decay = scale_normalizer_decay
        self._scale_normalizer_epsilon = scale_normalizer_epsilon

    @property
    def n_features(self) -> int:
        """Number of active features."""
        return self._n_features

    @property
    def n_tasks(self) -> int:
        """Number of output tasks."""
        return self._n_tasks

    def memory_accounting(self, state: InteractionFeatureState) -> dict[str, int | bool]:
        """Return exact persistent-array accounting for an initialized state.

        ``persistent_array_bytes`` counts every JAX array in the state,
        including the PRNG key, pair descriptors, archive heads, utilities,
        ages, provenance, and optional scale normalizers.  It intentionally
        excludes Python object overhead, compiled executables, and the two
        host timing floats.  The candidate archive is counted even though its
        features do not contribute directly to predictions.
        """
        scientific_fields = (
            value
            for name, value in vars(state).items()
            if name not in {"birth_timestamp", "uptime_s"}
        )
        array_bytes = sum(
            int(getattr(leaf, "nbytes", 0))
            for value in scientific_fields
            for leaf in jax.tree_util.tree_leaves(value)
        )
        normalizer_scalars = (
            state.feature_second_moments.size
            + state.candidate_second_moments.size
            + state.target_second_moments.size
        )
        return {
            "active_pair_slots": self._n_features,
            "candidate_archive_slots": self._candidate_count,
            "candidate_archive_is_counted": True,
            "active_pair_descriptor_scalars": 2 * self._n_features,
            "candidate_pair_descriptor_scalars": 2 * self._candidate_count,
            "active_output_weight_scalars": self._n_tasks * self._n_features,
            "relevance_probe_weight_scalars": self._n_tasks * self._n_features,
            "relevance_probe_weight_bytes": int(state.relevance_probe_weights.nbytes),
            "relevance_probe_bias_scalars": self._n_tasks,
            "relevance_probe_bias_bytes": int(state.relevance_probe_biases.nbytes),
            "relevance_probe_bytes": int(
                state.relevance_probe_weights.nbytes
                + state.relevance_probe_biases.nbytes
            ),
            "candidate_output_weight_scalars": self._n_tasks * self._candidate_count,
            "candidate_promotion_evidence_streak_scalars": int(
                state.candidate_promotion_evidence_streak.size
            ),
            "candidate_promotion_evidence_streak_bytes": int(
                state.candidate_promotion_evidence_streak.nbytes
            ),
            "candidate_reacquisition_required_scalars": int(
                state.candidate_reacquisition_required.size
            ),
            "candidate_reacquisition_required_bytes": int(
                state.candidate_reacquisition_required.nbytes
            ),
            "evidence_idle_step_scalars": int(state.evidence_idle_steps.size),
            "evidence_idle_step_bytes": int(state.evidence_idle_steps.size) * 4,
            "utility_evidence_streak_scalars": int(state.utility_evidence_streak.size),
            "utility_evidence_streak_bytes": int(state.utility_evidence_streak.size) * 4,
            "active_output_memory_committed_scalars": int(
                state.active_output_memory_committed.size
            ),
            "active_output_memory_committed_bytes": int(
                state.active_output_memory_committed.size
            ),
            "normalizer_scalars": int(normalizer_scalars),
            "normalizer_bytes": int(normalizer_scalars) * 4,
            "persistent_array_bytes": array_bytes,
        }

    def to_config(self) -> dict[str, Any]:
        """Serialize learner configuration."""
        return {
            "type": "FixedBudgetInteractionLearner",
            "n_features": self._n_features,
            "n_tasks": self._n_tasks,
            "step_size_output": self._step_size_output,
            "utility_decay": self._utility_decay,
            "replacement_interval": self._replacement_interval,
            "min_feature_age": self._min_feature_age,
            "candidate_count": self._candidate_count,
            "candidate_min_age": self._candidate_min_age,
            "promotion_margin": self._promotion_margin,
            "promotion_blend": self._promotion_blend,
            "generator_mix": list(self._generator_mix),
            "candidate_strategy": self._candidate_strategy,
            "utility_aggregation": self._utility_aggregation,
            "utility_top_k": self._utility_top_k,
            "utility_task_balancing": self._utility_task_balancing,
            "task_activity_decay": self._task_activity_decay,
            "future_utility_mix": self._future_utility_mix,
            "utility_retention_decay": self._utility_retention_decay,
            "utility_retention_grace_steps": self._utility_retention_grace_steps,
            "utility_evidence_threshold": self._utility_evidence_threshold,
            "evidence_gated_active_output_memory": (
                self._evidence_gated_active_output_memory
            ),
            "utility_evidence_confirmation_steps": (
                self._utility_evidence_confirmation_steps
            ),
            "independent_relevance_probe": self._independent_relevance_probe,
            "relevance_probe_mode": self._relevance_probe_mode,
            "retire_stale_features": self._retire_stale_features,
            "candidate_promotion_floor": self._candidate_promotion_floor,
            "candidate_promotion_confirmation_steps": (
                self._candidate_promotion_confirmation_steps
            ),
            "candidate_reacquisition_confirmation_steps": (
                self._candidate_reacquisition_confirmation_steps
            ),
            "candidate_utility_retention_decay": (self._candidate_utility_retention_decay),
            "refresh_candidates": self._refresh_candidates,
            "refresh_promoted_candidate": self._refresh_promoted_candidate,
            "include_squares": self._include_squares,
            "use_obgd": self._use_obgd,
            "obgd_kappa": self._obgd_kappa,
            "scale_robust": self._scale_robust,
            "scale_normalizer_decay": self._scale_normalizer_decay,
            "scale_normalizer_epsilon": self._scale_normalizer_epsilon,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FixedBudgetInteractionLearner":
        """Reconstruct an explicit, versioned :meth:`to_config` payload.

        Probe-enabled payloads written before ``relevance_probe_mode`` existed
        are intentionally rejected: both target-only and conditional semantics
        used the old unversioned shape, so guessing would silently change an
        experiment.  Probe-disabled legacy payloads migrate deterministically
        to ``conditional_v1`` because the dormant fixed-shape probe has no
        behavioral effect.
        """
        config = dict(config)
        config.pop("type", None)
        if "relevance_probe_mode" not in config:
            if config.get("independent_relevance_probe", False):
                raise ValueError(
                    "probe-enabled legacy config is ambiguous; set "
                    "relevance_probe_mode explicitly"
                )
            config["relevance_probe_mode"] = RELEVANCE_PROBE_MODE_CONDITIONAL_V1
        generator_mix = config.pop("generator_mix", (1.0, 0.0, 0.0))
        return cls(generator_mix=tuple(generator_mix), **config)

    def init(self, feature_dim: int, key: Array) -> InteractionFeatureState:
        """Initialize active and candidate pair banks."""
        if feature_dim < 2 and not self._include_squares:
            raise ValueError("feature_dim must be at least 2 when squares are disabled")

        key, k_active, k_candidate = jr.split(key, 3)
        feature_left, feature_right = self._random_pairs(k_active, self._n_features, feature_dim)
        if self._candidate_strategy == "all_pairs":
            candidate_left, candidate_right = self._candidate_pairs(
                k_candidate, self._candidate_count, feature_dim
            )
        else:
            candidate_left, candidate_right = self._random_pairs(
                k_candidate, self._candidate_count, feature_dim
            )

        normalizer_feature_count = self._n_features if self._scale_robust else 0
        normalizer_candidate_count = self._candidate_count if self._scale_robust else 0
        normalizer_target_count = self._n_tasks if self._scale_robust else 0
        return InteractionFeatureState(
            key=key,
            feature_left=feature_left,
            feature_right=feature_right,
            output_weights=jnp.zeros((self._n_tasks, self._n_features), dtype=jnp.float32),
            relevance_probe_weights=jnp.zeros(
                (self._n_tasks, self._n_features),
                dtype=jnp.float32,
            ),
            relevance_probe_biases=jnp.zeros(self._n_tasks, dtype=jnp.float32),
            output_biases=jnp.zeros(self._n_tasks, dtype=jnp.float32),
            utilities=jnp.zeros(self._n_features, dtype=jnp.float32),
            evidence_idle_steps=jnp.zeros(self._n_features, dtype=jnp.int32),
            utility_evidence_streak=jnp.zeros(self._n_features, dtype=jnp.int32),
            active_output_memory_committed=jnp.zeros(
                self._n_features,
                dtype=jnp.bool_,
            ),
            task_activity_ema=jnp.zeros(self._n_tasks, dtype=jnp.float32),
            ages=jnp.zeros(self._n_features, dtype=jnp.int32),
            candidate_left=candidate_left,
            candidate_right=candidate_right,
            candidate_output_weights=jnp.zeros(
                (self._n_tasks, self._candidate_count), dtype=jnp.float32
            ),
            candidate_utilities=jnp.zeros(self._candidate_count, dtype=jnp.float32),
            candidate_ages=jnp.zeros(self._candidate_count, dtype=jnp.int32),
            candidate_promotion_evidence_streak=jnp.zeros(
                self._candidate_count,
                dtype=jnp.int32,
            ),
            candidate_reacquisition_required=jnp.zeros(
                self._candidate_count,
                dtype=jnp.bool_,
            ),
            feature_second_moments=jnp.zeros(normalizer_feature_count, dtype=jnp.float32),
            candidate_second_moments=jnp.zeros(normalizer_candidate_count, dtype=jnp.float32),
            target_second_moments=jnp.zeros(normalizer_target_count, dtype=jnp.float32),
            feature_parent_a=jnp.full(self._n_features, -1, dtype=jnp.int32),
            feature_parent_b=jnp.full(self._n_features, -1, dtype=jnp.int32),
            feature_generator=jnp.full(self._n_features, GENERATOR_RANDOM, dtype=jnp.int32),
            candidate_parent_a=jnp.full(self._candidate_count, -1, dtype=jnp.int32),
            candidate_parent_b=jnp.full(self._candidate_count, -1, dtype=jnp.int32),
            candidate_generator=jnp.full(self._candidate_count, GENERATOR_RANDOM, dtype=jnp.int32),
            step_count=jnp.array(0, dtype=jnp.int32),
            birth_timestamp=jnp.asarray(time.time(), dtype=jnp.float32),
            uptime_s=jnp.asarray(0.0, dtype=jnp.float32),
        )

    def _all_pairs(self, feature_dim: int) -> tuple[Array, Array]:
        pairs = []
        for i in range(feature_dim):
            start = i if self._include_squares else i + 1
            for j in range(start, feature_dim):
                pairs.append((i, j))
        arr = jnp.array(pairs, dtype=jnp.int32)
        return arr[:, 0], arr[:, 1]

    def _candidate_pairs(
        self,
        key: Array,
        count: int,
        feature_dim: int,
    ) -> tuple[Array, Array]:
        """Generate candidate pairs, optionally covering the whole pair space."""
        if count == 0:
            empty = jnp.zeros((0,), dtype=jnp.int32)
            return empty, empty

        pair_left, pair_right = self._all_pairs(feature_dim)
        n_pairs = pair_left.shape[0]
        if count >= n_pairs:
            repeats = (count + n_pairs - 1) // n_pairs
            left = jnp.tile(pair_left, repeats)[:count]
            right = jnp.tile(pair_right, repeats)[:count]
            return left, right

        perm = jr.permutation(key, n_pairs)[:count]
        return pair_left[perm], pair_right[perm]

    def _random_pairs(
        self,
        key: Array,
        count: int,
        feature_dim: int,
    ) -> tuple[Array, Array]:
        """Generate canonical random pairs."""
        if count == 0:
            empty = jnp.zeros((0,), dtype=jnp.int32)
            return empty, empty

        k_left, k_offset = jr.split(key)
        left = jr.randint(k_left, (count,), 0, feature_dim, dtype=jnp.int32)
        if self._include_squares:
            right = jr.randint(k_offset, (count,), 0, feature_dim, dtype=jnp.int32)
        else:
            offset = jr.randint(k_offset, (count,), 1, feature_dim, dtype=jnp.int32)
            right = (left + offset) % feature_dim
        return self._canonicalize(left, right)

    def _canonicalize(self, left: Array, right: Array) -> tuple[Array, Array]:
        if self._include_squares:
            return left, right
        return jnp.minimum(left, right), jnp.maximum(left, right)

    def _live_descriptor_mask(
        self,
        left: Array,
        right: Array,
        feature_dim: int,
    ) -> Array:
        ordered = left <= right if self._include_squares else left < right
        return (left >= 0) & (right >= 0) & (left < feature_dim) & (right < feature_dim) & ordered

    def _values(self, left: Array, right: Array, observation: Array) -> Array:
        live = self._live_descriptor_mask(left, right, observation.shape[0])
        safe_left = jnp.where(live, left, 0)
        safe_right = jnp.where(live, right, 0)
        return observation[safe_left] * observation[safe_right] * live.astype(jnp.float32)

    def _task_activity_update(
        self,
        old_activity: Array,
        active_mask: Array,
    ) -> Array:
        """Track how often each task is observed for opt-in utility balancing."""
        return self._task_activity_decay * old_activity + (
            1.0 - self._task_activity_decay
        ) * active_mask.astype(jnp.float32)

    def _utility_signal(
        self,
        output_weights: Array,
        features: Array,
        active_mask: Array,
        task_activity_ema: Array,
    ) -> Array:
        """Score feature usefulness from current outgoing weights and activity."""
        weighted_activity = jnp.abs(output_weights) * jnp.abs(features)[None, :]
        if self._utility_task_balancing != "none":
            active = active_mask.astype(jnp.float32)
            if self._utility_task_balancing == "active_inverse_frequency":
                frequency_floor = jnp.array(1.0 - self._task_activity_decay, dtype=jnp.float32)
                task_weights = active / jnp.maximum(task_activity_ema, frequency_floor)
            else:
                task_weights = active
            weighted_activity = weighted_activity * task_weights[:, None]

        if self._utility_aggregation == "max":
            return jnp.max(weighted_activity, axis=0)
        if self._utility_aggregation == "topk":
            k = min(self._utility_top_k, self._n_tasks)
            return jnp.mean(jnp.sort(weighted_activity, axis=0)[-k:, :], axis=0)

        if self._utility_task_balancing == "active":
            active_count = jnp.maximum(jnp.sum(active_mask.astype(jnp.float32)), 1.0)
            return jnp.sum(weighted_activity, axis=0) / active_count
        if self._utility_task_balancing == "active_inverse_frequency":
            active_count = jnp.maximum(jnp.sum(active_mask.astype(jnp.float32)), 1.0)
            return jnp.sum(weighted_activity, axis=0) / active_count
        return jnp.mean(weighted_activity, axis=0)

    def _mixed_utility_signal(
        self,
        old_output_weights: Array,
        new_output_weights: Array,
        features: Array,
        errors: Array,
        active_mask: Array,
        task_activity_ema: Array,
        active_count: Array,
        effective_step_size: Array,
    ) -> Array:
        current_signal = self._utility_signal(
            old_output_weights, features, active_mask, task_activity_ema
        )
        if self._future_utility_mix == 0.0:
            return current_signal
        del new_output_weights
        future_signal = self._future_utility_signal(
            errors,
            features,
            active_mask,
            task_activity_ema,
            active_count,
            effective_step_size,
        )
        return (
            1.0 - self._future_utility_mix
        ) * current_signal + self._future_utility_mix * future_signal

    def _future_utility_signal(
        self,
        errors: Array,
        features: Array,
        active_mask: Array,
        task_activity_ema: Array,
        active_count: Array,
        effective_step_size: Array,
    ) -> Array:
        reductions = one_step_output_loss_reduction(
            errors,
            features,
            active_mask,
            effective_step_size,
            active_count,
        )
        return self._aggregate_task_feature_signal(
            reductions,
            active_mask,
            task_activity_ema,
        )

    def _aggregate_task_feature_signal(
        self,
        task_feature_signal: Array,
        active_mask: Array,
        task_activity_ema: Array,
    ) -> Array:
        weighted_signal = task_feature_signal
        if self._utility_task_balancing != "none":
            active = active_mask.astype(jnp.float32)
            if self._utility_task_balancing == "active_inverse_frequency":
                frequency_floor = jnp.array(1.0 - self._task_activity_decay, dtype=jnp.float32)
                task_weights = active / jnp.maximum(task_activity_ema, frequency_floor)
            else:
                task_weights = active
            weighted_signal = weighted_signal * task_weights[:, None]

        if self._utility_aggregation == "max":
            return jnp.max(weighted_signal, axis=0)
        if self._utility_aggregation == "topk":
            k = min(self._utility_top_k, self._n_tasks)
            return jnp.mean(jnp.sort(weighted_signal, axis=0)[-k:, :], axis=0)
        if self._utility_task_balancing in {"active", "active_inverse_frequency"}:
            active_count = jnp.maximum(jnp.sum(active_mask.astype(jnp.float32)), 1.0)
            return jnp.sum(weighted_signal, axis=0) / active_count
        return jnp.mean(weighted_signal, axis=0)

    def _second_moment_update(self, old_moment: Array, values: Array) -> Array:
        """Update a causal scale estimate used by the opt-in robust path."""
        return (
            self._scale_normalizer_decay * old_moment
            + (1.0 - self._scale_normalizer_decay) * values**2
        )

    def _normalized_output_deltas(
        self,
        errors: Array,
        features: Array,
        second_moments: Array,
        active_count: Array,
    ) -> tuple[Array, Array]:
        """Diagonally preconditioned NLMS for the deployed linear head.

        The diagonal contains the causal feature second moment, the current
        squared value, and the current bank's mean energy.  The shared energy
        term is a scale-equivariant ridge: a near-zero Gaussian product cannot
        receive an enormous ``error / feature`` weight merely because it was
        small on one sample.  Dividing by ``1 + phi.T D^-1 phi`` (the one is
        the bias coordinate) bounds the aggregate current-sample prediction
        change by ``alpha * abs(error)``.
        """
        bank_energy = jnp.mean(features**2)
        denominator = jnp.maximum(
            jnp.maximum(
                jnp.maximum(second_moments, features**2),
                bank_energy,
            ),
            self._scale_normalizer_epsilon,
        )
        leverage = 1.0 + jnp.sum(features**2 / denominator)
        normalization = jnp.maximum(leverage, 1.0)
        weight_delta = (
            self._step_size_output
            * errors[:, None]
            * features[None, :]
            / denominator[None, :]
            / normalization
            / active_count
        )
        bias_delta = self._step_size_output * errors / normalization / active_count
        return weight_delta, bias_delta

    def _normalized_candidate_delta(
        self,
        candidate_errors: Array,
        features: Array,
        second_moments: Array,
        active_count: Array,
    ) -> Array:
        """Independent normalized residual-probe updates for dormant candidates.

        Candidate heads are not summed into the deployed prediction, so each
        is a separate one-coordinate probe rather than one 91-coordinate
        model.  Its error is the deployed residual minus its own proposed
        contribution; this makes the head a stable residual predictor rather
        than an unbounded accumulator of deployed gradients.  The common
        bank-energy ridge still prevents a near-zero product from taking an
        unbounded parameter step.
        """
        bank_energy = jnp.mean(features**2)
        denominator = jnp.maximum(
            jnp.maximum(
                jnp.maximum(second_moments, features**2),
                bank_energy,
            ),
            self._scale_normalizer_epsilon,
        )
        return (
            self._step_size_output
            * candidate_errors
            * features[None, :]
            / denominator[None, :]
            / active_count
        )

    def _relevance_probe_update(
        self,
        probe_weights: Array,
        features: Array,
        conditional_errors: Array,
        safe_targets: Array,
        active_mask: Array,
        task_activity_ema: Array,
        target_second_moments: Array,
        feature_second_moments: Array,
        active_count: Array,
    ) -> tuple[Array, Array, Array]:
        """Score and update independent one-coordinate relevance probes.

        Under ``conditional_v1``, ``conditional_errors[:, i]`` is the deployed
        durable-bank residual after removing readable coordinate ``i``. Under
        ``target_only_v1``, every column is the same target-minus-probe-bias
        baseline. Probe heads never read one another in either mode.
        """
        relevance_errors = jnp.where(
            active_mask[:, None],
            conditional_errors,
            0.0,
        )
        probe_contributions = probe_weights * features[None, :]

        if self._scale_robust:
            target_scale = jnp.sqrt(
                jnp.maximum(target_second_moments, self._scale_normalizer_epsilon)
            )
            denominator = jnp.maximum(
                jnp.maximum(
                    jnp.maximum(feature_second_moments, features**2),
                    jnp.mean(features**2),
                ),
                self._scale_normalizer_epsilon,
            )
        else:
            target_scale = jnp.maximum(
                jnp.abs(safe_targets),
                jnp.asarray(1.0, dtype=jnp.float32),
            )
            denominator = jnp.maximum(
                jnp.maximum(features**2, jnp.mean(features**2)),
                self._scale_normalizer_epsilon,
            )

        normalized_error = relevance_errors / target_scale[:, None]
        normalized_probe = probe_contributions / target_scale[:, None]
        clip = jnp.asarray(1e6, dtype=jnp.float32)
        normalized_error = jnp.clip(
            jnp.nan_to_num(normalized_error, nan=0.0, posinf=1e6, neginf=-1e6),
            -clip,
            clip,
        )
        normalized_probe = jnp.clip(
            jnp.nan_to_num(normalized_probe, nan=0.0, posinf=1e6, neginf=-1e6),
            -clip,
            clip,
        )
        insertion_gain = jnp.maximum(
            normalized_error * normalized_probe - 0.5 * normalized_probe**2,
            0.0,
        )
        bounded_gain = insertion_gain / (1.0 + insertion_gain)
        bounded_gain = jnp.where(active_mask[:, None], bounded_gain, 0.0)
        scores = self._aggregate_task_feature_signal(
            bounded_gain,
            active_mask,
            task_activity_ema,
        )

        probe_errors = relevance_errors - probe_contributions
        probe_delta = (
            self._step_size_output
            * probe_errors
            * features[None, :]
            / denominator[None, :]
            / active_count
        )
        updated_probe = probe_weights + probe_delta
        return scores, relevance_errors, updated_probe

    def _marginal_candidate_signal(
        self,
        candidate_weights: Array,
        candidate_features: Array,
        relevance_baseline_errors: Array,
        safe_targets: Array,
        active_mask: Array,
        task_activity_ema: Array,
        target_second_moments: Array,
    ) -> Array:
        """Return bounded conditional insertion evidence for candidate probes.

        ``relevance_baseline_errors`` is the residual of the complete deployed
        durable bank.  Candidate heads are private probes, so the calculation
        remains fixed-shape and independent while measuring genuine
        incremental value beyond the currently readable representation.
        """
        candidate_contributions = candidate_weights * candidate_features[None, :]
        if self._scale_robust:
            target_scale = jnp.sqrt(
                jnp.maximum(target_second_moments, self._scale_normalizer_epsilon)
            )
        else:
            target_scale = jnp.maximum(
                jnp.abs(safe_targets),
                jnp.asarray(1.0, dtype=jnp.float32),
            )
        normalized_error = relevance_baseline_errors[:, None] / target_scale[:, None]
        normalized_contribution = candidate_contributions / target_scale[:, None]
        clip = jnp.asarray(1e6, dtype=jnp.float32)
        normalized_error = jnp.clip(
            jnp.nan_to_num(normalized_error, nan=0.0, posinf=1e6, neginf=-1e6),
            -clip,
            clip,
        )
        normalized_contribution = jnp.clip(
            jnp.nan_to_num(
                normalized_contribution,
                nan=0.0,
                posinf=1e6,
                neginf=-1e6,
            ),
            -clip,
            clip,
        )
        insertion_gain = jnp.maximum(
            normalized_error * normalized_contribution
            - 0.5 * normalized_contribution**2,
            0.0,
        )
        bounded_gain = insertion_gain / (1.0 + insertion_gain)
        bounded_gain = jnp.where(active_mask[:, None], bounded_gain, 0.0)
        return self._aggregate_task_feature_signal(
            bounded_gain,
            active_mask,
            task_activity_ema,
        )

    def _loss_proxy_utility_signal(
        self,
        output_weights: Array,
        features: Array,
        errors: Array,
        active_mask: Array,
        task_activity_ema: Array,
        target_second_moments: Array,
        *,
        deployed: bool,
    ) -> Array:
        """Bounded, scale-free signed utility from one-feature interventions.

        For a deployed feature with contribution ``q = w * phi``, removing it
        changes the residual from ``e`` to ``e + q``.  Its utility is the
        positive part of that deletion's loss increase.  A dormant candidate
        instead uses the positive loss reduction from changing ``e`` to
        ``e - q``.  Both are divided by the causal target second moment and
        squashed into ``[0, 1)`` before the utility EMA.

        This rejects a large wrong-sign shock contribution, unlike
        ``abs(w * phi)``.  It is still a one-coordinate proxy: correlated or
        redundant features can receive misleading deletion credit.
        """
        target_scale = jnp.sqrt(jnp.maximum(target_second_moments, self._scale_normalizer_epsilon))
        normalized_error = errors / target_scale
        normalized_contribution = output_weights * features[None, :] / target_scale[:, None]

        # The robust update keeps these values moderate from initialization,
        # but finite clipping makes the utility path fail-safe if handed a
        # manually corrupted or legacy state.
        clip = jnp.array(1e6, dtype=jnp.float32)
        normalized_error = jnp.clip(
            jnp.nan_to_num(normalized_error, nan=0.0, posinf=1e6, neginf=-1e6),
            -clip,
            clip,
        )
        normalized_contribution = jnp.clip(
            jnp.nan_to_num(
                normalized_contribution,
                nan=0.0,
                posinf=1e6,
                neginf=-1e6,
            ),
            -clip,
            clip,
        )

        signed_cross = normalized_error[:, None] * normalized_contribution
        contribution_energy = 0.5 * normalized_contribution**2
        if deployed:
            normalized_loss_change = signed_cross + contribution_energy
        else:
            normalized_loss_change = signed_cross - contribution_energy
        positive = jnp.maximum(normalized_loss_change, 0.0)
        bounded = positive / (1.0 + positive)
        bounded = jnp.where(active_mask[:, None], bounded, 0.0)
        return self._aggregate_task_feature_signal(
            bounded,
            active_mask,
            task_activity_ema,
        )

    def _utility_update_with_retention(
        self,
        old_utilities: Array,
        utility_signal: Array,
        retention_decay: float | None,
    ) -> Array:
        """Update utility with an optional recurrent-context retention floor."""
        ema = self._utility_decay * old_utilities + (1.0 - self._utility_decay) * utility_signal
        if retention_decay is None:
            return ema
        retained = retention_decay * old_utilities
        return jnp.maximum(ema, retained)

    def _utility_update(self, old_utilities: Array, utility_signal: Array) -> Array:
        """Update deployed-feature utility with optional slower retention."""
        return self._utility_update_with_retention(
            old_utilities,
            utility_signal,
            self._utility_retention_decay,
        )

    def _leased_utility_update(
        self,
        old_utilities: Array,
        utility_signal: Array,
        old_idle_steps: Array,
        live: Array,
        evidence_refreshed: Array,
        retention_evidence_refreshed: Array,
    ) -> tuple[Array, Array, Array]:
        """Apply optional evidence-bounded retention and update idle counters.

        Without a configured grace period this is exactly the historical
        retention update and the fixed-shape idle counters remain zero.  With
        a lease, sufficiently strong instantaneous utility evidence refreshes
        a live slot.  The slower retention floor applies only through the grace
        horizon; expired slots resume the ordinary utility EMA and can be
        retired independently at the next curation cadence.
        """
        if self._utility_retention_grace_steps is None:
            utilities = self._utility_update(old_utilities, utility_signal)
            return (
                jnp.where(live, utilities, 0.0),
                jnp.zeros_like(old_idle_steps),
                jnp.zeros_like(live),
            )

        grace = self._utility_retention_grace_steps
        evidence_refreshed = live & jnp.asarray(evidence_refreshed, dtype=jnp.bool_)
        refresh_lease = live & jnp.asarray(retention_evidence_refreshed, dtype=jnp.bool_)
        idle_cap = jnp.asarray(2**31 - 1, dtype=jnp.int32)
        incremented = (
            jnp.minimum(jnp.maximum(old_idle_steps, 0), idle_cap - 1) + 1
        )
        idle_steps = jnp.where(
            live,
            jnp.where(refresh_lease, 0, incremented),
            0,
        )
        ema = self._utility_decay * old_utilities + (1.0 - self._utility_decay) * utility_signal
        if self._utility_retention_decay is None:
            utilities = ema
        else:
            retained = self._utility_retention_decay * old_utilities
            protected = live & (idle_steps <= grace)
            utilities = jnp.where(protected, jnp.maximum(ema, retained), ema)
        return jnp.where(live, utilities, 0.0), idle_steps, evidence_refreshed

    def _confirmed_utility_evidence(
        self,
        old_streak: Array,
        raw_evidence: Array,
        live: Array,
    ) -> tuple[Array, Array]:
        """Return fixed-shape consecutive evidence state and lease/write confirmation."""
        if not self._evidence_gated_active_output_memory:
            return jnp.zeros_like(old_streak), raw_evidence

        evidence = live & jnp.asarray(raw_evidence, dtype=jnp.bool_)
        int32_max = jnp.asarray(2**31 - 1, dtype=jnp.int32)
        incremented = jnp.minimum(jnp.maximum(old_streak, 0), int32_max - 1) + 1
        streak = jnp.where(evidence, incremented, 0)
        confirmed = evidence & (
            streak
            >= jnp.asarray(
                self._utility_evidence_confirmation_steps,
                dtype=jnp.int32,
            )
        )
        return streak, confirmed

    def _generate_one(
        self,
        key: Array,
        observation: Array,
        active_left: Array,
        active_right: Array,
        utilities: Array,
    ) -> tuple[Array, Array, Array, Array, Array]:
        """Generate one candidate pair using random, mutation, or imprint."""
        feature_dim = observation.shape[0]
        key_kind, key_random, key_parent, key_mutate = jr.split(key, 4)
        mix = jnp.array(self._generator_mix, dtype=jnp.float32)
        generator = jr.categorical(key_kind, jnp.log(mix + 1e-8))
        live_parent = self._live_descriptor_mask(
            active_left,
            active_right,
            feature_dim,
        )
        has_live_parent = jnp.any(live_parent)
        generator = jnp.where(
            has_live_parent,
            generator,
            jnp.asarray(GENERATOR_RANDOM, dtype=jnp.int32),
        )

        random_left, random_right = self._random_pairs(key_random, 1, feature_dim)
        random_left = random_left[0]
        random_right = random_right[0]

        parent_logits = jnp.where(
            live_parent,
            jnp.log(jnp.maximum(utilities, 0.0) + 1e-3),
            -jnp.inf,
        )
        parent_idx = jr.categorical(key_parent, parent_logits).astype(jnp.int32)
        safe_parent_idx = jnp.where(has_live_parent, parent_idx, 0)
        parent_left = jnp.where(
            has_live_parent,
            active_left[safe_parent_idx],
            random_left,
        )
        parent_right = jnp.where(
            has_live_parent,
            active_right[safe_parent_idx],
            random_right,
        )
        key_dim, key_side = jr.split(key_mutate)
        mutate_dim = jr.randint(key_dim, (), 0, feature_dim, dtype=jnp.int32)
        mutate_left_side = jr.bernoulli(key_side)
        mutate_left = jnp.where(mutate_left_side, mutate_dim, parent_left)
        mutate_right = jnp.where(mutate_left_side, parent_right, mutate_dim)
        if not self._include_squares:
            mutate_right = jnp.where(
                mutate_left == mutate_right,
                (mutate_right + 1) % feature_dim,
                mutate_right,
            )
        mutate_left, mutate_right = self._canonicalize(mutate_left, mutate_right)

        top_two = jnp.argsort(jnp.abs(observation))[-2:]
        imprint_left, imprint_right = self._canonicalize(top_two[0], top_two[1])

        def random_branch() -> tuple[Array, Array, Array, Array, Array]:
            return (
                random_left,
                random_right,
                jnp.array(-1, dtype=jnp.int32),
                jnp.array(-1, dtype=jnp.int32),
                jnp.array(GENERATOR_RANDOM, dtype=jnp.int32),
            )

        def mutate_branch() -> tuple[Array, Array, Array, Array, Array]:
            return (
                mutate_left,
                mutate_right,
                safe_parent_idx,
                jnp.array(-1, dtype=jnp.int32),
                jnp.array(GENERATOR_MUTATE_PARENT, dtype=jnp.int32),
            )

        def imprint_branch() -> tuple[Array, Array, Array, Array, Array]:
            return (
                imprint_left,
                imprint_right,
                jnp.array(-1, dtype=jnp.int32),
                jnp.array(-1, dtype=jnp.int32),
                jnp.array(GENERATOR_IMPRINT, dtype=jnp.int32),
            )

        return jax.lax.switch(generator, (random_branch, mutate_branch, imprint_branch))

    @functools.partial(jax.jit, static_argnums=(0,))
    def constructed_features(
        self,
        state: InteractionFeatureState,
        observation: Array,
    ) -> Array:
        """Return active constructed pair-product features for ``observation``.

        These are literal Step 2 features made by combining existing features.
        Downstream GVF/Horde learners can consume them as a fixed representation
        or concatenate them with raw observations.
        """
        return self._values(state.feature_left, state.feature_right, observation)

    @functools.partial(jax.jit, static_argnums=(0,))
    def augmented_observation(
        self,
        state: InteractionFeatureState,
        observation: Array,
    ) -> Array:
        """Concatenate raw observation with active pair-product features."""
        return jnp.concatenate([observation, self.constructed_features(state, observation)])

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(
        self,
        state: InteractionFeatureState,
        observation: Array,
        external_read_mask: Array | None = None,
    ) -> Array:
        """Predict all tasks from active interaction features."""
        features = self.constructed_features(state, observation)
        if not self._independent_relevance_probe:
            return state.output_weights @ features + state.output_biases
        external = (
            jnp.ones((self._n_features,), dtype=jnp.bool_)
            if external_read_mask is None
            else jnp.asarray(external_read_mask, dtype=jnp.bool_).reshape(
                (self._n_features,)
            )
        )
        live = self._live_descriptor_mask(
            state.feature_left,
            state.feature_right,
            observation.shape[0],
        )
        readable = live & state.active_output_memory_committed & external
        readable_weights = jnp.where(
            readable[None, :],
            state.output_weights,
            0.0,
        )
        readable_features = jnp.where(readable, features, 0.0)
        return readable_weights @ readable_features + state.output_biases

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: InteractionFeatureState,
        observation: Array,
        targets: Array,
        external_read_mask: Array | None = None,
        curation_priority_override: InteractionCurationPriorityOverride | None = None,
    ) -> InteractionFeatureUpdateResult:
        """Perform one temporally-uniform interaction-feature update.

        Shapes and effective JAX dtypes are checked while tracing.  NaN targets
        remain the documented missing-head sentinel.  A nonfinite observation
        or infinite target is dynamically rejected: the returned state is the
        exact input state, lifecycle events are neutral, and
        ``update_rejected`` is true.  This is an atomic no-op under eager use,
        :func:`jax.jit`, and :func:`jax.lax.scan`.
        """
        raw_observation = _require_array_contract(
            observation,
            name="observation",
            shape=None,
            dtype=jnp.dtype(jnp.float32),
        )
        raw_targets = _require_array_contract(
            targets,
            name="targets",
            shape=(self._n_tasks,),
            dtype=jnp.dtype(jnp.float32),
        )
        external = (
            jnp.ones((self._n_features,), dtype=jnp.bool_)
            if external_read_mask is None
            else _require_array_contract(
                external_read_mask,
                name="external_read_mask",
                shape=(self._n_features,),
                dtype=jnp.dtype(jnp.bool_),
            )
        )
        if curation_priority_override is None:
            priority_override = InteractionCurationPriorityOverride(
                enabled=jnp.asarray(False, dtype=jnp.bool_),
                active_ranks=jnp.zeros((self._n_features,), dtype=jnp.float32),
                candidate_ranks=jnp.zeros(
                    (self._candidate_count,),
                    dtype=jnp.float32,
                ),
            )
        elif isinstance(
            curation_priority_override,
            InteractionCurationPriorityOverride,
        ):
            priority_override = curation_priority_override
        else:
            raise TypeError(
                "curation_priority_override must be an "
                "InteractionCurationPriorityOverride"
            )
        priority_override_enabled = _require_array_contract(
            priority_override.enabled,
            name="curation_priority_override.enabled",
            shape=(),
            dtype=jnp.dtype(jnp.bool_),
        )
        raw_active_priority_ranks = _require_array_contract(
            priority_override.active_ranks,
            name="curation_priority_override.active_ranks",
            shape=(self._n_features,),
            dtype=jnp.dtype(jnp.float32),
        )
        raw_candidate_priority_ranks = _require_array_contract(
            priority_override.candidate_ranks,
            name="curation_priority_override.candidate_ranks",
            shape=(self._candidate_count,),
            dtype=jnp.dtype(jnp.float32),
        )
        priority_override_valid = jnp.all(
            jnp.isfinite(raw_active_priority_ranks)
        ) & jnp.all(jnp.isfinite(raw_candidate_priority_ranks))
        active_priority_ranks = jnp.where(
            jnp.isfinite(raw_active_priority_ranks),
            raw_active_priority_ranks,
            0.0,
        )
        candidate_priority_ranks = jnp.where(
            jnp.isfinite(raw_candidate_priority_ranks),
            raw_candidate_priority_ranks,
            0.0,
        )
        input_valid = (
            jnp.all(jnp.isfinite(raw_observation))
            & jnp.all(jnp.isfinite(raw_targets) | jnp.isnan(raw_targets))
            & priority_override_valid
        )
        observation = jnp.where(jnp.isfinite(raw_observation), raw_observation, 0.0)
        active_mask = jnp.isfinite(raw_targets)
        safe_targets = jnp.where(active_mask, raw_targets, 0.0)
        active_count = jnp.maximum(jnp.sum(active_mask.astype(jnp.float32)), 1.0)
        task_activity_ema = self._task_activity_update(state.task_activity_ema, active_mask)

        feature_dim = observation.shape[0]
        live_features = self._live_descriptor_mask(
            state.feature_left,
            state.feature_right,
            feature_dim,
        )
        features = self._values(state.feature_left, state.feature_right, observation)
        durable_read_mask = (
            live_features & state.active_output_memory_committed & external
            if self._independent_relevance_probe
            else live_features
        )
        readable_output_weights = (
            jnp.where(
                durable_read_mask[None, :],
                state.output_weights,
                0.0,
            )
            if self._independent_relevance_probe
            else state.output_weights
        )
        readable_features = (
            jnp.where(durable_read_mask, features, 0.0)
            if self._independent_relevance_probe
            else features
        )
        predictions = readable_output_weights @ readable_features + state.output_biases
        input_valid = (
            input_valid
            & jnp.all(jnp.isfinite(readable_output_weights))
            & jnp.all(jnp.isfinite(predictions))
        )
        errors = jnp.where(active_mask, safe_targets - predictions, 0.0)
        reported_errors = jnp.where(active_mask, errors, jnp.nan)
        deployed_contributions = readable_output_weights * readable_features[None, :]
        conditional_relevance_errors = jnp.where(
            active_mask[:, None],
            errors[:, None] + deployed_contributions,
            0.0,
        )
        candidate_learning_errors = errors
        if (
            self._independent_relevance_probe
            and self._relevance_probe_mode == RELEVANCE_PROBE_MODE_TARGET_ONLY_V1
        ):
            target_only_errors = jnp.where(
                active_mask,
                safe_targets - state.relevance_probe_biases,
                0.0,
            )
            conditional_relevance_errors = jnp.broadcast_to(
                target_only_errors[:, None],
                (self._n_tasks, self._n_features),
            )
            candidate_learning_errors = target_only_errors
            input_valid = input_valid & jnp.all(
                jnp.isfinite(state.relevance_probe_biases)
            )

        candidate_features = jnp.zeros(self._candidate_count, dtype=jnp.float32)
        if self._candidate_count > 0:
            candidate_features = self._values(
                state.candidate_left, state.candidate_right, observation
            )

        feature_second_moments = state.feature_second_moments
        candidate_second_moments = state.candidate_second_moments
        target_second_moments = state.target_second_moments
        if self._scale_robust:
            feature_second_moments = self._second_moment_update(feature_second_moments, features)
            candidate_second_moments = self._second_moment_update(
                candidate_second_moments, candidate_features
            )
            updated_target_moments = self._second_moment_update(target_second_moments, safe_targets)
            target_second_moments = jnp.where(
                active_mask,
                updated_target_moments,
                target_second_moments,
            )
            utility_target_moments = jnp.maximum(
                target_second_moments,
                safe_targets**2,
            )
            output_delta, output_bias_delta = self._normalized_output_deltas(
                errors,
                features,
                feature_second_moments,
                active_count,
            )
            candidate_output_delta = jnp.zeros_like(state.candidate_output_weights)
            if self._candidate_count > 0:
                candidate_errors = jnp.where(
                    active_mask[:, None],
                    candidate_learning_errors[:, None]
                    - (state.candidate_output_weights * candidate_features[None, :]),
                    0.0,
                )
                candidate_output_delta = self._normalized_candidate_delta(
                    candidate_errors,
                    candidate_features,
                    candidate_second_moments,
                    active_count,
                )
        else:
            utility_target_moments = target_second_moments
            output_delta = (
                self._step_size_output * errors[:, None] * features[None, :] / active_count
            )
            output_bias_delta = self._step_size_output * errors / active_count
            candidate_output_delta = jnp.zeros_like(state.candidate_output_weights)
            if self._candidate_count > 0:
                if self._independent_relevance_probe:
                    candidate_errors = jnp.where(
                        active_mask[:, None],
                        candidate_learning_errors[:, None]
                        - (
                            state.candidate_output_weights
                            * candidate_features[None, :]
                        ),
                        0.0,
                    )
                    candidate_output_delta = (
                        self._step_size_output
                        * candidate_errors
                        * candidate_features[None, :]
                        / active_count
                    )
                else:
                    candidate_output_delta = (
                        self._step_size_output
                        * errors[:, None]
                        * candidate_features[None, :]
                        / active_count
                    )

        if self._candidate_count == 0:
            candidate_output_delta = jnp.zeros_like(state.candidate_output_weights)

        bounding_scale = jnp.array(1.0, dtype=jnp.float32)
        if self._use_obgd and not self._scale_robust:
            total_step = (
                jnp.sum(jnp.abs(output_delta))
                + jnp.sum(jnp.abs(output_bias_delta))
            )
            if not self._independent_relevance_probe:
                total_step = total_step + jnp.sum(jnp.abs(candidate_output_delta))
            err_norm = jnp.linalg.norm(errors)
            bound_magnitude = self._obgd_kappa * jnp.maximum(err_norm, 1.0) * total_step
            bounding_scale = 1.0 / jnp.maximum(bound_magnitude, 1.0)
            output_delta = bounding_scale * output_delta
            output_bias_delta = bounding_scale * output_bias_delta
            if not self._independent_relevance_probe:
                candidate_output_delta = bounding_scale * candidate_output_delta

        proposed_output_weights = state.output_weights + output_delta
        output_biases = state.output_biases + output_bias_delta
        candidate_output_weights = state.candidate_output_weights + candidate_output_delta
        relevance_probe_scores = jnp.zeros(
            (self._n_features,),
            dtype=jnp.float32,
        )
        relevance_probe_errors = jnp.zeros(
            (self._n_tasks, self._n_features),
            dtype=jnp.float32,
        )
        relevance_probe_weights = state.relevance_probe_weights
        # Under conditional_v1 this fixed-shape field mirrors the deployed
        # output bias exactly, preserving that mode's original recurrence.
        # target_only_v1 instead owns and learns an independent target baseline.
        relevance_probe_biases = output_biases
        if (
            self._independent_relevance_probe
            and self._relevance_probe_mode == RELEVANCE_PROBE_MODE_TARGET_ONLY_V1
        ):
            relevance_probe_biases = state.relevance_probe_biases + (
                self._step_size_output * candidate_learning_errors / active_count
            )
        if self._independent_relevance_probe:
            (
                relevance_probe_scores,
                relevance_probe_errors,
                relevance_probe_weights,
            ) = self._relevance_probe_update(
                state.relevance_probe_weights,
                features,
                conditional_relevance_errors,
                safe_targets,
                active_mask,
                task_activity_ema,
                utility_target_moments,
                feature_second_moments,
                active_count,
            )
            utility_signal = relevance_probe_scores
        elif self._scale_robust:
            utility_signal = self._loss_proxy_utility_signal(
                state.output_weights,
                features,
                errors,
                active_mask,
                task_activity_ema,
                utility_target_moments,
                deployed=True,
            )
        else:
            utility_signal = self._mixed_utility_signal(
                state.output_weights,
                proposed_output_weights,
                features,
                errors,
                active_mask,
                task_activity_ema,
                active_count,
                self._step_size_output * bounding_scale,
            )
        if self._utility_retention_grace_steps is None:
            raw_evidence_refreshed = jnp.zeros_like(live_features)
        else:
            raw_evidence_refreshed = live_features & (
                utility_signal
                >= jnp.asarray(
                    self._utility_evidence_threshold,
                    dtype=jnp.float32,
                )
            )
        (
            utility_evidence_streak,
            retention_evidence_refreshed,
        ) = self._confirmed_utility_evidence(
            state.utility_evidence_streak,
            raw_evidence_refreshed,
            live_features,
        )
        if self._independent_relevance_probe:
            first_commit = (
                live_features
                & ~state.active_output_memory_committed
                & retention_evidence_refreshed
            )
            output_weights = jnp.where(
                state.active_output_memory_committed[None, :],
                state.output_weights,
                jnp.where(
                    first_commit[None, :],
                    state.relevance_probe_weights,
                    jnp.zeros_like(state.output_weights),
                ),
            )
            active_output_memory_committed = live_features & (
                state.active_output_memory_committed | first_commit
            )
        elif self._evidence_gated_active_output_memory:
            output_write_gate = (
                ~state.active_output_memory_committed
                | retention_evidence_refreshed
            ) & live_features
            output_weights = jnp.where(
                output_write_gate[None, :],
                proposed_output_weights,
                state.output_weights,
            )
            active_output_memory_committed = live_features & (
                state.active_output_memory_committed
                | retention_evidence_refreshed
            )
        else:
            output_weights = proposed_output_weights
            active_output_memory_committed = jnp.zeros_like(
                state.active_output_memory_committed
            )
        (
            new_utilities,
            evidence_idle_steps,
            evidence_refreshed,
        ) = self._leased_utility_update(
            state.utilities,
            utility_signal,
            state.evidence_idle_steps,
            live_features,
            raw_evidence_refreshed,
            retention_evidence_refreshed,
        )
        candidate_promotion_signal = jnp.zeros(
            (self._candidate_count,),
            dtype=jnp.float32,
        )
        if self._candidate_count > 0:
            if self._independent_relevance_probe:
                candidate_signal = self._marginal_candidate_signal(
                    state.candidate_output_weights,
                    candidate_features,
                    candidate_learning_errors,
                    safe_targets,
                    active_mask,
                    task_activity_ema,
                    utility_target_moments,
                )
            elif self._scale_robust:
                candidate_signal = self._loss_proxy_utility_signal(
                    state.candidate_output_weights,
                    candidate_features,
                    candidate_learning_errors,
                    active_mask,
                    task_activity_ema,
                    utility_target_moments,
                    deployed=False,
                )
            else:
                candidate_signal = self._mixed_utility_signal(
                    state.candidate_output_weights,
                    candidate_output_weights,
                    candidate_features,
                    candidate_learning_errors,
                    active_mask,
                    task_activity_ema,
                    active_count,
                    self._step_size_output
                    * (
                        jnp.asarray(1.0, dtype=jnp.float32)
                        if self._independent_relevance_probe
                        else bounding_scale
                    ),
                )
            candidate_promotion_signal = candidate_signal
            # Candidate scores are testing evidence, not deployed memory. By
            # default they decay normally so a chance correlation from an old
            # context cannot become an immortal promotion candidate.
            new_candidate_utilities = self._utility_update_with_retention(
                state.candidate_utilities,
                candidate_signal,
                self._candidate_utility_retention_decay,
            )
        else:
            new_candidate_utilities = state.candidate_utilities
        candidate_live = self._live_descriptor_mask(
            state.candidate_left,
            state.candidate_right,
            feature_dim,
        )
        candidate_promotion_raw_evidence = jnp.zeros_like(candidate_live)
        if self._independent_relevance_probe:
            candidate_promotion_raw_evidence = (
                candidate_live
                & jnp.isfinite(candidate_promotion_signal)
                & (candidate_promotion_signal > 0.0)
                & (
                    candidate_promotion_signal
                    >= jnp.asarray(
                        self._utility_evidence_threshold,
                        dtype=jnp.float32,
                    )
                )
            )
        candidate_promotion_evidence_streak_pre = (
            state.candidate_promotion_evidence_streak
        )
        candidate_reacquisition_required_pre = (
            state.candidate_reacquisition_required
        )
        candidate_reacquisition_required = (
            candidate_reacquisition_required_pre
            & candidate_live
            & jnp.asarray(
                self._candidate_reacquisition_confirmation_steps > 1,
                dtype=jnp.bool_,
            )
        )
        candidate_promotion_confirmation_cap = jnp.asarray(
            self._candidate_promotion_confirmation_steps,
            dtype=jnp.int32,
        )
        candidate_reacquisition_confirmation_cap = jnp.asarray(
            max(
                self._candidate_promotion_confirmation_steps,
                self._candidate_reacquisition_confirmation_steps,
            ),
            dtype=jnp.int32,
        )
        candidate_required_confirmation_steps = jnp.where(
            candidate_reacquisition_required,
            candidate_reacquisition_confirmation_cap,
            candidate_promotion_confirmation_cap,
        )
        candidate_confirmation_gate_active = (
            self._independent_relevance_probe
            & (candidate_required_confirmation_steps > 1)
        )
        candidate_incremented_streak = (
            jnp.minimum(
                jnp.maximum(candidate_promotion_evidence_streak_pre, 0),
                candidate_required_confirmation_steps - 1,
            )
            + 1
        )
        candidate_promotion_evidence_streak_updated = jnp.where(
            candidate_promotion_raw_evidence & candidate_confirmation_gate_active,
            candidate_incremented_streak,
            0,
        )
        candidate_promotion_confirmed = candidate_live & (
            ~candidate_confirmation_gate_active
            | (
                candidate_promotion_evidence_streak_updated
                >= candidate_required_confirmation_steps
            )
        )
        candidate_reacquisition_confirmed = (
            candidate_live
            & candidate_reacquisition_required
            & (
                candidate_promotion_evidence_streak_updated
                >= candidate_required_confirmation_steps
            )
        )
        candidate_promotion_evidence_streak = (
            candidate_promotion_evidence_streak_updated
        )
        ages = jnp.where(
            live_features,
            _saturating_int32_increment(state.ages),
            0,
        )
        candidate_ages = _saturating_int32_increment(state.candidate_ages)
        step_count = _saturating_int32_increment(state.step_count)
        key, replacement_key = jr.split(state.key)

        feature_left = state.feature_left
        feature_right = state.feature_right
        candidate_left = state.candidate_left
        candidate_right = state.candidate_right
        feature_parent_a = state.feature_parent_a
        feature_parent_b = state.feature_parent_b
        feature_generator = state.feature_generator
        candidate_parent_a = state.candidate_parent_a
        candidate_parent_b = state.candidate_parent_b
        candidate_generator = state.candidate_generator

        # Exact matched-control boundary.  This snapshot includes every
        # ordinary online-learning and recurrence update above, but precedes
        # retirement, promotion, replacement, candidate refresh, and all
        # resets coupled to those descriptor transactions.  In particular,
        # identities and lineage are copied from the entry state while the RNG
        # key, counters, ages, moments, evidence, and learned heads advance.
        pre_curation_state = InteractionFeatureState(
            key=key,
            feature_left=state.feature_left,
            feature_right=state.feature_right,
            output_weights=output_weights,
            relevance_probe_weights=relevance_probe_weights,
            relevance_probe_biases=relevance_probe_biases,
            output_biases=output_biases,
            utilities=new_utilities,
            evidence_idle_steps=evidence_idle_steps,
            utility_evidence_streak=utility_evidence_streak,
            active_output_memory_committed=active_output_memory_committed,
            task_activity_ema=task_activity_ema,
            ages=ages,
            candidate_left=state.candidate_left,
            candidate_right=state.candidate_right,
            candidate_output_weights=candidate_output_weights,
            candidate_utilities=new_candidate_utilities,
            candidate_ages=candidate_ages,
            candidate_promotion_evidence_streak=(
                candidate_promotion_evidence_streak
            ),
            candidate_reacquisition_required=candidate_reacquisition_required,
            feature_second_moments=feature_second_moments,
            candidate_second_moments=candidate_second_moments,
            target_second_moments=target_second_moments,
            feature_parent_a=state.feature_parent_a,
            feature_parent_b=state.feature_parent_b,
            feature_generator=state.feature_generator,
            candidate_parent_a=state.candidate_parent_a,
            candidate_parent_b=state.candidate_parent_b,
            candidate_generator=state.candidate_generator,
            step_count=step_count,
            birth_timestamp=state.birth_timestamp,
            uptime_s=state.uptime_s,
        )

        replaced_slot = jnp.array(-1, dtype=jnp.int32)
        promoted_candidate = jnp.array(-1, dtype=jnp.int32)
        refreshed_candidate = jnp.array(-1, dtype=jnp.int32)
        retired_slot = jnp.array(-1, dtype=jnp.int32)
        retired_left = jnp.array(-1, dtype=jnp.int32)
        retired_right = jnp.array(-1, dtype=jnp.int32)

        should_try_replace = (self._replacement_interval > 0) & (
            step_count % jnp.array(max(self._replacement_interval, 1)) == 0
        )
        stale_scores = jnp.where(
            live_features
            & (ages >= self._min_feature_age)
            & (
                evidence_idle_steps
                > (
                    -1
                    if self._utility_retention_grace_steps is None
                    else self._utility_retention_grace_steps
                )
            ),
            evidence_idle_steps,
            -1,
        )
        stale_slot = jnp.argmax(stale_scores).astype(jnp.int32)
        has_stale_slot = jnp.any(stale_scores >= 0)
        has_vacancy_before = jnp.any(~live_features)
        should_retire = (
            should_try_replace & self._retire_stale_features & has_stale_slot & ~has_vacancy_before
        )
        retired_slot = jnp.where(should_retire, stale_slot, retired_slot)
        retired_left = jnp.where(
            should_retire,
            feature_left[stale_slot],
            retired_left,
        )
        retired_right = jnp.where(
            should_retire,
            feature_right[stale_slot],
            retired_right,
        )
        retired_candidate_mask = (
            should_retire
            & (candidate_left == feature_left[stale_slot])
            & (candidate_right == feature_right[stale_slot])
        )
        feature_left = jax.lax.select(
            should_retire,
            feature_left.at[stale_slot].set(-1),
            feature_left,
        )
        feature_right = jax.lax.select(
            should_retire,
            feature_right.at[stale_slot].set(-1),
            feature_right,
        )
        output_weights = jax.lax.select(
            should_retire,
            output_weights.at[:, stale_slot].set(0.0),
            output_weights,
        )
        relevance_probe_weights = jax.lax.select(
            should_retire,
            relevance_probe_weights.at[:, stale_slot].set(0.0),
            relevance_probe_weights,
        )
        new_utilities = jax.lax.select(
            should_retire,
            new_utilities.at[stale_slot].set(0.0),
            new_utilities,
        )
        evidence_idle_steps = jax.lax.select(
            should_retire,
            evidence_idle_steps.at[stale_slot].set(0),
            evidence_idle_steps,
        )
        utility_evidence_streak = jax.lax.select(
            should_retire,
            utility_evidence_streak.at[stale_slot].set(0),
            utility_evidence_streak,
        )
        active_output_memory_committed = jax.lax.select(
            should_retire,
            active_output_memory_committed.at[stale_slot].set(False),
            active_output_memory_committed,
        )
        ages = jax.lax.select(
            should_retire,
            ages.at[stale_slot].set(0),
            ages,
        )
        feature_parent_a = jax.lax.select(
            should_retire,
            feature_parent_a.at[stale_slot].set(-1),
            feature_parent_a,
        )
        feature_parent_b = jax.lax.select(
            should_retire,
            feature_parent_b.at[stale_slot].set(-1),
            feature_parent_b,
        )
        feature_generator = jax.lax.select(
            should_retire,
            feature_generator.at[stale_slot].set(-1),
            feature_generator,
        )
        candidate_output_weights = jnp.where(
            retired_candidate_mask[None, :],
            0.0,
            candidate_output_weights,
        )
        new_candidate_utilities = jnp.where(
            retired_candidate_mask,
            0.0,
            new_candidate_utilities,
        )
        candidate_ages = jnp.where(
            retired_candidate_mask,
            0,
            candidate_ages,
        )
        candidate_promotion_evidence_streak = jnp.where(
            retired_candidate_mask,
            0,
            candidate_promotion_evidence_streak,
        )
        candidate_reacquisition_required = jnp.where(
            retired_candidate_mask,
            jnp.asarray(
                self._candidate_reacquisition_confirmation_steps > 1,
                dtype=jnp.bool_,
            ),
            candidate_reacquisition_required,
        )
        if self._scale_robust:
            feature_second_moments = jax.lax.select(
                should_retire,
                feature_second_moments.at[stale_slot].set(0.0),
                feature_second_moments,
            )
            candidate_second_moments = jnp.where(
                retired_candidate_mask,
                0.0,
                candidate_second_moments,
            )

        live_after_retirement = (feature_left >= 0) & (feature_right >= 0)
        inactive_slots = ~live_after_retirement
        first_inactive = jnp.argmax(inactive_slots).astype(jnp.int32)
        has_inactive_slot = jnp.any(inactive_slots)
        eligible_active = live_after_retirement & (ages >= self._min_feature_age)
        active_selection_scores = jnp.where(
            priority_override_enabled,
            active_priority_ranks,
            new_utilities,
        )
        active_scores = jnp.where(
            eligible_active,
            active_selection_scores,
            jnp.inf,
        )
        worst_active = jnp.argmin(active_scores).astype(jnp.int32)
        has_active_slot = jnp.any(eligible_active)
        curation_selected_active_worst_slot = jnp.where(
            has_active_slot,
            worst_active,
            jnp.asarray(-1, dtype=jnp.int32),
        )
        destination_slot = jnp.where(
            has_inactive_slot,
            first_inactive,
            worst_active,
        )
        destination_utility = jnp.where(
            has_inactive_slot,
            0.0,
            new_utilities[worst_active],
        )
        has_destination = has_inactive_slot | has_active_slot
        curation_selected_promotion_candidate = jnp.asarray(-1, dtype=jnp.int32)
        curation_selected_refresh_candidate = jnp.asarray(-1, dtype=jnp.int32)

        if self._candidate_count > 0:
            # Promoting a pair that is already active wastes a fixed-budget
            # slot and can erase useful output weights for no representational
            # gain.  This matters especially for ``all_pairs`` candidate banks:
            # retained candidates intentionally remain available as an archive
            # after promotion, so they must be dormant until their matching
            # active feature has actually been evicted.
            candidate_matches_active = jnp.any(
                (candidate_left[:, None] == feature_left[None, :])
                & (candidate_right[:, None] == feature_right[None, :]),
                axis=1,
            )
            eligible_candidates = (
                candidate_ages >= self._candidate_min_age
            ) & ~candidate_matches_active & candidate_promotion_confirmed
            candidate_selection_scores = jnp.where(
                priority_override_enabled,
                candidate_priority_ranks,
                new_candidate_utilities,
            )
            candidate_scores = jnp.where(
                eligible_candidates,
                candidate_selection_scores,
                -jnp.inf,
            )
            best_candidate = jnp.argmax(candidate_scores).astype(jnp.int32)
            worst_candidate = jnp.argmin(candidate_selection_scores).astype(jnp.int32)
            has_candidate = jnp.any(eligible_candidates)
            curation_selected_promotion_candidate = jnp.where(
                has_candidate,
                best_candidate,
                curation_selected_promotion_candidate,
            )
            curation_selected_refresh_candidate = worst_candidate
            should_promote = (
                should_try_replace
                & ~should_retire
                & has_destination
                & has_candidate
                & (
                    new_candidate_utilities[best_candidate]
                    > jnp.maximum(
                        jnp.asarray(
                            self._candidate_promotion_floor,
                            dtype=jnp.float32,
                        ),
                        self._promotion_margin * destination_utility,
                    )
                )
            )
            should_refresh_candidate = (
                ~should_promote
                & should_try_replace
                & ~should_retire
                & self._refresh_candidates
            )
            should_refresh_promoted_candidate = (
                should_promote & self._refresh_promoted_candidate
            )
            refreshed_candidate = jnp.where(
                should_refresh_promoted_candidate,
                best_candidate,
                jnp.where(
                    should_refresh_candidate,
                    worst_candidate,
                    refreshed_candidate,
                ),
            )

            def promote_branch(
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
                    Array,
                    Array,
                    Array,
                    Array,
                    Array,
                ],
            ) -> tuple[
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
                Array,
                Array,
                Array,
                Array,
                Array,
            ]:
                (
                    fl,
                    fr,
                    ow,
                    util,
                    age,
                    cl,
                    cr,
                    cow,
                    cutil,
                    cage,
                    fpa,
                    fpb,
                    fg,
                    cpa,
                    cpb,
                    cg,
                ) = args
                fl = fl.at[destination_slot].set(cl[best_candidate])
                fr = fr.at[destination_slot].set(cr[best_candidate])
                ow = ow.at[:, destination_slot].set(self._promotion_blend * cow[:, best_candidate])
                util = util.at[destination_slot].set(cutil[best_candidate])
                age = age.at[destination_slot].set(0)
                fpa = fpa.at[destination_slot].set(cpa[best_candidate])
                fpb = fpb.at[destination_slot].set(cpb[best_candidate])
                fg = fg.at[destination_slot].set(cg[best_candidate])

                if self._refresh_promoted_candidate:
                    new_l, new_r, new_pa, new_pb, new_gen = self._generate_one(
                        replacement_key, observation, fl, fr, util
                    )
                    cl = cl.at[best_candidate].set(new_l)
                    cr = cr.at[best_candidate].set(new_r)
                    cpa = cpa.at[best_candidate].set(new_pa)
                    cpb = cpb.at[best_candidate].set(new_pb)
                    cg = cg.at[best_candidate].set(new_gen)
                cow = cow.at[:, best_candidate].set(0.0)
                cutil = cutil.at[best_candidate].set(0.0)
                cage = cage.at[best_candidate].set(0)
                return fl, fr, ow, util, age, cl, cr, cow, cutil, cage, fpa, fpb, fg, cpa, cpb, cg

            def refresh_candidate_branch(
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
                    Array,
                    Array,
                    Array,
                    Array,
                    Array,
                ],
            ) -> tuple[
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
                Array,
                Array,
                Array,
                Array,
                Array,
            ]:
                (
                    fl,
                    fr,
                    ow,
                    util,
                    age,
                    cl,
                    cr,
                    cow,
                    cutil,
                    cage,
                    fpa,
                    fpb,
                    fg,
                    cpa,
                    cpb,
                    cg,
                ) = args
                new_l, new_r, new_pa, new_pb, new_gen = self._generate_one(
                    replacement_key, observation, fl, fr, util
                )
                do_refresh = should_try_replace & ~should_retire & self._refresh_candidates
                cl = jax.lax.select(do_refresh, cl.at[worst_candidate].set(new_l), cl)
                cr = jax.lax.select(do_refresh, cr.at[worst_candidate].set(new_r), cr)
                cow = jax.lax.select(do_refresh, cow.at[:, worst_candidate].set(0.0), cow)
                cutil = jax.lax.select(do_refresh, cutil.at[worst_candidate].set(0.0), cutil)
                cage = jax.lax.select(do_refresh, cage.at[worst_candidate].set(0), cage)
                cpa = jax.lax.select(do_refresh, cpa.at[worst_candidate].set(new_pa), cpa)
                cpb = jax.lax.select(do_refresh, cpb.at[worst_candidate].set(new_pb), cpb)
                cg = jax.lax.select(do_refresh, cg.at[worst_candidate].set(new_gen), cg)
                return fl, fr, ow, util, age, cl, cr, cow, cutil, cage, fpa, fpb, fg, cpa, cpb, cg

            promoted_probe_seed = (
                self._promotion_blend
                * candidate_output_weights[:, best_candidate]
            )
            carry = (
                feature_left,
                feature_right,
                output_weights,
                new_utilities,
                ages,
                candidate_left,
                candidate_right,
                candidate_output_weights,
                new_candidate_utilities,
                candidate_ages,
                feature_parent_a,
                feature_parent_b,
                feature_generator,
                candidate_parent_a,
                candidate_parent_b,
                candidate_generator,
            )
            (
                feature_left,
                feature_right,
                output_weights,
                new_utilities,
                ages,
                candidate_left,
                candidate_right,
                candidate_output_weights,
                new_candidate_utilities,
                candidate_ages,
                feature_parent_a,
                feature_parent_b,
                feature_generator,
                candidate_parent_a,
                candidate_parent_b,
                candidate_generator,
            ) = jax.lax.cond(should_promote, promote_branch, refresh_candidate_branch, carry)
            candidate_indices = jnp.arange(self._candidate_count, dtype=jnp.int32)
            promoted_candidate_reset = should_promote & (
                candidate_indices == best_candidate
            )
            refreshed_candidate_reset = (
                ~should_promote
                & should_try_replace
                & ~should_retire
                & self._refresh_candidates
                & (candidate_indices == worst_candidate)
            )
            candidate_promotion_evidence_streak = jnp.where(
                promoted_candidate_reset | refreshed_candidate_reset,
                0,
                candidate_promotion_evidence_streak,
            )
            candidate_reacquisition_required = jnp.where(
                promoted_candidate_reset | refreshed_candidate_reset,
                False,
                candidate_reacquisition_required,
            )
            probe_destination_value = (
                promoted_probe_seed
                if self._independent_relevance_probe
                else jnp.zeros((self._n_tasks,), dtype=jnp.float32)
            )
            relevance_probe_weights = jax.lax.select(
                should_promote,
                relevance_probe_weights.at[:, destination_slot].set(
                    probe_destination_value
                ),
                relevance_probe_weights,
            )
            if self._independent_relevance_probe:
                output_weights = jax.lax.select(
                    should_promote,
                    output_weights.at[:, destination_slot].set(0.0),
                    output_weights,
                )
            replaced_slot = jnp.where(
                should_promote,
                destination_slot,
                replaced_slot,
            )
            evidence_idle_steps = jax.lax.select(
                should_promote,
                evidence_idle_steps.at[destination_slot].set(0),
                evidence_idle_steps,
            )
            utility_evidence_streak = jax.lax.select(
                should_promote,
                utility_evidence_streak.at[destination_slot].set(0),
                utility_evidence_streak,
            )
            active_output_memory_committed = jax.lax.select(
                should_promote,
                active_output_memory_committed.at[destination_slot].set(False),
                active_output_memory_committed,
            )
            promoted_candidate = jnp.where(should_promote, best_candidate, promoted_candidate)
            if self._scale_robust:
                promoted_feature_moments = feature_second_moments.at[destination_slot].set(
                    candidate_second_moments[best_candidate]
                )
                feature_second_moments = jax.lax.select(
                    should_promote,
                    promoted_feature_moments,
                    feature_second_moments,
                )
                descriptor_was_refreshed = (should_promote & self._refresh_promoted_candidate) | (
                    ~should_promote & should_try_replace & ~should_retire & self._refresh_candidates
                )
                refreshed_candidate_moments = candidate_second_moments.at[
                    jnp.where(
                        should_promote,
                        best_candidate,
                        worst_candidate,
                    )
                ].set(0.0)
                candidate_second_moments = jax.lax.select(
                    descriptor_was_refreshed,
                    refreshed_candidate_moments,
                    candidate_second_moments,
                )
        else:

            def replace_active_branch(
                args: tuple[Array, Array, Array, Array, Array, Array, Array, Array],
            ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
                fl, fr, ow, util, age, fpa, fpb, fg = args
                new_l, new_r, new_pa, new_pb, new_gen = self._generate_one(
                    replacement_key, observation, fl, fr, util
                )
                fl = fl.at[destination_slot].set(new_l)
                fr = fr.at[destination_slot].set(new_r)
                ow = ow.at[:, destination_slot].set(0.0)
                util = util.at[destination_slot].set(0.0)
                age = age.at[destination_slot].set(0)
                fpa = fpa.at[destination_slot].set(new_pa)
                fpb = fpb.at[destination_slot].set(new_pb)
                fg = fg.at[destination_slot].set(new_gen)
                return fl, fr, ow, util, age, fpa, fpb, fg

            def keep_active_branch(
                args: tuple[Array, Array, Array, Array, Array, Array, Array, Array],
            ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
                return args

            do_replace = should_try_replace & ~should_retire & has_destination
            (
                feature_left,
                feature_right,
                output_weights,
                new_utilities,
                ages,
                feature_parent_a,
                feature_parent_b,
                feature_generator,
            ) = jax.lax.cond(
                do_replace,
                replace_active_branch,
                keep_active_branch,
                (
                    feature_left,
                    feature_right,
                    output_weights,
                    new_utilities,
                    ages,
                    feature_parent_a,
                    feature_parent_b,
                    feature_generator,
                ),
            )
            relevance_probe_weights = jax.lax.select(
                do_replace,
                relevance_probe_weights.at[:, destination_slot].set(0.0),
                relevance_probe_weights,
            )
            replaced_slot = jnp.where(do_replace, destination_slot, replaced_slot)
            evidence_idle_steps = jax.lax.select(
                do_replace,
                evidence_idle_steps.at[destination_slot].set(0),
                evidence_idle_steps,
            )
            utility_evidence_streak = jax.lax.select(
                do_replace,
                utility_evidence_streak.at[destination_slot].set(0),
                utility_evidence_streak,
            )
            active_output_memory_committed = jax.lax.select(
                do_replace,
                active_output_memory_committed.at[destination_slot].set(False),
                active_output_memory_committed,
            )
            if self._scale_robust:
                reset_feature_moments = feature_second_moments.at[destination_slot].set(0.0)
                feature_second_moments = jax.lax.select(
                    do_replace,
                    reset_feature_moments,
                    feature_second_moments,
                )

        new_state = InteractionFeatureState(
            key=key,
            feature_left=feature_left,
            feature_right=feature_right,
            output_weights=output_weights,
            relevance_probe_weights=relevance_probe_weights,
            relevance_probe_biases=relevance_probe_biases,
            output_biases=output_biases,
            utilities=new_utilities,
            evidence_idle_steps=evidence_idle_steps,
            utility_evidence_streak=utility_evidence_streak,
            active_output_memory_committed=active_output_memory_committed,
            task_activity_ema=task_activity_ema,
            ages=ages,
            candidate_left=candidate_left,
            candidate_right=candidate_right,
            candidate_output_weights=candidate_output_weights,
            candidate_utilities=new_candidate_utilities,
            candidate_ages=candidate_ages,
            candidate_promotion_evidence_streak=(
                candidate_promotion_evidence_streak
            ),
            candidate_reacquisition_required=candidate_reacquisition_required,
            feature_second_moments=feature_second_moments,
            candidate_second_moments=candidate_second_moments,
            target_second_moments=target_second_moments,
            feature_parent_a=feature_parent_a,
            feature_parent_b=feature_parent_b,
            feature_generator=feature_generator,
            candidate_parent_a=candidate_parent_a,
            candidate_parent_b=candidate_parent_b,
            candidate_generator=candidate_generator,
            step_count=step_count,
            birth_timestamp=state.birth_timestamp,
            uptime_s=state.uptime_s,
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

        committed_state = jax.lax.cond(
            input_valid,
            lambda _: new_state,
            lambda _: state,
            operand=None,
        )
        committed_pre_curation_state = jax.lax.cond(
            input_valid,
            lambda _: pre_curation_state,
            lambda _: state,
            operand=None,
        )
        rejected_index = jnp.asarray(-1, dtype=jnp.int32)
        false_features = jnp.zeros((self._n_features,), dtype=jnp.bool_)
        false_candidates = jnp.zeros((self._candidate_count,), dtype=jnp.bool_)
        zero_feature_scores = jnp.zeros((self._n_features,), dtype=jnp.float32)
        zero_probe_errors = jnp.zeros(
            (self._n_tasks, self._n_features),
            dtype=jnp.float32,
        )
        zero_candidate_scores = jnp.zeros(
            (self._candidate_count,),
            dtype=jnp.float32,
        )
        old_live_count = jnp.sum(
            (state.feature_left >= 0) & (state.feature_right >= 0),
            dtype=jnp.int32,
        )
        old_vacancy_count = jnp.sum(
            (state.feature_left < 0) | (state.feature_right < 0),
            dtype=jnp.int32,
        )

        return InteractionFeatureUpdateResult(
            state=committed_state,
            pre_curation_state=committed_pre_curation_state,
            predictions=jnp.where(
                input_valid,
                predictions,
                jnp.full_like(predictions, jnp.nan),
            ),
            errors=jnp.where(
                input_valid,
                reported_errors,
                jnp.full_like(reported_errors, jnp.nan),
            ),
            metrics=jnp.where(input_valid, metrics, jnp.zeros_like(metrics)),
            replaced_slot=jnp.where(input_valid, replaced_slot, rejected_index),
            promoted_candidate=jnp.where(
                input_valid,
                promoted_candidate,
                rejected_index,
            ),
            refreshed_candidate=jnp.where(
                input_valid,
                refreshed_candidate,
                rejected_index,
            ),
            retired_slot=jnp.where(input_valid, retired_slot, rejected_index),
            retired_left=jnp.where(input_valid, retired_left, rejected_index),
            retired_right=jnp.where(input_valid, retired_right, rejected_index),
            evidence_refreshed=jnp.where(
                input_valid,
                evidence_refreshed,
                false_features,
            ),
            retention_evidence_refreshed=jnp.where(
                input_valid,
                retention_evidence_refreshed,
                false_features,
            ),
            relevance_probe_scores=jnp.where(
                input_valid,
                relevance_probe_scores,
                zero_feature_scores,
            ),
            relevance_probe_errors=jnp.where(
                input_valid,
                relevance_probe_errors,
                zero_probe_errors,
            ),
            candidate_promotion_signal=jnp.where(
                input_valid,
                candidate_promotion_signal,
                zero_candidate_scores,
            ),
            candidate_promotion_raw_evidence=(
                jnp.where(
                    input_valid,
                    candidate_promotion_raw_evidence,
                    false_candidates,
                )
            ),
            candidate_promotion_evidence_streak_pre=(
                candidate_promotion_evidence_streak_pre
            ),
            candidate_promotion_evidence_streak_updated=(
                jnp.where(
                    input_valid,
                    candidate_promotion_evidence_streak_updated,
                    state.candidate_promotion_evidence_streak,
                )
            ),
            candidate_promotion_confirmed=jnp.where(
                input_valid,
                candidate_promotion_confirmed,
                false_candidates,
            ),
            candidate_reacquisition_required_pre=(
                candidate_reacquisition_required_pre
            ),
            candidate_reacquisition_required_post=(
                jnp.where(
                    input_valid,
                    candidate_reacquisition_required,
                    state.candidate_reacquisition_required,
                )
            ),
            candidate_reacquisition_confirmed=jnp.where(
                input_valid,
                candidate_reacquisition_confirmed,
                false_candidates,
            ),
            durable_read_mask=jnp.where(
                input_valid,
                durable_read_mask,
                false_features,
            ),
            matching_candidate_reset_mask=jnp.where(
                input_valid,
                retired_candidate_mask,
                false_candidates,
            ),
            live_feature_count=jnp.where(
                input_valid,
                jnp.sum(
                    (feature_left >= 0) & (feature_right >= 0),
                    dtype=jnp.int32,
                ),
                old_live_count,
            ),
            vacancy_count=jnp.where(
                input_valid,
                jnp.sum(
                    (feature_left < 0) | (feature_right < 0),
                    dtype=jnp.int32,
                ),
                old_vacancy_count,
            ),
            promoted_into_vacancy=jnp.where(
                input_valid,
                (promoted_candidate >= 0) & has_inactive_slot,
                False,
            ),
            curation_priority_override_enabled=priority_override_enabled,
            curation_attempted=input_valid & should_try_replace,
            curation_priority_override_applied=(
                input_valid
                & should_try_replace
                & priority_override_enabled
            ),
            curation_selected_active_worst_slot=jnp.where(
                input_valid & should_try_replace,
                curation_selected_active_worst_slot,
                rejected_index,
            ),
            curation_selected_promotion_candidate=jnp.where(
                input_valid & should_try_replace,
                curation_selected_promotion_candidate,
                rejected_index,
            ),
            curation_selected_refresh_candidate=jnp.where(
                input_valid & should_try_replace,
                curation_selected_refresh_candidate,
                rejected_index,
            ),
            update_rejected=~input_valid,
        )


def run_interaction_feature_arrays(
    learner: FixedBudgetInteractionLearner,
    state: InteractionFeatureState,
    observations: Array,
    targets: Array,
) -> InteractionFeatureLearningResult:
    """Run an interaction-feature learner over pre-collected arrays."""

    def step_fn(
        carry: InteractionFeatureState,
        inputs: tuple[Array, Array],
    ) -> tuple[InteractionFeatureState, Array]:
        observation, target = inputs
        result = learner.update(carry, observation, target)
        return result.state, result.metrics

    t0 = time.time()
    final_state, metrics = jax.lax.scan(step_fn, state, (observations, targets))
    elapsed = time.time() - t0
    final_state = final_state.replace(uptime_s=final_state.uptime_s + elapsed)  # type: ignore[attr-defined]
    return InteractionFeatureLearningResult(state=final_state, metrics=metrics)


def save_interaction_feature_checkpoint(
    learner: FixedBudgetInteractionLearner,
    state: InteractionFeatureState,
    path: str | Path,
    *,
    feature_dim: int,
) -> None:
    """Persist state with its explicit probe semantics and resource contract."""
    if isinstance(feature_dim, bool) or not isinstance(feature_dim, int) or feature_dim < 1:
        raise ValueError("feature_dim must be a positive integer")
    save_checkpoint(
        state,
        path,
        metadata={
            "schema": INTERACTION_FEATURE_CHECKPOINT_SCHEMA,
            "learner_config": learner.to_config(),
            "feature_dim": feature_dim,
            "memory_accounting": learner.memory_accounting(state),
        },
    )


def load_interaction_feature_checkpoint(
    path: str | Path,
) -> tuple[FixedBudgetInteractionLearner, InteractionFeatureState]:
    """Restore a checkpoint only after reconstructing its versioned probe mode."""
    metadata = load_checkpoint_metadata(path)
    expected_fields = {
        "schema",
        "learner_config",
        "feature_dim",
        "memory_accounting",
    }
    if set(metadata) != expected_fields:
        raise ValueError("interaction-feature checkpoint metadata fields are invalid")
    if metadata.get("schema") != INTERACTION_FEATURE_CHECKPOINT_SCHEMA:
        raise ValueError("interaction-feature checkpoint schema is unsupported")
    config = metadata.get("learner_config")
    if not isinstance(config, dict):
        raise ValueError("interaction-feature checkpoint is missing learner_config")
    learner = FixedBudgetInteractionLearner.from_config(config)
    feature_dim = metadata.get("feature_dim")
    if isinstance(feature_dim, bool) or not isinstance(feature_dim, int) or feature_dim < 1:
        raise ValueError("interaction-feature checkpoint feature_dim is invalid")
    template = learner.init(feature_dim=feature_dim, key=jr.key(0))
    restored, restored_metadata = load_checkpoint(template, path)
    if restored_metadata != metadata:
        raise ValueError("interaction-feature checkpoint metadata changed between reads")
    if learner.memory_accounting(restored) != metadata.get("memory_accounting"):
        raise ValueError("interaction-feature checkpoint resource contract does not match")
    return learner, cast(InteractionFeatureState, restored)


def run_interaction_feature_loop(
    learner: FixedBudgetInteractionLearner,
    stream: Any,
    num_steps: int,
    key: Array,
    learner_state: InteractionFeatureState | None = None,
) -> InteractionFeatureLearningResult:
    """Run interaction feature discovery directly from a scan-compatible stream."""
    stream_key, learner_key = jr.split(key)
    stream_state = stream.init(stream_key)
    if learner_state is None:
        learner_state = learner.init(stream.feature_dim, learner_key)

    def step_fn(
        carry: tuple[InteractionFeatureState, Any],
        idx: Array,
    ) -> tuple[tuple[InteractionFeatureState, Any], Array]:
        l_state, s_state = carry
        timestep, new_s_state = stream.step(s_state, idx)
        result = learner.update(l_state, timestep.observation, timestep.target)
        return (result.state, new_s_state), result.metrics

    t0 = time.time()
    (final_state, _), metrics = jax.lax.scan(
        step_fn, (learner_state, stream_state), jnp.arange(num_steps)
    )
    elapsed = time.time() - t0
    final_state = final_state.replace(uptime_s=final_state.uptime_s + elapsed)  # type: ignore[attr-defined]
    return InteractionFeatureLearningResult(state=final_state, metrics=metrics)
