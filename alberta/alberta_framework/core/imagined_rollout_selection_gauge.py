# mypy: disable-error-code="attr-defined,call-arg,no-any-return"
"""Grounded selection and bounded training for imagined rollout proposals.

This module is an isolated L0 mechanism for WP4.6.  It consumes the immutable
fixed-shape :class:`ImaginedRolloutBatch` emitted by
``EnsembleShortRolloutPlanner``.  It never calls the planner, dispatches an
action, or owns or mutates planner, model, environment, safety-envelope, or
Prototype state.

The selection gauge freezes one source/model generation and retains a bounded
caller-grounded audit.  Audit records contain exact realized reward,
successor observation, termination, validity, and success fields.  Statistics
are maintained causally for each primitive-action/fixed-region cell: a record
observes its pre-update cell statistics and only then updates the calibration
state.  Candidate authorization is a separate operation.  A proposal already
present in the audit store can never authorize itself.

Every authorization receipt binds every word of the candidate batch, its
caller-owned fixed-region assignments, full safety/protected masks, the frozen
generation, and the exact calibration revision/content.  The thresholds are
caller declarations and development-only evidence floors.  They are not a
calibration, safety, efficacy, control-benefit, or promotion claim.

All content tags in this module provide unkeyed post-mint integrity only.  The
planner does not issue a batch-content seal, so this gauge cannot authenticate
that candidate tensors were emitted by the planner.  Realized audit fields and
competent-real labels are caller-owned truth claims, not environment-attested
facts.  The public constants, configs, and diagnostics keep that boundary
machine-readable.

The companion actor/critic owns a separate linear policy, value function, and
momentum state.  Its proposal is authorization metadata only.  Commit first
revalidates the complete source and proposal, then performs at most one
fixed-shape ``jax.value_and_grad`` pass.  Dream actor updates use diagnosed
graded positive-advantage self-imitation; critic targets replace terminal
return targets with the exact terminal reward.  A matched competent-real
episode cloning source uses the same transition and update budgets.  Neither
source has output, dispatch, promotion, or safety authority.  Clocks bound an
accepted functional state lineage; a caller that discards returned state can
repeat a pure call, so they are not global wall-clock or compute authority.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
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

from alberta_framework.core.checkpoints import (
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)
from alberta_framework.core.ensemble_short_rollouts import (
    EnsembleShortRolloutPlanner,
    ImaginedRolloutBatch,
)

IMAGINED_ROLLOUT_GAUGE_CONFIG_SCHEMA = (
    "alberta.imagined-rollout-selection-gauge.config.v1"
)
IMAGINED_ROLLOUT_GAUGE_CHECKPOINT_SCHEMA = (
    "alberta.imagined-rollout-selection-gauge.checkpoint.v1"
)
IMAGINED_ROLLOUT_ACTOR_CRITIC_CONFIG_SCHEMA = (
    "alberta.imagined-rollout-actor-critic.config.v1"
)
IMAGINED_ROLLOUT_ACTOR_CRITIC_CHECKPOINT_SCHEMA = (
    "alberta.imagined-rollout-actor-critic.checkpoint.v1"
)
IMAGINED_ROLLOUT_GAUGE_MECHANISM_STATUS = "l0-development-only-not-assessed"
IMAGINED_ROLLOUT_GAUGE_EVIDENCE_LEVEL = "L0"
IMAGINED_ROLLOUT_SCIENTIFIC_PROMOTION_ALLOWED = False
IMAGINED_ROLLOUT_CONTENT_INTEGRITY_SCOPE = "post-mint-unkeyed-integrity-only"
IMAGINED_ROLLOUT_PLANNER_ISSUANCE_AUTHENTICATED = False
IMAGINED_ROLLOUT_COMPETENT_REAL_TRUTH_AUTHENTICATED = False

_INT32_MAX = 2_147_483_647
_UINT32_MAX = 4_294_967_295
_FLOAT32_MAX = float(np.finfo(np.float32).max)
_FLOAT32_TINY = float(np.finfo(np.float32).tiny)
_MAX_AUDIT_CAPACITY = 4_096
_MAX_REGIONS = 256
_MAX_AUTHORIZATIONS = _INT32_MAX
_MAX_UPDATES = _INT32_MAX
_TAG_OFFSET = 2_166_136_261
_TAG_PRIME = 16_777_619
_CALIBRATION_TAG_SALT = 0x43414C49
_STATE_TAG_SALT = 0x47535441
_RECORD_TAG_SALT = 0x52454344
_PROPOSAL_TAG_SALT = 0x50524F50
_TRANSITION_TAG_SALT = 0x5452414E
_RECEIPT_TAG_SALT = 0x52435054
_REAL_BATCH_TAG_SALT = 0x5245414C
_LEARNER_STATE_TAG_SALT = 0x4C535441
_LEARNER_PROPOSAL_TAG_SALT = 0x4C505250


def _positive_int(value: object, *, name: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be a strict integer in [1, {maximum}]")
    return value


def _finite_float32(
    value: object,
    *,
    name: str,
    minimum: float,
    maximum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real non-boolean scalar")
    parsed = float(value)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        narrowed = float(np.float32(parsed))
    if not math.isfinite(parsed) or not math.isfinite(narrowed):
        raise ValueError(f"{name} must remain finite in float32")
    if parsed != 0.0 and abs(narrowed) < _FLOAT32_TINY:
        raise ValueError(f"{name} must not underflow in float32")
    if narrowed < minimum or (strictly_positive and narrowed == 0.0):
        comparator = "positive" if strictly_positive else f">= {minimum}"
        raise ValueError(f"{name} must be {comparator}")
    if maximum is not None and narrowed > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return narrowed


def _array_contract(
    value: object,
    *,
    shape: tuple[int, ...],
    dtype: Any,
) -> bool:
    return (
        hasattr(value, "shape")
        and hasattr(value, "dtype")
        and tuple(cast(Any, value).shape) == shape
        and cast(Any, value).dtype == jnp.dtype(dtype)
    )


def _require_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> None:
    if not _array_contract(value, shape=shape, dtype=dtype):
        raise TypeError(f"{name} must have exact shape {shape} and dtype {jnp.dtype(dtype)}")


def _words_nonzero(words: Array) -> Bool[Array, ""]:
    return jnp.any(words != jnp.asarray(0, dtype=jnp.uint32))


def _words_less(left: Array, right: Array) -> Bool[Array, ""]:
    return (left[0] < right[0]) | ((left[0] == right[0]) & (left[1] < right[1]))


def _words_less_equal(left: Array, right: Array) -> Bool[Array, ""]:
    return (left[0] < right[0]) | ((left[0] == right[0]) & (left[1] <= right[1]))


def _checked_words_add_small(words: Array, amount: Array | int) -> tuple[Array, Array]:
    increment = jnp.asarray(amount, dtype=jnp.uint32)
    low = words[1] + increment
    carry = (low < words[1]).astype(jnp.uint32)
    high = words[0] + carry
    overflow = (carry != 0) & (high == jnp.asarray(0, dtype=jnp.uint32))
    candidate = jnp.stack((high, low)).astype(jnp.uint32)
    return jnp.where(overflow, words, candidate), ~overflow


def _words_leq_limit(words: Array, limit: int) -> Bool[Array, ""]:
    limit_words = jnp.asarray(
        ((limit >> 32) & _UINT32_MAX, limit & _UINT32_MAX),
        dtype=jnp.uint32,
    )
    return _words_less_equal(words, limit_words)


def _saturating_int32(words: Array) -> Int[Array, ""]:
    saturated = (words[0] != 0) | (
        words[1] >= jnp.asarray(_INT32_MAX, dtype=jnp.uint32)
    )
    return jnp.where(
        saturated,
        jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        words[1].astype(jnp.int32),
    )


def _float_words(value: Array) -> Array:
    return jax.lax.bitcast_convert_type(value, jnp.uint32)


def _tree_content_words(tree: object) -> Array:
    parts: list[Array] = []
    for leaf in jax.tree.leaves(tree):
        if leaf is None:
            continue
        array = jnp.asarray(leaf)
        if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
            words = jr.key_data(leaf)
        elif array.dtype == jnp.dtype(jnp.float32):
            words = _float_words(array)
        elif array.dtype in (
            jnp.dtype(jnp.int32),
            jnp.dtype(jnp.uint32),
            jnp.dtype(jnp.bool_),
        ):
            words = array.astype(jnp.uint32)
        else:
            raise TypeError(f"unsupported integrity-receipt dtype {array.dtype}")
        parts.append(jnp.ravel(words))
    if not parts:
        return jnp.zeros((0,), dtype=jnp.uint32)
    return jnp.concatenate(parts)


def _mix_words(words: Array, *, salt: int) -> UInt[Array, ""]:
    flat = jnp.ravel(words).astype(jnp.uint32)

    def body(index: int, tag: Array) -> Array:
        position = (jnp.asarray(index, dtype=jnp.uint32) + 1) * jnp.asarray(
            0x9E3779B9,
            dtype=jnp.uint32,
        )
        mixed = (tag ^ flat[index] ^ position) * jnp.asarray(
            _TAG_PRIME,
            dtype=jnp.uint32,
        )
        return (mixed << jnp.asarray(13, dtype=jnp.uint32)) | (
            mixed >> jnp.asarray(19, dtype=jnp.uint32)
        )

    tag = jax.lax.fori_loop(
        0,
        flat.shape[0],
        body,
        jnp.asarray(_TAG_OFFSET ^ salt, dtype=jnp.uint32),
    )
    return jnp.where(
        tag == jnp.asarray(0, dtype=jnp.uint32),
        jnp.asarray(salt, dtype=jnp.uint32),
        tag,
    )


def _tree_equal(left: object, right: object) -> Bool[Array, ""]:
    left_leaves, left_structure = jax.tree.flatten(left)
    right_leaves, right_structure = jax.tree.flatten(right)
    if (
        cast(Any, left_structure) != cast(Any, right_structure)
        or len(left_leaves) != len(right_leaves)
    ):
        return jnp.asarray(False, dtype=jnp.bool_)
    result = jnp.asarray(True, dtype=jnp.bool_)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = jnp.asarray(left_leaf)
        right_array = jnp.asarray(right_leaf)
        result = result & jnp.array_equal(left_array, right_array)
    return result


def _tree_finite(tree: object) -> Bool[Array, ""]:
    valid = jnp.asarray(True, dtype=jnp.bool_)
    for leaf in jax.tree.leaves(tree):
        array = jnp.asarray(leaf)
        if jnp.issubdtype(array.dtype, jnp.inexact):
            valid = valid & jnp.all(jnp.isfinite(array))
    return valid


def _logical_tree_size(tree: object) -> tuple[int, int]:
    scalars = 0
    nbytes = 0
    for leaf in jax.tree.leaves(tree):
        array = jnp.asarray(leaf)
        if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
            array = jr.key_data(leaf)
        scalars += int(array.size)
        nbytes += int(array.nbytes)
    return scalars, nbytes


def _tree_static_signature(tree: object) -> tuple[object, tuple[tuple[object, ...], ...]]:
    leaves, structure = jax.tree.flatten(tree)
    signatures: list[tuple[object, ...]] = []
    for leaf in leaves:
        array = jnp.asarray(leaf)
        if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
            signatures.append((array.shape, "typed-prng", str(jr.key_impl(leaf))))
        else:
            signatures.append((array.shape, np.dtype(array.dtype).str))
    return structure, tuple(signatures)


def _terminal_path_semantics_valid(
    rewards: Array,
    continuations: Array,
    return_targets: Array,
    terminated: Array,
    transition_valid: Array,
) -> Bool[Array, ""]:
    """Require canonical valid prefixes and prohibit post-terminal resurrection."""

    horizon = transition_valid.shape[1]
    valid_counts = jnp.sum(transition_valid.astype(jnp.int32), axis=1)
    prefix = jnp.arange(horizon, dtype=jnp.int32)[None, :] < valid_counts[:, None]
    terminal = terminated & transition_valid
    terminal_seen = jnp.cumsum(terminal.astype(jnp.int32), axis=1)
    terminal_seen_before = terminal_seen - terminal.astype(jnp.int32)
    return (
        jnp.all(transition_valid == prefix)
        & jnp.all(~terminated | transition_valid)
        & jnp.all(~transition_valid | (terminal_seen_before == 0))
        & jnp.all(~transition_valid | (continuations >= 0.0))
        & jnp.all(~transition_valid | (continuations <= 1.0))
        & jnp.all(~terminal | (continuations == 0.0))
        & jnp.all(
            ~terminal
            | jnp.isclose(
                return_targets,
                rewards,
                rtol=1.0e-5,
                atol=1.0e-6,
            )
        )
    )


def _prefix_closed_transition_mask(
    transition_valid: Array,
    local_admitted: Array,
) -> Bool[Array, "rollout_budget rollout_horizon"]:
    """Admit a transition only when every valid predecessor is admitted."""

    prefix_admitted = jnp.cumprod(
        (~transition_valid | local_admitted).astype(jnp.int32),
        axis=1,
    ).astype(jnp.bool_)
    return transition_valid & prefix_admitted


def _config_digest(config: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _config_fingerprint(config: Mapping[str, object]) -> UInt[Array, " 8"]:
    encoded = json.dumps(
        dict(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).digest()
    return jnp.asarray(
        [
            int.from_bytes(digest[offset : offset + 4], "big")
            for offset in range(0, 32, 4)
        ],
        dtype=jnp.uint32,
    )


@dataclasses.dataclass(frozen=True)
class ImaginedRolloutSelectionGaugeConfig:
    """Declared audit capacity and noncompensating development-only floors."""

    audit_capacity: int = 64
    n_regions: int = 8
    min_evidence_count: int = 8
    min_realized_valid_fraction: float = 1.0
    max_mean_abs_reward_error: float = 1.0
    max_root_mean_square_next_observation_error: float = 1.0
    min_termination_accuracy: float = 1.0
    require_success_lcb: bool = True
    success_lcb_z: float = 1.96
    min_success_lcb: float = 0.0
    require_top_quantile_purity: bool = True
    top_quantile_fraction: float = 0.2
    min_top_quantile_purity: float = 0.0
    max_authorizations: int = _INT32_MAX

    def __post_init__(self) -> None:
        _positive_int(
            self.audit_capacity,
            name="audit_capacity",
            maximum=_MAX_AUDIT_CAPACITY,
        )
        _positive_int(self.n_regions, name="n_regions", maximum=_MAX_REGIONS)
        _positive_int(
            self.min_evidence_count,
            name="min_evidence_count",
            maximum=self.audit_capacity,
        )
        _positive_int(
            self.max_authorizations,
            name="max_authorizations",
            maximum=_MAX_AUTHORIZATIONS,
        )
        for name in ("require_success_lcb", "require_top_quantile_purity"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a strict boolean")
        specs = (
            ("min_realized_valid_fraction", 0.0, 1.0, False),
            ("max_mean_abs_reward_error", 0.0, None, False),
            (
                "max_root_mean_square_next_observation_error",
                0.0,
                None,
                False,
            ),
            ("min_termination_accuracy", 0.0, 1.0, False),
            ("success_lcb_z", 0.0, None, False),
            ("min_success_lcb", 0.0, 1.0, False),
            ("top_quantile_fraction", _FLOAT32_TINY, 1.0, True),
            ("min_top_quantile_purity", 0.0, 1.0, False),
        )
        for name, minimum, maximum, positive in specs:
            object.__setattr__(
                self,
                name,
                _finite_float32(
                    getattr(self, name),
                    name=name,
                    minimum=minimum,
                    maximum=maximum,
                    strictly_positive=positive,
                ),
            )


@chex.dataclass(frozen=True)
class GroundedRolloutAuditRecord:
    """One caller-owned realized outcome bound to one immutable proposal slot."""

    proposal_content_tag: UInt[Array, ""]
    transition_content_tag: UInt[Array, ""]
    rollout_index: Int[Array, ""]
    step_index: Int[Array, ""]
    action: Int[Array, ""]
    region_id: Int[Array, ""]
    record_id_words: UInt[Array, " 2"]
    source_revision_words: UInt[Array, " 2"]
    model_revision_words: UInt[Array, " 2"]
    source_integrity_tag: UInt[Array, ""]
    model_integrity_tag: UInt[Array, ""]
    predicted_reward: Float[Array, ""]
    predicted_next_observation: Float[Array, " observation_dim"]
    predicted_terminated: Bool[Array, ""]
    predicted_score: Float[Array, ""]
    proposal_batch_valid: Bool[Array, ""]
    proposal_transition_valid: Bool[Array, ""]
    realized_valid: Bool[Array, ""]
    realized_reward: Float[Array, ""]
    realized_next_observation: Float[Array, " observation_dim"]
    realized_terminated: Bool[Array, ""]
    realized_success: Bool[Array, ""]
    record_integrity_tag: UInt[Array, ""]


@chex.dataclass(frozen=True)
class ImaginedRolloutSelectionGaugeState:
    """Frozen generation, bounded grounded records, summaries, and clocks."""

    bound_source_revision_words: UInt[Array, " 2"]
    bound_model_revision_words: UInt[Array, " 2"]
    bound_source_integrity_tag: UInt[Array, ""]
    bound_model_integrity_tag: UInt[Array, ""]
    record_valid: Bool[Array, " audit_capacity"]
    record_id_words: UInt[Array, "audit_capacity 2"]
    record_proposal_tags: UInt[Array, " audit_capacity"]
    record_transition_tags: UInt[Array, " audit_capacity"]
    record_rollout_indices: Int[Array, " audit_capacity"]
    record_step_indices: Int[Array, " audit_capacity"]
    record_actions: Int[Array, " audit_capacity"]
    record_regions: Int[Array, " audit_capacity"]
    record_predicted_rewards: Float[Array, " audit_capacity"]
    record_realized_rewards: Float[Array, " audit_capacity"]
    record_predicted_next_observations: Float[
        Array, "audit_capacity observation_dim"
    ]
    record_realized_next_observations: Float[
        Array, "audit_capacity observation_dim"
    ]
    record_predicted_terminated: Bool[Array, " audit_capacity"]
    record_realized_terminated: Bool[Array, " audit_capacity"]
    record_realized_valid: Bool[Array, " audit_capacity"]
    record_realized_success: Bool[Array, " audit_capacity"]
    record_scores: Float[Array, " audit_capacity"]
    record_integrity_tags: UInt[Array, " audit_capacity"]
    evidence_counts: Int[Array, "n_actions n_regions"]
    realized_valid_counts: Int[Array, "n_actions n_regions"]
    reward_absolute_error_sums: Float[Array, "n_actions n_regions"]
    next_state_squared_error_sums: Float[Array, "n_actions n_regions"]
    termination_correct_counts: Int[Array, "n_actions n_regions"]
    success_counts: Int[Array, "n_actions n_regions"]
    top_quantile_counts: Int[Array, "n_actions n_regions"]
    top_quantile_purity: Float[Array, "n_actions n_regions"]
    last_record_id_words: UInt[Array, " 2"]
    calibration_revision_words: UInt[Array, " 2"]
    authorization_count_words: UInt[Array, " 2"]
    calibration_content_tag: UInt[Array, ""]
    state_integrity_tag: UInt[Array, ""]


@chex.dataclass(frozen=True)
class GroundedRolloutAuditDiagnostics:
    """Causal pre-update cell statistics and atomic audit verdict."""

    state_valid: Bool[Array, ""]
    record_valid: Bool[Array, ""]
    generation_matches: Bool[Array, ""]
    record_identity_fresh: Bool[Array, ""]
    proposal_slot_fresh: Bool[Array, ""]
    capacity_available: Bool[Array, ""]
    pre_evidence_count: Int[Array, ""]
    pre_realized_valid_count: Int[Array, ""]
    pre_mean_absolute_reward_error: Float[Array, ""]
    pre_root_mean_square_next_state_error: Float[Array, ""]
    pre_termination_accuracy: Float[Array, ""]
    pre_success_count: Int[Array, ""]
    pre_top_quantile_purity: Float[Array, ""]
    applied: Bool[Array, ""]
    pre_calibration_revision_words: UInt[Array, " 2"]
    post_calibration_revision_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class GroundedRolloutAuditResult:
    state: ImaginedRolloutSelectionGaugeState
    diagnostics: GroundedRolloutAuditDiagnostics


@chex.dataclass(frozen=True)
class ImaginedRolloutAuthorizationReceipt:
    """Exact candidate/calibration binding with full transition channels."""

    planner_fingerprint: UInt[Array, " 8"]
    proposal_content_tag: UInt[Array, ""]
    calibration_revision_words: UInt[Array, " 2"]
    calibration_content_tag: UInt[Array, ""]
    authorization_words: UInt[Array, " 2"]
    source_revision_words: UInt[Array, " 2"]
    model_revision_words: UInt[Array, " 2"]
    source_integrity_tag: UInt[Array, ""]
    model_integrity_tag: UInt[Array, ""]
    region_ids: Int[Array, "rollout_budget rollout_horizon"]
    safety_admitted: Bool[Array, "rollout_budget rollout_horizon"]
    protected: Bool[Array, "rollout_budget rollout_horizon"]
    evidence_counts: Int[Array, "rollout_budget rollout_horizon"]
    realized_valid_fractions: Float[Array, "rollout_budget rollout_horizon"]
    mean_absolute_reward_errors: Float[Array, "rollout_budget rollout_horizon"]
    root_mean_square_next_state_errors: Float[
        Array, "rollout_budget rollout_horizon"
    ]
    termination_accuracies: Float[Array, "rollout_budget rollout_horizon"]
    success_lcbs: Float[Array, "rollout_budget rollout_horizon"]
    top_quantile_purities: Float[Array, "rollout_budget rollout_horizon"]
    evidence_count_passed: Bool[Array, "rollout_budget rollout_horizon"]
    realized_validity_passed: Bool[Array, "rollout_budget rollout_horizon"]
    reward_error_passed: Bool[Array, "rollout_budget rollout_horizon"]
    next_state_error_passed: Bool[Array, "rollout_budget rollout_horizon"]
    termination_passed: Bool[Array, "rollout_budget rollout_horizon"]
    success_lcb_passed: Bool[Array, "rollout_budget rollout_horizon"]
    top_quantile_purity_passed: Bool[Array, "rollout_budget rollout_horizon"]
    evidence_floor_passed: Bool[Array, "rollout_budget rollout_horizon"]
    transition_authorized: Bool[Array, "rollout_budget rollout_horizon"]
    authorized: Bool[Array, ""]
    receipt_integrity_tag: UInt[Array, ""]


@chex.dataclass(frozen=True)
class ImaginedRolloutAuthorizationDiagnostics:
    state_valid: Bool[Array, ""]
    batch_valid: Bool[Array, ""]
    generation_matches: Bool[Array, ""]
    audit_candidate_separated: Bool[Array, ""]
    region_ids_valid: Bool[Array, ""]
    terminal_semantics_valid: Bool[Array, ""]
    authorization_capacity_available: Bool[Array, ""]
    post_mint_content_integrity_only: Bool[Array, ""]
    planner_issuance_authenticated: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]
    receipt_valid: Bool[Array, ""]
    pre_authorization_words: UInt[Array, " 2"]
    post_authorization_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class ImaginedRolloutAuthorizationResult:
    state: ImaginedRolloutSelectionGaugeState
    receipt: ImaginedRolloutAuthorizationReceipt
    diagnostics: ImaginedRolloutAuthorizationDiagnostics


@dataclasses.dataclass(frozen=True)
class ImaginedRolloutSelectionGaugeResourceBudget:
    persistent_state_scalars: int
    persistent_state_bytes: int
    receipt_scalars: int
    receipt_bytes: int
    max_grounded_records: int
    max_authorizations: int
    per_action_region_cells: int
    planner_or_model_state_owned: int
    dispatch_authority: int
    safety_authority: int
    output_authority: int
    scientific_promotion_allowed: bool

    def to_config(self) -> dict[str, object]:
        return dataclasses.asdict(self)


class ImaginedRolloutSelectionGauge:
    """Bounded grounded audit and source-bound imagined-rollout authorizer."""

    def __init__(
        self,
        planner: EnsembleShortRolloutPlanner,
        config: ImaginedRolloutSelectionGaugeConfig | None = None,
    ) -> None:
        if not isinstance(planner, EnsembleShortRolloutPlanner):
            raise TypeError("planner must be an EnsembleShortRolloutPlanner")
        self._config = config or ImaginedRolloutSelectionGaugeConfig()
        self._planner_config = planner.to_config()
        self._planner_fingerprint = _config_fingerprint(self._planner_config)
        self._observation_dim = planner.observation_dim
        self._n_actions = planner.n_actions
        self._rollout_budget = planner.config.rollout_budget
        self._rollout_horizon = planner.config.rollout_horizon
        empty = self._empty_state()
        self._state_signature = _tree_static_signature(empty)
        self._record_signature = _tree_static_signature(self._zero_record())
        self._batch_signature = _tree_static_signature(self._zero_batch())
        self._receipt_signature = _tree_static_signature(
            self._zero_receipt()
        )

    @property
    def config(self) -> ImaginedRolloutSelectionGaugeConfig:
        return self._config

    @property
    def observation_dim(self) -> int:
        return self._observation_dim

    @property
    def n_actions(self) -> int:
        return self._n_actions

    @property
    def rollout_budget(self) -> int:
        return self._rollout_budget

    @property
    def rollout_horizon(self) -> int:
        return self._rollout_horizon

    def to_config(self) -> dict[str, object]:
        return {
            "schema": IMAGINED_ROLLOUT_GAUGE_CONFIG_SCHEMA,
            "type": type(self).__name__,
            "mechanism_status": IMAGINED_ROLLOUT_GAUGE_MECHANISM_STATUS,
            "evidence_level": IMAGINED_ROLLOUT_GAUGE_EVIDENCE_LEVEL,
            "scientific_promotion_allowed": False,
            "calibration_claimed": False,
            "content_integrity_scope": IMAGINED_ROLLOUT_CONTENT_INTEGRITY_SCOPE,
            "planner_issuance_authenticated": False,
            "safety_authority": False,
            "dispatch_authority": False,
            "output_authority": False,
            "planner": self._planner_config,
            "gauge": dataclasses.asdict(self._config),
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> ImaginedRolloutSelectionGauge:
        payload = dict(config)
        expected = {
            "schema",
            "type",
            "mechanism_status",
            "evidence_level",
            "scientific_promotion_allowed",
            "calibration_claimed",
            "content_integrity_scope",
            "planner_issuance_authenticated",
            "safety_authority",
            "dispatch_authority",
            "output_authority",
            "planner",
            "gauge",
        }
        if set(payload) != expected:
            raise ValueError("imagined-rollout gauge config fields do not match v1")
        if payload.pop("schema") != IMAGINED_ROLLOUT_GAUGE_CONFIG_SCHEMA:
            raise ValueError("unsupported imagined-rollout gauge config schema")
        if payload.pop("type") != cls.__name__:
            raise ValueError("unexpected imagined-rollout gauge config type")
        if payload.pop("mechanism_status") != IMAGINED_ROLLOUT_GAUGE_MECHANISM_STATUS:
            raise ValueError("imagined-rollout gauge must remain not assessed")
        if payload.pop("evidence_level") != "L0":
            raise ValueError("imagined-rollout gauge evidence level must remain L0")
        if (
            payload.pop("content_integrity_scope")
            != IMAGINED_ROLLOUT_CONTENT_INTEGRITY_SCOPE
        ):
            raise ValueError("imagined-rollout content tags must remain post-mint only")
        for name in (
            "scientific_promotion_allowed",
            "calibration_claimed",
            "planner_issuance_authenticated",
            "safety_authority",
            "dispatch_authority",
            "output_authority",
        ):
            if payload.pop(name) is not False:
                raise ValueError(f"imagined-rollout gauge {name} must remain false")
        planner_payload = payload.pop("planner")
        gauge_payload = payload.pop("gauge")
        if not isinstance(planner_payload, Mapping) or not isinstance(
            gauge_payload,
            Mapping,
        ):
            raise ValueError("imagined-rollout gauge nested configs are missing")
        gauge_fields = {field.name for field in dataclasses.fields(
            ImaginedRolloutSelectionGaugeConfig
        )}
        if set(gauge_payload) != gauge_fields:
            raise ValueError("imagined-rollout gauge threshold fields do not match v1")
        for name in (
            "audit_capacity",
            "n_regions",
            "min_evidence_count",
            "max_authorizations",
        ):
            if type(gauge_payload[name]) is not int:
                raise ValueError(f"serialized {name} must be an integer")
        for name in ("require_success_lcb", "require_top_quantile_purity"):
            if type(gauge_payload[name]) is not bool:
                raise ValueError(f"serialized {name} must be a boolean")
        restored = cls(
            EnsembleShortRolloutPlanner.from_config(planner_payload),
            ImaginedRolloutSelectionGaugeConfig(
                **cast(dict[str, Any], dict(gauge_payload))
            ),
        )
        if restored.to_config() != dict(config):
            raise ValueError("imagined-rollout gauge config is not canonical")
        return restored

    def _zero_batch(self) -> ImaginedRolloutBatch:
        budget = self._rollout_budget
        horizon = self._rollout_horizon
        observation_dim = self._observation_dim
        bh = (budget, horizon)
        revisions = jnp.zeros((budget, 2), dtype=jnp.uint32)
        tags = jnp.zeros((budget,), dtype=jnp.uint32)
        return ImaginedRolloutBatch(
            observations=jnp.zeros((*bh, observation_dim), dtype=jnp.float32),
            actions=jnp.zeros(bh, dtype=jnp.int32),
            rewards=jnp.zeros(bh, dtype=jnp.float32),
            continuations=jnp.zeros(bh, dtype=jnp.float32),
            next_observations=jnp.zeros(
                (*bh, observation_dim),
                dtype=jnp.float32,
            ),
            return_targets=jnp.zeros(bh, dtype=jnp.float32),
            bootstrap_values=jnp.zeros((budget,), dtype=jnp.float32),
            root_returns=jnp.zeros((budget,), dtype=jnp.float32),
            transition_valid=jnp.zeros(bh, dtype=jnp.bool_),
            terminated=jnp.zeros(bh, dtype=jnp.bool_),
            path_accepted=jnp.zeros((budget,), dtype=jnp.bool_),
            decision_id_words=revisions,
            source_revision_words=revisions,
            model_revision_words=revisions,
            policy_revision_words=revisions,
            value_revision_words=revisions,
            source_integrity_tags=tags,
            policy_integrity_tags=tags,
            value_integrity_tags=tags,
            authority_integrity_tags=tags,
            model_integrity_tags=tags,
            anchor_integrity_tags=tags,
        )

    def _zero_record(self) -> GroundedRolloutAuditRecord:
        zero_words = jnp.zeros((2,), dtype=jnp.uint32)
        zero_i = jnp.asarray(0, dtype=jnp.int32)
        zero_f = jnp.asarray(0.0, dtype=jnp.float32)
        zero_b = jnp.asarray(False, dtype=jnp.bool_)
        zero_u = jnp.asarray(0, dtype=jnp.uint32)
        return GroundedRolloutAuditRecord(
            proposal_content_tag=zero_u,
            transition_content_tag=zero_u,
            rollout_index=zero_i,
            step_index=zero_i,
            action=zero_i,
            region_id=zero_i,
            record_id_words=zero_words,
            source_revision_words=zero_words,
            model_revision_words=zero_words,
            source_integrity_tag=zero_u,
            model_integrity_tag=zero_u,
            predicted_reward=zero_f,
            predicted_next_observation=jnp.zeros(
                (self._observation_dim,),
                dtype=jnp.float32,
            ),
            predicted_terminated=zero_b,
            predicted_score=zero_f,
            proposal_batch_valid=zero_b,
            proposal_transition_valid=zero_b,
            realized_valid=zero_b,
            realized_reward=zero_f,
            realized_next_observation=jnp.zeros(
                (self._observation_dim,),
                dtype=jnp.float32,
            ),
            realized_terminated=zero_b,
            realized_success=zero_b,
            record_integrity_tag=zero_u,
        )

    def _calibration_tag(
        self,
        state: ImaginedRolloutSelectionGaugeState,
    ) -> UInt[Array, ""]:
        payload = (
            self._planner_fingerprint,
            state.bound_source_revision_words,
            state.bound_model_revision_words,
            state.bound_source_integrity_tag,
            state.bound_model_integrity_tag,
            state.record_valid,
            state.record_id_words,
            state.record_proposal_tags,
            state.record_transition_tags,
            state.record_rollout_indices,
            state.record_step_indices,
            state.record_actions,
            state.record_regions,
            state.record_predicted_rewards,
            state.record_realized_rewards,
            state.record_predicted_next_observations,
            state.record_realized_next_observations,
            state.record_predicted_terminated,
            state.record_realized_terminated,
            state.record_realized_valid,
            state.record_realized_success,
            state.record_scores,
            state.record_integrity_tags,
            state.evidence_counts,
            state.realized_valid_counts,
            state.reward_absolute_error_sums,
            state.next_state_squared_error_sums,
            state.termination_correct_counts,
            state.success_counts,
            state.top_quantile_counts,
            state.top_quantile_purity,
            state.last_record_id_words,
            state.calibration_revision_words,
        )
        return _mix_words(
            _tree_content_words(payload),
            salt=_CALIBRATION_TAG_SALT,
        )

    def _state_tag(
        self,
        state: ImaginedRolloutSelectionGaugeState,
    ) -> UInt[Array, ""]:
        return _mix_words(
            jnp.concatenate(
                (
                    self._planner_fingerprint,
                    jnp.reshape(state.calibration_content_tag, (1,)),
                    state.calibration_revision_words,
                    state.authorization_count_words,
                )
            ),
            salt=_STATE_TAG_SALT,
        )

    def _seal_state(
        self,
        state: ImaginedRolloutSelectionGaugeState,
    ) -> ImaginedRolloutSelectionGaugeState:
        calibration_tag = self._calibration_tag(state)
        with_calibration = cast(
            ImaginedRolloutSelectionGaugeState,
            cast(Any, state).replace(calibration_content_tag=calibration_tag),
        )
        return cast(
            ImaginedRolloutSelectionGaugeState,
            cast(Any, with_calibration).replace(
                state_integrity_tag=self._state_tag(with_calibration)
            ),
        )

    def _empty_state(self) -> ImaginedRolloutSelectionGaugeState:
        capacity = self._config.audit_capacity
        cells = (self._n_actions, self._config.n_regions)
        zero_words = jnp.zeros((2,), dtype=jnp.uint32)
        state = ImaginedRolloutSelectionGaugeState(
            bound_source_revision_words=zero_words,
            bound_model_revision_words=zero_words,
            bound_source_integrity_tag=jnp.asarray(0, dtype=jnp.uint32),
            bound_model_integrity_tag=jnp.asarray(0, dtype=jnp.uint32),
            record_valid=jnp.zeros((capacity,), dtype=jnp.bool_),
            record_id_words=jnp.zeros((capacity, 2), dtype=jnp.uint32),
            record_proposal_tags=jnp.zeros((capacity,), dtype=jnp.uint32),
            record_transition_tags=jnp.zeros((capacity,), dtype=jnp.uint32),
            record_rollout_indices=jnp.zeros((capacity,), dtype=jnp.int32),
            record_step_indices=jnp.zeros((capacity,), dtype=jnp.int32),
            record_actions=jnp.zeros((capacity,), dtype=jnp.int32),
            record_regions=jnp.zeros((capacity,), dtype=jnp.int32),
            record_predicted_rewards=jnp.zeros((capacity,), dtype=jnp.float32),
            record_realized_rewards=jnp.zeros((capacity,), dtype=jnp.float32),
            record_predicted_next_observations=jnp.zeros(
                (capacity, self._observation_dim),
                dtype=jnp.float32,
            ),
            record_realized_next_observations=jnp.zeros(
                (capacity, self._observation_dim),
                dtype=jnp.float32,
            ),
            record_predicted_terminated=jnp.zeros((capacity,), dtype=jnp.bool_),
            record_realized_terminated=jnp.zeros((capacity,), dtype=jnp.bool_),
            record_realized_valid=jnp.zeros((capacity,), dtype=jnp.bool_),
            record_realized_success=jnp.zeros((capacity,), dtype=jnp.bool_),
            record_scores=jnp.zeros((capacity,), dtype=jnp.float32),
            record_integrity_tags=jnp.zeros((capacity,), dtype=jnp.uint32),
            evidence_counts=jnp.zeros(cells, dtype=jnp.int32),
            realized_valid_counts=jnp.zeros(cells, dtype=jnp.int32),
            reward_absolute_error_sums=jnp.zeros(cells, dtype=jnp.float32),
            next_state_squared_error_sums=jnp.zeros(cells, dtype=jnp.float32),
            termination_correct_counts=jnp.zeros(cells, dtype=jnp.int32),
            success_counts=jnp.zeros(cells, dtype=jnp.int32),
            top_quantile_counts=jnp.zeros(cells, dtype=jnp.int32),
            top_quantile_purity=jnp.zeros(cells, dtype=jnp.float32),
            last_record_id_words=zero_words,
            calibration_revision_words=zero_words,
            authorization_count_words=zero_words,
            calibration_content_tag=jnp.asarray(0, dtype=jnp.uint32),
            state_integrity_tag=jnp.asarray(0, dtype=jnp.uint32),
        )
        return self._seal_state(state)

    def _zero_receipt(self) -> ImaginedRolloutAuthorizationReceipt:
        bh = (self._rollout_budget, self._rollout_horizon)
        zeros_f = jnp.zeros(bh, dtype=jnp.float32)
        zeros_b = jnp.zeros(bh, dtype=jnp.bool_)
        zeros_i = jnp.zeros(bh, dtype=jnp.int32)
        zero_words = jnp.zeros((2,), dtype=jnp.uint32)
        return ImaginedRolloutAuthorizationReceipt(
            planner_fingerprint=self._planner_fingerprint,
            proposal_content_tag=jnp.asarray(0, dtype=jnp.uint32),
            calibration_revision_words=zero_words,
            calibration_content_tag=jnp.asarray(0, dtype=jnp.uint32),
            authorization_words=zero_words,
            source_revision_words=zero_words,
            model_revision_words=zero_words,
            source_integrity_tag=jnp.asarray(0, dtype=jnp.uint32),
            model_integrity_tag=jnp.asarray(0, dtype=jnp.uint32),
            region_ids=zeros_i,
            safety_admitted=zeros_b,
            protected=zeros_b,
            evidence_counts=zeros_i,
            realized_valid_fractions=zeros_f,
            mean_absolute_reward_errors=zeros_f,
            root_mean_square_next_state_errors=zeros_f,
            termination_accuracies=zeros_f,
            success_lcbs=zeros_f,
            top_quantile_purities=zeros_f,
            evidence_count_passed=zeros_b,
            realized_validity_passed=zeros_b,
            reward_error_passed=zeros_b,
            next_state_error_passed=zeros_b,
            termination_passed=zeros_b,
            success_lcb_passed=zeros_b,
            top_quantile_purity_passed=zeros_b,
            evidence_floor_passed=zeros_b,
            transition_authorized=zeros_b,
            authorized=jnp.asarray(False),
            receipt_integrity_tag=jnp.asarray(0, dtype=jnp.uint32),
        )

    def _batch_static_valid(self, batch: object) -> bool:
        return (
            isinstance(batch, ImaginedRolloutBatch)
            and _tree_static_signature(batch) == self._batch_signature
        )

    def _state_static_valid(self, state: object) -> bool:
        return (
            isinstance(state, ImaginedRolloutSelectionGaugeState)
            and _tree_static_signature(state) == self._state_signature
        )

    def _record_static_valid(self, record: object) -> bool:
        return (
            isinstance(record, GroundedRolloutAuditRecord)
            and _tree_static_signature(record) == self._record_signature
        )

    def _receipt_static_valid(self, receipt: object) -> bool:
        return (
            isinstance(receipt, ImaginedRolloutAuthorizationReceipt)
            and _tree_static_signature(receipt) == self._receipt_signature
        )

    def proposal_content_tag(self, batch: ImaginedRolloutBatch) -> UInt[Array, ""]:
        """Bind every fixed-shape proposal array word, revision, and tag."""

        if not self._batch_static_valid(batch):
            raise TypeError("batch does not match this gauge's exact planner shape")
        return _mix_words(
            _tree_content_words(batch),
            salt=_PROPOSAL_TAG_SALT,
        )

    def _transition_content_tag(
        self,
        batch: ImaginedRolloutBatch,
        rollout_index: Array | int,
        step_index: Array | int,
    ) -> UInt[Array, ""]:
        """Bind the training-relevant content of one proposal transition.

        Batch-level summaries and slot coordinates are deliberately absent.  A
        caller therefore cannot turn an audited transition into a nominally new
        candidate by editing an unused bootstrap/root summary or moving the same
        transition to another slot.
        """

        payload = (
            self._planner_fingerprint,
            batch.source_revision_words[rollout_index],
            batch.model_revision_words[rollout_index],
            batch.source_integrity_tags[rollout_index],
            batch.model_integrity_tags[rollout_index],
            batch.observations[rollout_index, step_index],
            batch.actions[rollout_index, step_index],
            batch.rewards[rollout_index, step_index],
            batch.continuations[rollout_index, step_index],
            batch.next_observations[rollout_index, step_index],
            batch.return_targets[rollout_index, step_index],
            batch.terminated[rollout_index, step_index],
        )
        return _mix_words(
            _tree_content_words(payload),
            salt=_TRANSITION_TAG_SALT,
        )

    def _batch_transition_content_tags(
        self,
        batch: ImaginedRolloutBatch,
    ) -> UInt[Array, "rollout_budget rollout_horizon"]:
        return jnp.stack(
            tuple(
                jnp.stack(
                    tuple(
                        self._transition_content_tag(batch, rollout_index, step_index)
                        for step_index in range(self._rollout_horizon)
                    )
                )
                for rollout_index in range(self._rollout_budget)
            )
        )

    def _batch_terminal_semantics_valid(
        self,
        batch: ImaginedRolloutBatch,
    ) -> Bool[Array, ""]:
        return _terminal_path_semantics_valid(
            batch.rewards,
            batch.continuations,
            batch.return_targets,
            batch.terminated,
            batch.transition_valid,
        )

    def _batch_valid(self, batch: ImaginedRolloutBatch) -> Bool[Array, ""]:
        transition_valid = batch.transition_valid
        action_valid = (
            (batch.actions >= 0) & (batch.actions < self._n_actions)
        )
        paths_broadcast = jnp.broadcast_to(
            batch.path_accepted[:, None],
            transition_valid.shape,
        )
        identity_nonzero = (
            jnp.all(jnp.any(batch.decision_id_words != 0, axis=1))
            & jnp.all(jnp.any(batch.source_revision_words != 0, axis=1))
            & jnp.all(batch.source_integrity_tags != 0)
            & jnp.all(batch.model_integrity_tags != 0)
            & jnp.all(batch.policy_integrity_tags != 0)
            & jnp.all(batch.value_integrity_tags != 0)
            & jnp.all(batch.authority_integrity_tags != 0)
            & jnp.all(batch.anchor_integrity_tags != 0)
        )
        generation_consistent = (
            jnp.all(batch.source_revision_words == batch.source_revision_words[0])
            & jnp.all(batch.model_revision_words == batch.model_revision_words[0])
            & jnp.all(batch.source_integrity_tags == batch.source_integrity_tags[0])
            & jnp.all(batch.model_integrity_tags == batch.model_integrity_tags[0])
        )
        finite = _tree_finite(batch)
        return (
            finite
            & jnp.any(transition_valid)
            & jnp.all(~transition_valid | paths_broadcast)
            & jnp.all(~transition_valid | action_valid)
            & identity_nonzero
            & generation_consistent
            & self._batch_terminal_semantics_valid(batch)
        )

    def _generation_matches_batch(
        self,
        state: ImaginedRolloutSelectionGaugeState,
        batch: ImaginedRolloutBatch,
    ) -> Bool[Array, ""]:
        return (
            jnp.all(batch.source_revision_words == state.bound_source_revision_words)
            & jnp.all(batch.model_revision_words == state.bound_model_revision_words)
            & jnp.all(
                batch.source_integrity_tags == state.bound_source_integrity_tag
            )
            & jnp.all(batch.model_integrity_tags == state.bound_model_integrity_tag)
        )

    def _state_valid(
        self,
        state: ImaginedRolloutSelectionGaugeState,
    ) -> Bool[Array, ""]:
        record_count = jnp.sum(state.record_valid.astype(jnp.int32))
        exact_record_clock = (
            state.calibration_revision_words[0] == 0
        ) & (
            state.calibration_revision_words[1]
            == record_count.astype(jnp.uint32)
        )
        record_prefix = jnp.all(
            state.record_valid
            == (
                jnp.arange(self._config.audit_capacity, dtype=jnp.int32)
                < record_count
            )
        )
        return (
            _tree_finite(state)
            & _words_nonzero(state.bound_source_revision_words)
            & (state.bound_source_integrity_tag != 0)
            & (state.bound_model_integrity_tag != 0)
            & exact_record_clock
            & record_prefix
            & (record_count <= self._config.audit_capacity)
            & jnp.all(~state.record_valid | (state.record_transition_tags != 0))
            & jnp.all(~state.record_valid | (state.record_integrity_tags != 0))
            & _words_leq_limit(
                state.authorization_count_words,
                self._config.max_authorizations,
            )
            & (state.calibration_content_tag == self._calibration_tag(state))
            & (state.state_integrity_tag == self._state_tag(state))
        )

    def state_valid(
        self,
        state: ImaginedRolloutSelectionGaugeState,
    ) -> Bool[Array, ""]:
        if not self._state_static_valid(state):
            raise TypeError("state does not match this gauge's exact state contract")
        return self._state_valid(state)

    def init(
        self,
        frozen_generation_batch: ImaginedRolloutBatch,
    ) -> ImaginedRolloutSelectionGaugeState:
        """Freeze the exact source/model generation without retaining the batch."""

        if not self._batch_static_valid(frozen_generation_batch):
            raise TypeError("frozen generation batch has the wrong static contract")
        if not bool(jax.device_get(self._batch_valid(frozen_generation_batch))):
            raise ValueError("frozen generation batch is dynamically invalid")
        state = cast(
            ImaginedRolloutSelectionGaugeState,
            cast(Any, self._empty_state()).replace(
                bound_source_revision_words=(
                    frozen_generation_batch.source_revision_words[0]
                ),
                bound_model_revision_words=(
                    frozen_generation_batch.model_revision_words[0]
                ),
                bound_source_integrity_tag=(
                    frozen_generation_batch.source_integrity_tags[0]
                ),
                bound_model_integrity_tag=(
                    frozen_generation_batch.model_integrity_tags[0]
                ),
            ),
        )
        sealed = self._seal_state(state)
        if not bool(jax.device_get(self._state_valid(sealed))):
            raise ValueError("failed to construct a valid frozen audit state")
        return sealed

    def _record_tag(
        self,
        record: GroundedRolloutAuditRecord,
    ) -> UInt[Array, ""]:
        payload = cast(
            GroundedRolloutAuditRecord,
            cast(Any, record).replace(
                record_integrity_tag=jnp.asarray(0, dtype=jnp.uint32)
            ),
        )
        return _mix_words(
            _tree_content_words((self._planner_fingerprint, payload)),
            salt=_RECORD_TAG_SALT,
        )

    def bind_grounded_record(
        self,
        batch: ImaginedRolloutBatch,
        *,
        rollout_index: Array,
        step_index: Array,
        region_id: Array,
        record_id_words: Array,
        realized_valid: Array,
        realized_reward: Array,
        realized_next_observation: Array,
        realized_terminated: Array,
        realized_success: Array,
    ) -> GroundedRolloutAuditRecord:
        """Bind caller-owned realized fields to one exact proposal transition."""

        if not self._batch_static_valid(batch):
            raise TypeError("batch has the wrong static contract")
        for name, value, shape, dtype in (
            ("rollout_index", rollout_index, (), jnp.int32),
            ("step_index", step_index, (), jnp.int32),
            ("region_id", region_id, (), jnp.int32),
            ("record_id_words", record_id_words, (2,), jnp.uint32),
            ("realized_valid", realized_valid, (), jnp.bool_),
            ("realized_reward", realized_reward, (), jnp.float32),
            (
                "realized_next_observation",
                realized_next_observation,
                (self._observation_dim,),
                jnp.float32,
            ),
            ("realized_terminated", realized_terminated, (), jnp.bool_),
            ("realized_success", realized_success, (), jnp.bool_),
        ):
            _require_array(value, name=name, shape=shape, dtype=dtype)
        return self._bind_grounded_record_jit(
            batch,
            rollout_index,
            step_index,
            region_id,
            record_id_words,
            realized_valid,
            realized_reward,
            realized_next_observation,
            realized_terminated,
            realized_success,
        )

    def _bind_grounded_record_jit(
        self,
        batch: ImaginedRolloutBatch,
        rollout_index: Array,
        step_index: Array,
        region_id: Array,
        record_id_words: Array,
        realized_valid: Array,
        realized_reward: Array,
        realized_next_observation: Array,
        realized_terminated: Array,
        realized_success: Array,
    ) -> GroundedRolloutAuditRecord:
        safe_rollout = jnp.clip(rollout_index, 0, self._rollout_budget - 1)
        safe_step = jnp.clip(step_index, 0, self._rollout_horizon - 1)
        provisional = GroundedRolloutAuditRecord(
            proposal_content_tag=self.proposal_content_tag(batch),
            transition_content_tag=self._transition_content_tag(
                batch,
                safe_rollout,
                safe_step,
            ),
            rollout_index=rollout_index,
            step_index=step_index,
            action=batch.actions[safe_rollout, safe_step],
            region_id=region_id,
            record_id_words=record_id_words,
            source_revision_words=batch.source_revision_words[safe_rollout],
            model_revision_words=batch.model_revision_words[safe_rollout],
            source_integrity_tag=batch.source_integrity_tags[safe_rollout],
            model_integrity_tag=batch.model_integrity_tags[safe_rollout],
            predicted_reward=batch.rewards[safe_rollout, safe_step],
            predicted_next_observation=(
                batch.next_observations[safe_rollout, safe_step]
            ),
            predicted_terminated=batch.terminated[safe_rollout, safe_step],
            predicted_score=batch.return_targets[safe_rollout, safe_step],
            proposal_batch_valid=self._batch_valid(batch),
            proposal_transition_valid=batch.transition_valid[safe_rollout, safe_step],
            realized_valid=realized_valid,
            realized_reward=realized_reward,
            realized_next_observation=realized_next_observation,
            realized_terminated=realized_terminated,
            realized_success=realized_success,
            record_integrity_tag=jnp.asarray(0, dtype=jnp.uint32),
        )
        return cast(
            GroundedRolloutAuditRecord,
            cast(Any, provisional).replace(
                record_integrity_tag=self._record_tag(provisional)
            ),
        )

    def _record_valid(
        self,
        record: GroundedRolloutAuditRecord,
    ) -> Bool[Array, ""]:
        return (
            (record.proposal_content_tag != 0)
            & (record.transition_content_tag != 0)
            & (record.rollout_index >= 0)
            & (record.rollout_index < self._rollout_budget)
            & (record.step_index >= 0)
            & (record.step_index < self._rollout_horizon)
            & (record.action >= 0)
            & (record.action < self._n_actions)
            & (record.region_id >= 0)
            & (record.region_id < self._config.n_regions)
            & record.proposal_batch_valid
            & record.proposal_transition_valid
            & _words_nonzero(record.record_id_words)
            & _words_nonzero(record.source_revision_words)
            & (record.source_integrity_tag != 0)
            & (record.model_integrity_tag != 0)
            & _tree_finite(record)
            & (record.record_integrity_tag == self._record_tag(record))
        )

    def _top_quantile_summaries(
        self,
        record_valid: Array,
        actions: Array,
        regions: Array,
        scores: Array,
        realized_valid: Array,
        successes: Array,
    ) -> tuple[Array, Array]:
        counts = jnp.zeros(
            (self._n_actions, self._config.n_regions),
            dtype=jnp.int32,
        )
        purity = jnp.zeros_like(counts, dtype=jnp.float32)
        fraction = jnp.asarray(
            self._config.top_quantile_fraction,
            dtype=jnp.float32,
        )
        capacity = self._config.audit_capacity
        ranks = jnp.arange(capacity, dtype=jnp.int32)
        negative = jnp.asarray(-_FLOAT32_MAX, dtype=jnp.float32)
        for action in range(self._n_actions):
            for region in range(self._config.n_regions):
                cell_mask = (
                    record_valid
                    & (actions == action)
                    & (regions == region)
                )
                cell_count = jnp.sum(cell_mask.astype(jnp.int32))
                top_count = jnp.where(
                    cell_count > 0,
                    jnp.maximum(
                        jnp.asarray(1, dtype=jnp.int32),
                        jnp.ceil(cell_count.astype(jnp.float32) * fraction).astype(
                            jnp.int32
                        ),
                    ),
                    jnp.asarray(0, dtype=jnp.int32),
                )
                order = jnp.argsort(
                    jnp.where(cell_mask, scores, negative),
                    descending=True,
                )
                ordered_success = (
                    successes[order] & realized_valid[order] & cell_mask[order]
                )
                selected = ranks < top_count
                selected_successes = jnp.sum(
                    (ordered_success & selected).astype(jnp.float32)
                )
                safe_count = jnp.maximum(top_count, 1).astype(jnp.float32)
                counts = counts.at[action, region].set(top_count)
                purity = purity.at[action, region].set(
                    jnp.where(top_count > 0, selected_successes / safe_count, 0.0)
                )
        return counts, purity

    def record_grounded_outcome(
        self,
        state: ImaginedRolloutSelectionGaugeState,
        record: GroundedRolloutAuditRecord,
    ) -> GroundedRolloutAuditResult:
        """Atomically append one grounded outcome after exposing prior stats."""

        if not self._state_static_valid(state):
            raise TypeError("state has the wrong static contract")
        if not self._record_static_valid(record):
            raise TypeError("record has the wrong static contract")
        return self._record_grounded_outcome_jit(state, record)

    def _record_grounded_outcome_jit(
        self,
        state: ImaginedRolloutSelectionGaugeState,
        record: GroundedRolloutAuditRecord,
    ) -> GroundedRolloutAuditResult:
        state_valid = self._state_valid(state)
        record_valid = self._record_valid(record)
        generation_matches = (
            jnp.array_equal(
                record.source_revision_words,
                state.bound_source_revision_words,
            )
            & jnp.array_equal(
                record.model_revision_words,
                state.bound_model_revision_words,
            )
            & (record.source_integrity_tag == state.bound_source_integrity_tag)
            & (record.model_integrity_tag == state.bound_model_integrity_tag)
        )
        identity_fresh = _words_less(
            state.last_record_id_words,
            record.record_id_words,
        )
        proposal_slot_fresh = ~jnp.any(
            state.record_valid
            & (state.record_transition_tags == record.transition_content_tag)
        )
        record_index = state.calibration_revision_words[1].astype(jnp.int32)
        capacity_available = (
            state.calibration_revision_words[0] == 0
        ) & (record_index < self._config.audit_capacity)
        safe_action = jnp.clip(record.action, 0, self._n_actions - 1)
        safe_region = jnp.clip(record.region_id, 0, self._config.n_regions - 1)
        pre_count = state.evidence_counts[safe_action, safe_region]
        pre_valid_count = state.realized_valid_counts[safe_action, safe_region]
        safe_pre_count = jnp.maximum(pre_count, 1).astype(jnp.float32)
        pre_reward_error = (
            state.reward_absolute_error_sums[safe_action, safe_region]
            / safe_pre_count
        )
        pre_next_error = jnp.sqrt(
            state.next_state_squared_error_sums[safe_action, safe_region]
            / safe_pre_count
        )
        pre_termination = (
            state.termination_correct_counts[safe_action, safe_region].astype(
                jnp.float32
            )
            / safe_pre_count
        )
        pre_success = state.success_counts[safe_action, safe_region]
        pre_purity = state.top_quantile_purity[safe_action, safe_region]
        proposed_revision, revision_capacity = _checked_words_add_small(
            state.calibration_revision_words,
            1,
        )
        applied_pre = (
            state_valid
            & record_valid
            & generation_matches
            & identity_fresh
            & proposal_slot_fresh
            & capacity_available
            & revision_capacity
        )
        safe_index = jnp.clip(record_index, 0, self._config.audit_capacity - 1)
        record_valid_array = state.record_valid.at[safe_index].set(True)
        record_id_words = state.record_id_words.at[safe_index].set(
            record.record_id_words
        )
        record_proposal_tags = state.record_proposal_tags.at[safe_index].set(
            record.proposal_content_tag
        )
        record_transition_tags = state.record_transition_tags.at[safe_index].set(
            record.transition_content_tag
        )
        record_rollout_indices = state.record_rollout_indices.at[safe_index].set(
            record.rollout_index
        )
        record_step_indices = state.record_step_indices.at[safe_index].set(
            record.step_index
        )
        record_actions = state.record_actions.at[safe_index].set(record.action)
        record_regions = state.record_regions.at[safe_index].set(record.region_id)
        predicted_rewards = state.record_predicted_rewards.at[safe_index].set(
            record.predicted_reward
        )
        realized_rewards = state.record_realized_rewards.at[safe_index].set(
            record.realized_reward
        )
        predicted_next = state.record_predicted_next_observations.at[safe_index].set(
            record.predicted_next_observation
        )
        realized_next = state.record_realized_next_observations.at[safe_index].set(
            record.realized_next_observation
        )
        predicted_terminated = state.record_predicted_terminated.at[safe_index].set(
            record.predicted_terminated
        )
        realized_terminated = state.record_realized_terminated.at[safe_index].set(
            record.realized_terminated
        )
        realized_valid = state.record_realized_valid.at[safe_index].set(
            record.realized_valid
        )
        realized_success = state.record_realized_success.at[safe_index].set(
            record.realized_success
        )
        scores = state.record_scores.at[safe_index].set(record.predicted_score)
        record_tags = state.record_integrity_tags.at[safe_index].set(
            record.record_integrity_tag
        )
        evidence_counts = state.evidence_counts.at[safe_action, safe_region].add(1)
        valid_counts = state.realized_valid_counts.at[safe_action, safe_region].add(
            record.realized_valid.astype(jnp.int32)
        )
        reward_error = jnp.abs(record.predicted_reward - record.realized_reward)
        next_error = jnp.mean(
            jnp.square(
                record.predicted_next_observation
                - record.realized_next_observation
            )
        )
        reward_sums = state.reward_absolute_error_sums.at[
            safe_action, safe_region
        ].add(reward_error)
        next_sums = state.next_state_squared_error_sums.at[
            safe_action, safe_region
        ].add(next_error)
        termination_counts = state.termination_correct_counts.at[
            safe_action, safe_region
        ].add(
            (
                record.realized_valid
                & (record.predicted_terminated == record.realized_terminated)
            ).astype(jnp.int32)
        )
        success_counts = state.success_counts.at[safe_action, safe_region].add(
            (record.realized_valid & record.realized_success).astype(jnp.int32)
        )
        top_counts, top_purity = self._top_quantile_summaries(
            record_valid_array,
            record_actions,
            record_regions,
            scores,
            realized_valid,
            realized_success,
        )
        candidate = ImaginedRolloutSelectionGaugeState(
            bound_source_revision_words=state.bound_source_revision_words,
            bound_model_revision_words=state.bound_model_revision_words,
            bound_source_integrity_tag=state.bound_source_integrity_tag,
            bound_model_integrity_tag=state.bound_model_integrity_tag,
            record_valid=record_valid_array,
            record_id_words=record_id_words,
            record_proposal_tags=record_proposal_tags,
            record_transition_tags=record_transition_tags,
            record_rollout_indices=record_rollout_indices,
            record_step_indices=record_step_indices,
            record_actions=record_actions,
            record_regions=record_regions,
            record_predicted_rewards=predicted_rewards,
            record_realized_rewards=realized_rewards,
            record_predicted_next_observations=predicted_next,
            record_realized_next_observations=realized_next,
            record_predicted_terminated=predicted_terminated,
            record_realized_terminated=realized_terminated,
            record_realized_valid=realized_valid,
            record_realized_success=realized_success,
            record_scores=scores,
            record_integrity_tags=record_tags,
            evidence_counts=evidence_counts,
            realized_valid_counts=valid_counts,
            reward_absolute_error_sums=reward_sums,
            next_state_squared_error_sums=next_sums,
            termination_correct_counts=termination_counts,
            success_counts=success_counts,
            top_quantile_counts=top_counts,
            top_quantile_purity=top_purity,
            last_record_id_words=record.record_id_words,
            calibration_revision_words=proposed_revision,
            authorization_count_words=state.authorization_count_words,
            calibration_content_tag=jnp.asarray(0, dtype=jnp.uint32),
            state_integrity_tag=jnp.asarray(0, dtype=jnp.uint32),
        )
        candidate = self._seal_state(candidate)
        applied = applied_pre & self._state_valid(candidate)
        next_state = cast(
            ImaginedRolloutSelectionGaugeState,
            jax.lax.cond(applied, lambda: candidate, lambda: state),
        )
        return GroundedRolloutAuditResult(
            state=next_state,
            diagnostics=GroundedRolloutAuditDiagnostics(
                state_valid=state_valid,
                record_valid=record_valid,
                generation_matches=generation_matches,
                record_identity_fresh=identity_fresh,
                proposal_slot_fresh=proposal_slot_fresh,
                capacity_available=capacity_available & revision_capacity,
                pre_evidence_count=pre_count,
                pre_realized_valid_count=pre_valid_count,
                pre_mean_absolute_reward_error=pre_reward_error,
                pre_root_mean_square_next_state_error=pre_next_error,
                pre_termination_accuracy=pre_termination,
                pre_success_count=pre_success,
                pre_top_quantile_purity=pre_purity,
                applied=applied,
                pre_calibration_revision_words=state.calibration_revision_words,
                post_calibration_revision_words=next_state.calibration_revision_words,
            ),
        )

    def _candidate_separated(
        self,
        state: ImaginedRolloutSelectionGaugeState,
        batch: ImaginedRolloutBatch,
    ) -> Bool[Array, ""]:
        transition_tags = self._batch_transition_content_tags(batch)
        audited_match = jnp.any(
            (transition_tags[..., None] == state.record_transition_tags[None, None, :])
            & state.record_valid[None, None, :],
            axis=-1,
        )
        return jnp.all(~batch.transition_valid | ~audited_match)

    def _wilson_success_lcb(self, successes: Array, counts: Array) -> Array:
        count = counts.astype(jnp.float32)
        safe_count = jnp.maximum(count, 1.0)
        probability = successes.astype(jnp.float32) / safe_count
        z = jnp.asarray(self._config.success_lcb_z, dtype=jnp.float32)
        z_squared = z * z
        denominator = 1.0 + z_squared / safe_count
        center = probability + z_squared / (2.0 * safe_count)
        radius = z * jnp.sqrt(
            jnp.maximum(
                probability * (1.0 - probability) / safe_count
                + z_squared / (4.0 * safe_count * safe_count),
                0.0,
            )
        )
        lower = (center - radius) / denominator
        return jnp.where(counts > 0, jnp.maximum(lower, 0.0), 0.0)

    def _receipt_tag(
        self,
        receipt: ImaginedRolloutAuthorizationReceipt,
    ) -> UInt[Array, ""]:
        payload = cast(
            ImaginedRolloutAuthorizationReceipt,
            cast(Any, receipt).replace(
                receipt_integrity_tag=jnp.asarray(0, dtype=jnp.uint32)
            ),
        )
        return _mix_words(
            _tree_content_words(payload),
            salt=_RECEIPT_TAG_SALT,
        )

    def _base_candidate_valid(
        self,
        state: ImaginedRolloutSelectionGaugeState,
        batch: ImaginedRolloutBatch,
        region_ids: Array,
    ) -> tuple[Array, Array, Array, Array, Array]:
        state_valid = self._state_valid(state)
        batch_valid = self._batch_valid(batch)
        generation_matches = self._generation_matches_batch(state, batch)
        separated = self._candidate_separated(state, batch)
        region_valid = jnp.all(
            (~batch.transition_valid)
            | ((region_ids >= 0) & (region_ids < self._config.n_regions))
        )
        terminal_valid = self._batch_terminal_semantics_valid(batch)
        base_valid = (
            state_valid
            & batch_valid
            & generation_matches
            & separated
            & region_valid
            & terminal_valid
        )
        return (
            base_valid,
            generation_matches,
            separated,
            region_valid,
            terminal_valid,
        )

    def _build_receipt(
        self,
        state: ImaginedRolloutSelectionGaugeState,
        batch: ImaginedRolloutBatch,
        region_ids: Array,
        safety_admitted: Array,
        protected: Array,
        authorization_words: Array,
        base_valid: Array,
    ) -> ImaginedRolloutAuthorizationReceipt:
        proposal_tag = self.proposal_content_tag(batch)
        safe_actions = jnp.clip(batch.actions, 0, self._n_actions - 1)
        safe_regions = jnp.clip(region_ids, 0, self._config.n_regions - 1)
        evidence_counts = state.evidence_counts[safe_actions, safe_regions]
        valid_counts = state.realized_valid_counts[safe_actions, safe_regions]
        safe_counts = jnp.maximum(evidence_counts, 1).astype(jnp.float32)
        valid_fraction = valid_counts.astype(jnp.float32) / safe_counts
        reward_errors = (
            state.reward_absolute_error_sums[safe_actions, safe_regions]
            / safe_counts
        )
        next_errors = jnp.sqrt(
            state.next_state_squared_error_sums[safe_actions, safe_regions]
            / safe_counts
        )
        termination_accuracy = (
            state.termination_correct_counts[safe_actions, safe_regions].astype(
                jnp.float32
            )
            / safe_counts
        )
        success_counts = state.success_counts[safe_actions, safe_regions]
        success_lcb = self._wilson_success_lcb(success_counts, evidence_counts)
        top_purity = state.top_quantile_purity[safe_actions, safe_regions]
        evidence_pass = evidence_counts >= self._config.min_evidence_count
        validity_pass = (
            valid_fraction
            >= jnp.asarray(
                self._config.min_realized_valid_fraction,
                dtype=jnp.float32,
            )
        )
        reward_pass = (
            reward_errors
            <= jnp.asarray(
                self._config.max_mean_abs_reward_error,
                dtype=jnp.float32,
            )
        )
        next_pass = (
            next_errors
            <= jnp.asarray(
                self._config.max_root_mean_square_next_observation_error,
                dtype=jnp.float32,
            )
        )
        termination_pass = (
            termination_accuracy
            >= jnp.asarray(
                self._config.min_termination_accuracy,
                dtype=jnp.float32,
            )
        )
        success_pass = (
            success_lcb
            >= jnp.asarray(self._config.min_success_lcb, dtype=jnp.float32)
        )
        if not self._config.require_success_lcb:
            success_pass = jnp.ones_like(success_pass)
        purity_pass = (
            top_purity
            >= jnp.asarray(
                self._config.min_top_quantile_purity,
                dtype=jnp.float32,
            )
        )
        if not self._config.require_top_quantile_purity:
            purity_pass = jnp.ones_like(purity_pass)
        floor_pass = (
            evidence_pass
            & validity_pass
            & reward_pass
            & next_pass
            & termination_pass
            & success_pass
            & purity_pass
        )
        accepted_paths = jnp.broadcast_to(
            batch.path_accepted[:, None],
            batch.transition_valid.shape,
        )
        local_authorized = (
            base_valid
            & batch.transition_valid
            & accepted_paths
            & safety_admitted
            & ~protected
            & floor_pass
        )
        transition_authorized = _prefix_closed_transition_mask(
            batch.transition_valid,
            local_authorized,
        )
        provisional = ImaginedRolloutAuthorizationReceipt(
            planner_fingerprint=self._planner_fingerprint,
            proposal_content_tag=proposal_tag,
            calibration_revision_words=state.calibration_revision_words,
            calibration_content_tag=state.calibration_content_tag,
            authorization_words=authorization_words,
            source_revision_words=state.bound_source_revision_words,
            model_revision_words=state.bound_model_revision_words,
            source_integrity_tag=state.bound_source_integrity_tag,
            model_integrity_tag=state.bound_model_integrity_tag,
            region_ids=region_ids,
            safety_admitted=safety_admitted,
            protected=protected,
            evidence_counts=evidence_counts,
            realized_valid_fractions=valid_fraction,
            mean_absolute_reward_errors=reward_errors,
            root_mean_square_next_state_errors=next_errors,
            termination_accuracies=termination_accuracy,
            success_lcbs=success_lcb,
            top_quantile_purities=top_purity,
            evidence_count_passed=evidence_pass,
            realized_validity_passed=validity_pass,
            reward_error_passed=reward_pass,
            next_state_error_passed=next_pass,
            termination_passed=termination_pass,
            success_lcb_passed=success_pass,
            top_quantile_purity_passed=purity_pass,
            evidence_floor_passed=floor_pass,
            transition_authorized=transition_authorized,
            authorized=jnp.any(transition_authorized),
            receipt_integrity_tag=jnp.asarray(0, dtype=jnp.uint32),
        )
        return cast(
            ImaginedRolloutAuthorizationReceipt,
            cast(Any, provisional).replace(
                receipt_integrity_tag=self._receipt_tag(provisional)
            ),
        )

    def authorize(
        self,
        state: ImaginedRolloutSelectionGaugeState,
        batch: ImaginedRolloutBatch,
        *,
        region_ids: Array,
        safety_admitted: Array,
        protected: Array,
    ) -> ImaginedRolloutAuthorizationResult:
        """Authorize against prior audit state; never ingest candidate outcomes."""

        if not self._state_static_valid(state):
            raise TypeError("state has the wrong static contract")
        if not self._batch_static_valid(batch):
            raise TypeError("batch has the wrong static contract")
        shape = (self._rollout_budget, self._rollout_horizon)
        _require_array(
            region_ids,
            name="region_ids",
            shape=shape,
            dtype=jnp.int32,
        )
        _require_array(
            safety_admitted,
            name="safety_admitted",
            shape=shape,
            dtype=jnp.bool_,
        )
        _require_array(
            protected,
            name="protected",
            shape=shape,
            dtype=jnp.bool_,
        )
        return self._authorize_jit(
            state,
            batch,
            region_ids,
            safety_admitted,
            protected,
        )

    def _authorize_jit(
        self,
        state: ImaginedRolloutSelectionGaugeState,
        batch: ImaginedRolloutBatch,
        region_ids: Array,
        safety_admitted: Array,
        protected: Array,
    ) -> ImaginedRolloutAuthorizationResult:
        (
            base_valid,
            generation_matches,
            separated,
            region_valid,
            terminal_valid,
        ) = self._base_candidate_valid(state, batch, region_ids)
        proposed_words, word_capacity = _checked_words_add_small(
            state.authorization_count_words,
            1,
        )
        configured_capacity = _words_leq_limit(
            proposed_words,
            self._config.max_authorizations,
        )
        capacity = word_capacity & configured_capacity
        transaction_applied = base_valid & capacity
        candidate_unsealed = cast(
            ImaginedRolloutSelectionGaugeState,
            cast(Any, state).replace(
                authorization_count_words=proposed_words,
                state_integrity_tag=jnp.asarray(0, dtype=jnp.uint32),
            ),
        )
        candidate = cast(
            ImaginedRolloutSelectionGaugeState,
            cast(Any, candidate_unsealed).replace(
                state_integrity_tag=self._state_tag(candidate_unsealed)
            ),
        )
        transaction_applied = transaction_applied & self._state_valid(candidate)
        next_state = cast(
            ImaginedRolloutSelectionGaugeState,
            jax.lax.cond(transaction_applied, lambda: candidate, lambda: state),
        )
        receipt = self._build_receipt(
            next_state,
            batch,
            region_ids,
            safety_admitted,
            protected,
            next_state.authorization_count_words,
            base_valid & transaction_applied,
        )
        receipt_valid = self._receipt_valid_dynamic(next_state, batch, receipt)
        return ImaginedRolloutAuthorizationResult(
            state=next_state,
            receipt=receipt,
            diagnostics=ImaginedRolloutAuthorizationDiagnostics(
                state_valid=self._state_valid(state),
                batch_valid=self._batch_valid(batch),
                generation_matches=generation_matches,
                audit_candidate_separated=separated,
                region_ids_valid=region_valid,
                terminal_semantics_valid=terminal_valid,
                authorization_capacity_available=capacity,
                post_mint_content_integrity_only=jnp.asarray(True),
                planner_issuance_authenticated=jnp.asarray(False),
                transaction_applied=transaction_applied,
                receipt_valid=receipt_valid,
                pre_authorization_words=state.authorization_count_words,
                post_authorization_words=next_state.authorization_count_words,
            ),
        )

    def _receipt_valid_dynamic(
        self,
        state: ImaginedRolloutSelectionGaugeState,
        batch: ImaginedRolloutBatch,
        receipt: ImaginedRolloutAuthorizationReceipt,
    ) -> Bool[Array, ""]:
        proposal_tag = self.proposal_content_tag(batch)
        base_valid, _, _, _, _ = self._base_candidate_valid(
            state,
            batch,
            receipt.region_ids,
        )
        expected = self._build_receipt(
            state,
            batch,
            receipt.region_ids,
            receipt.safety_admitted,
            receipt.protected,
            receipt.authorization_words,
            base_valid,
        )
        return (
            self._state_valid(state)
            & (receipt.proposal_content_tag == proposal_tag)
            & jnp.array_equal(receipt.planner_fingerprint, self._planner_fingerprint)
            & jnp.array_equal(
                receipt.calibration_revision_words,
                state.calibration_revision_words,
            )
            & (receipt.calibration_content_tag == state.calibration_content_tag)
            & jnp.array_equal(
                receipt.authorization_words,
                state.authorization_count_words,
            )
            & _words_nonzero(receipt.authorization_words)
            & (receipt.receipt_integrity_tag == self._receipt_tag(receipt))
            & _tree_equal(receipt, expected)
        )

    def receipt_valid(
        self,
        state: ImaginedRolloutSelectionGaugeState,
        batch: ImaginedRolloutBatch,
        receipt: ImaginedRolloutAuthorizationReceipt,
    ) -> Bool[Array, ""]:
        """Recompute a current receipt; later audit/authorization makes it stale."""

        if not self._state_static_valid(state):
            raise TypeError("state has the wrong static contract")
        if not self._batch_static_valid(batch):
            raise TypeError("batch has the wrong static contract")
        if not self._receipt_static_valid(receipt):
            return jnp.asarray(False, dtype=jnp.bool_)
        return self._receipt_valid_dynamic(state, batch, receipt)

    @property
    def resource_budget(self) -> ImaginedRolloutSelectionGaugeResourceBudget:
        state = self._empty_state()
        receipt = self._zero_receipt()
        state_scalars, state_bytes = _logical_tree_size(state)
        receipt_scalars, receipt_bytes = _logical_tree_size(receipt)
        return ImaginedRolloutSelectionGaugeResourceBudget(
            persistent_state_scalars=state_scalars,
            persistent_state_bytes=state_bytes,
            receipt_scalars=receipt_scalars,
            receipt_bytes=receipt_bytes,
            max_grounded_records=self._config.audit_capacity,
            max_authorizations=self._config.max_authorizations,
            per_action_region_cells=self._n_actions * self._config.n_regions,
            planner_or_model_state_owned=0,
            dispatch_authority=0,
            safety_authority=0,
            output_authority=0,
            scientific_promotion_allowed=False,
        )


@dataclasses.dataclass(frozen=True)
class ImaginedRolloutActorCriticConfig:
    """Bounded optimizer settings shared by dream and competent-real modes."""

    actor_step_size: float = 0.01
    critic_step_size: float = 0.01
    momentum_decay: float = 0.9
    gradient_clip: float = 10.0
    initialization_scale: float = 0.05
    max_positive_advantage: float = 1_000_000.0
    max_update_calls: int = _INT32_MAX
    max_backward_transitions: int = _INT32_MAX

    def __post_init__(self) -> None:
        _positive_int(
            self.max_update_calls,
            name="max_update_calls",
            maximum=_MAX_UPDATES,
        )
        _positive_int(
            self.max_backward_transitions,
            name="max_backward_transitions",
            maximum=_INT32_MAX,
        )
        specs = (
            ("actor_step_size", _FLOAT32_TINY, None, True),
            ("critic_step_size", _FLOAT32_TINY, None, True),
            ("momentum_decay", 0.0, 1.0, False),
            ("gradient_clip", _FLOAT32_TINY, None, True),
            ("initialization_scale", 0.0, None, False),
            ("max_positive_advantage", _FLOAT32_TINY, None, True),
        )
        for name, minimum, maximum, positive in specs:
            value = _finite_float32(
                getattr(self, name),
                name=name,
                minimum=minimum,
                maximum=maximum,
                strictly_positive=positive,
            )
            object.__setattr__(self, name, value)
        if not float(np.float32(self.momentum_decay)) < 1.0:
            raise ValueError("momentum_decay must remain below one in float32")


@chex.dataclass(frozen=True)
class LinearActorParameters:
    weights: Float[Array, "n_actions observation_dim"]
    bias: Float[Array, " n_actions"]


@chex.dataclass(frozen=True)
class LinearCriticParameters:
    weights: Float[Array, " observation_dim"]
    bias: Float[Array, ""]


@chex.dataclass(frozen=True)
class CompetentRealEpisodeBatch:
    """Caller-owned fixed-shape competent-real cloning control source."""

    observations: Float[Array, "rollout_budget rollout_horizon observation_dim"]
    actions: Int[Array, "rollout_budget rollout_horizon"]
    rewards: Float[Array, "rollout_budget rollout_horizon"]
    continuations: Float[Array, "rollout_budget rollout_horizon"]
    next_observations: Float[
        Array, "rollout_budget rollout_horizon observation_dim"
    ]
    return_targets: Float[Array, "rollout_budget rollout_horizon"]
    terminated: Bool[Array, "rollout_budget rollout_horizon"]
    transition_valid: Bool[Array, "rollout_budget rollout_horizon"]
    competent: Bool[Array, "rollout_budget rollout_horizon"]
    safety_admitted: Bool[Array, "rollout_budget rollout_horizon"]
    protected: Bool[Array, "rollout_budget rollout_horizon"]
    episode_revision_words: UInt[Array, " 2"]
    source_revision_words: UInt[Array, " 2"]
    source_integrity_tag: UInt[Array, ""]
    batch_integrity_tag: UInt[Array, ""]


@chex.dataclass(frozen=True)
class ImaginedRolloutActorCriticState:
    """Learner-owned parameters, momentum, source clocks, and integrity tag."""

    actor_parameters: LinearActorParameters
    critic_parameters: LinearCriticParameters
    actor_momentum: LinearActorParameters
    critic_momentum: LinearCriticParameters
    update_count_words: UInt[Array, " 2"]
    dream_update_count_words: UInt[Array, " 2"]
    real_update_count_words: UInt[Array, " 2"]
    backward_transition_count_words: UInt[Array, " 2"]
    last_dream_authorization_words: UInt[Array, " 2"]
    last_real_episode_revision_words: UInt[Array, " 2"]
    state_integrity_tag: UInt[Array, ""]


@chex.dataclass(frozen=True)
class ImaginedRolloutActorCriticUpdateProposal:
    """Autodiff-free source authorization bound to one learner revision."""

    source_mode: Int[Array, ""]
    source_state_integrity_tag: UInt[Array, ""]
    source_update_count_words: UInt[Array, " 2"]
    source_identity_words: UInt[Array, " 2"]
    source_content_tag: UInt[Array, ""]
    eligible_transition_count: Int[Array, ""]
    terminal_semantics_valid: Bool[Array, ""]
    source_authorized: Bool[Array, ""]
    update_capacity_available: Bool[Array, ""]
    backward_capacity_available: Bool[Array, ""]
    valid: Bool[Array, ""]
    proposal_integrity_tag: UInt[Array, ""]


@chex.dataclass(frozen=True)
class ImaginedRolloutActorCriticCommitTrace:
    """Training channels produced by the sole guarded backward pass."""

    actor_gradient: LinearActorParameters
    critic_gradient: LinearCriticParameters
    actor_momentum_candidate: LinearActorParameters
    critic_momentum_candidate: LinearCriticParameters
    actor_parameter_update: LinearActorParameters
    critic_parameter_update: LinearCriticParameters
    training_mask: Bool[Array, "rollout_budget rollout_horizon"]
    safety_admitted: Bool[Array, "rollout_budget rollout_horizon"]
    protected: Bool[Array, "rollout_budget rollout_horizon"]
    critic_targets: Float[Array, "rollout_budget rollout_horizon"]
    advantages: Float[Array, "rollout_budget rollout_horizon"]
    positive_advantages: Float[Array, "rollout_budget rollout_horizon"]
    imitation_weights: Float[Array, "rollout_budget rollout_horizon"]
    actor_loss: Float[Array, ""]
    critic_loss: Float[Array, ""]
    actor_gradient_norm: Float[Array, ""]
    critic_gradient_norm: Float[Array, ""]
    backward_transition_count: Int[Array, ""]
    backward_work_performed: Bool[Array, ""]
    candidate_finite: Bool[Array, ""]


@chex.dataclass(frozen=True)
class ImaginedRolloutActorCriticCommitDiagnostics:
    state_valid: Bool[Array, ""]
    proposal_integrity_valid: Bool[Array, ""]
    source_matches: Bool[Array, ""]
    source_mode_matches: Bool[Array, ""]
    source_authorized: Bool[Array, ""]
    receipt_or_batch_fresh: Bool[Array, ""]
    update_capacity_available: Bool[Array, ""]
    backward_capacity_available: Bool[Array, ""]
    source_truth_authenticated: Bool[Array, ""]
    preflight_valid: Bool[Array, ""]
    backward_work_performed: Bool[Array, ""]
    autodiff_pass_count: Int[Array, ""]
    backward_transition_count: Int[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    applied: Bool[Array, ""]
    pre_update_count_words: UInt[Array, " 2"]
    post_update_count_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class ImaginedRolloutActorCriticCommitResult:
    state: ImaginedRolloutActorCriticState
    trace: ImaginedRolloutActorCriticCommitTrace
    diagnostics: ImaginedRolloutActorCriticCommitDiagnostics


@dataclasses.dataclass(frozen=True)
class ImaginedRolloutActorCriticResourceBudget:
    persistent_state_scalars: int
    persistent_state_bytes: int
    proposal_scalars: int
    proposal_bytes: int
    commit_trace_scalars: int
    commit_trace_bytes: int
    max_transitions_per_update: int
    max_update_calls: int
    max_backward_transitions: int
    proposal_autodiff_passes: int
    max_autodiff_passes_per_preflight_valid_commit: int
    rejected_preflight_autodiff_passes: int
    backward_clock_counts_accepted_transitions: bool
    discarded_functional_state_can_repeat_pure_calls: bool
    actor_parameter_scalars: int
    critic_parameter_scalars: int
    planner_model_or_gauge_state_owned: int
    dispatch_authority: int
    safety_authority: int
    output_authority: int
    scientific_promotion_allowed: bool

    def to_config(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def _scale_safe_l2_norm(tree: object) -> Float[Array, ""]:
    leaves = [jnp.ravel(jnp.asarray(leaf, dtype=jnp.float32)) for leaf in jax.tree.leaves(tree)]
    vector = jnp.concatenate(leaves) if leaves else jnp.zeros((0,), dtype=jnp.float32)
    maximum = jnp.max(jnp.abs(vector), initial=jnp.asarray(0.0, dtype=jnp.float32))
    mantissa, exponent = jnp.frexp(maximum)
    safe_mantissa = jnp.where(maximum > 0.0, mantissa, 1.0)
    scaled = jnp.ldexp(vector, -exponent) / safe_mantissa
    scaled_norm = jnp.sqrt(jnp.sum(scaled * scaled))
    safe_scaled_norm = jnp.where(scaled_norm > 0.0, scaled_norm, 1.0)
    overflow = maximum > jnp.asarray(_FLOAT32_MAX, dtype=jnp.float32) / safe_scaled_norm
    product = maximum * scaled_norm
    return jnp.where(
        maximum == 0.0,
        0.0,
        jnp.where(overflow, _FLOAT32_MAX, product),
    )


def _clip_tree_l2(tree: object, clip: float) -> tuple[object, Array]:
    norm = _scale_safe_l2_norm(tree)
    clip_array = jnp.asarray(clip, dtype=jnp.float32)
    scale = jnp.where(norm > clip_array, clip_array / jnp.maximum(norm, 1.0), 1.0)
    return jax.tree.map(lambda value: value * scale, tree), norm


class AuthorizedImaginedRolloutActorCritic:
    """Separate bounded learner for authorized dream or competent-real data."""

    DREAM_SOURCE_MODE = 0
    COMPETENT_REAL_SOURCE_MODE = 1

    def __init__(
        self,
        gauge: ImaginedRolloutSelectionGauge,
        config: ImaginedRolloutActorCriticConfig | None = None,
    ) -> None:
        if not isinstance(gauge, ImaginedRolloutSelectionGauge):
            raise TypeError("gauge must be an ImaginedRolloutSelectionGauge")
        self._gauge = gauge
        self._config = config or ImaginedRolloutActorCriticConfig()
        self._max_transition_budget = gauge.rollout_budget * gauge.rollout_horizon
        if self._config.max_backward_transitions < self._max_transition_budget:
            raise ValueError(
                "max_backward_transitions must admit one complete fixed-shape batch"
            )
        self._config_fingerprint = _config_fingerprint(self.to_config())
        zero_state = self._zero_state()
        self._state_signature = _tree_static_signature(zero_state)
        self._real_batch_signature = _tree_static_signature(self._zero_real_batch())
        self._proposal_signature = _tree_static_signature(
            self._zero_proposal(zero_state)
        )

    @property
    def config(self) -> ImaginedRolloutActorCriticConfig:
        return self._config

    @property
    def gauge(self) -> ImaginedRolloutSelectionGauge:
        return self._gauge

    @property
    def max_transition_budget(self) -> int:
        return self._max_transition_budget

    def to_config(self) -> dict[str, object]:
        return {
            "schema": IMAGINED_ROLLOUT_ACTOR_CRITIC_CONFIG_SCHEMA,
            "type": type(self).__name__,
            "mechanism_status": IMAGINED_ROLLOUT_GAUGE_MECHANISM_STATUS,
            "evidence_level": "L0",
            "scientific_promotion_allowed": False,
            "control_benefit_assessed": False,
            "competent_real_truth_authenticated": False,
            "dispatch_authority": False,
            "safety_authority": False,
            "output_authority": False,
            "gauge": self._gauge.to_config(),
            "learner": dataclasses.asdict(self._config),
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> AuthorizedImaginedRolloutActorCritic:
        payload = dict(config)
        expected = {
            "schema",
            "type",
            "mechanism_status",
            "evidence_level",
            "scientific_promotion_allowed",
            "control_benefit_assessed",
            "competent_real_truth_authenticated",
            "dispatch_authority",
            "safety_authority",
            "output_authority",
            "gauge",
            "learner",
        }
        if set(payload) != expected:
            raise ValueError("imagined-rollout actor/critic config fields do not match v1")
        if payload.pop("schema") != IMAGINED_ROLLOUT_ACTOR_CRITIC_CONFIG_SCHEMA:
            raise ValueError("unsupported imagined-rollout actor/critic schema")
        if payload.pop("type") != cls.__name__:
            raise ValueError("unexpected imagined-rollout actor/critic type")
        if payload.pop("mechanism_status") != IMAGINED_ROLLOUT_GAUGE_MECHANISM_STATUS:
            raise ValueError("imagined-rollout actor/critic must remain not assessed")
        if payload.pop("evidence_level") != "L0":
            raise ValueError("imagined-rollout actor/critic must remain L0")
        for name in (
            "scientific_promotion_allowed",
            "control_benefit_assessed",
            "competent_real_truth_authenticated",
            "dispatch_authority",
            "safety_authority",
            "output_authority",
        ):
            if payload.pop(name) is not False:
                raise ValueError(f"imagined-rollout actor/critic {name} must be false")
        gauge_payload = payload.pop("gauge")
        learner_payload = payload.pop("learner")
        if not isinstance(gauge_payload, Mapping) or not isinstance(
            learner_payload,
            Mapping,
        ):
            raise ValueError("imagined-rollout actor/critic nested configs are missing")
        fields = {
            field.name for field in dataclasses.fields(ImaginedRolloutActorCriticConfig)
        }
        if set(learner_payload) != fields:
            raise ValueError("imagined-rollout actor/critic learner fields mismatch")
        restored = cls(
            ImaginedRolloutSelectionGauge.from_config(gauge_payload),
            ImaginedRolloutActorCriticConfig(
                **cast(dict[str, Any], dict(learner_payload))
            ),
        )
        if restored.to_config() != dict(config):
            raise ValueError("imagined-rollout actor/critic config is not canonical")
        return restored

    def _zero_actor(self) -> LinearActorParameters:
        return LinearActorParameters(
            weights=jnp.zeros(
                (self._gauge.n_actions, self._gauge.observation_dim),
                dtype=jnp.float32,
            ),
            bias=jnp.zeros((self._gauge.n_actions,), dtype=jnp.float32),
        )

    def _zero_critic(self) -> LinearCriticParameters:
        return LinearCriticParameters(
            weights=jnp.zeros((self._gauge.observation_dim,), dtype=jnp.float32),
            bias=jnp.asarray(0.0, dtype=jnp.float32),
        )

    def _state_tag(
        self,
        state: ImaginedRolloutActorCriticState,
    ) -> UInt[Array, ""]:
        payload = cast(
            ImaginedRolloutActorCriticState,
            cast(Any, state).replace(
                state_integrity_tag=jnp.asarray(0, dtype=jnp.uint32)
            ),
        )
        return _mix_words(
            _tree_content_words((self._config_fingerprint, payload)),
            salt=_LEARNER_STATE_TAG_SALT,
        )

    def _seal_state(
        self,
        state: ImaginedRolloutActorCriticState,
    ) -> ImaginedRolloutActorCriticState:
        return cast(
            ImaginedRolloutActorCriticState,
            cast(Any, state).replace(state_integrity_tag=self._state_tag(state)),
        )

    def _zero_state(self) -> ImaginedRolloutActorCriticState:
        zero_words = jnp.zeros((2,), dtype=jnp.uint32)
        state = ImaginedRolloutActorCriticState(
            actor_parameters=self._zero_actor(),
            critic_parameters=self._zero_critic(),
            actor_momentum=self._zero_actor(),
            critic_momentum=self._zero_critic(),
            update_count_words=zero_words,
            dream_update_count_words=zero_words,
            real_update_count_words=zero_words,
            backward_transition_count_words=zero_words,
            last_dream_authorization_words=zero_words,
            last_real_episode_revision_words=zero_words,
            state_integrity_tag=jnp.asarray(0, dtype=jnp.uint32),
        )
        return self._seal_state(state)

    def init(self, key: Array) -> ImaginedRolloutActorCriticState:
        """Initialize learner-owned linear parameters; no RNG remains in state."""

        try:
            key_array = jnp.asarray(key)
            implementation = str(jr.key_impl(cast(Any, key)))
        except (TypeError, ValueError) as exc:
            raise TypeError("key must be a typed scalar threefry2x32 key") from exc
        if (
            key_array.shape != ()
            or not jax.dtypes.issubdtype(key_array.dtype, jax.dtypes.prng_key)
            or implementation != "threefry2x32"
        ):
            raise TypeError("key must be a typed scalar threefry2x32 key")
        actor_key, critic_key = jr.split(key)
        scale = jnp.asarray(self._config.initialization_scale, dtype=jnp.float32)
        actor = LinearActorParameters(
            weights=scale
            * jr.normal(
                actor_key,
                (self._gauge.n_actions, self._gauge.observation_dim),
                dtype=jnp.float32,
            ),
            bias=jnp.zeros((self._gauge.n_actions,), dtype=jnp.float32),
        )
        critic = LinearCriticParameters(
            weights=scale
            * jr.normal(
                critic_key,
                (self._gauge.observation_dim,),
                dtype=jnp.float32,
            ),
            bias=jnp.asarray(0.0, dtype=jnp.float32),
        )
        return self._seal_state(
            cast(
                ImaginedRolloutActorCriticState,
                cast(Any, self._zero_state()).replace(
                    actor_parameters=actor,
                    critic_parameters=critic,
                ),
            )
        )

    def _state_static_valid(self, state: object) -> bool:
        return (
            isinstance(state, ImaginedRolloutActorCriticState)
            and _tree_static_signature(state) == self._state_signature
        )

    def _state_valid(
        self,
        state: ImaginedRolloutActorCriticState,
    ) -> Bool[Array, ""]:
        update_sum_valid = (
            state.update_count_words[0] == 0
        ) & (
            state.dream_update_count_words[0] == 0
        ) & (
            state.real_update_count_words[0] == 0
        ) & (
            state.update_count_words[1]
            == state.dream_update_count_words[1] + state.real_update_count_words[1]
        )
        return (
            _tree_finite(state)
            & update_sum_valid
            & _words_leq_limit(
                state.update_count_words,
                self._config.max_update_calls,
            )
            & _words_leq_limit(
                state.backward_transition_count_words,
                self._config.max_backward_transitions,
            )
            & (state.state_integrity_tag == self._state_tag(state))
        )

    def state_valid(
        self,
        state: ImaginedRolloutActorCriticState,
    ) -> Bool[Array, ""]:
        if not self._state_static_valid(state):
            raise TypeError("state has the wrong actor/critic static contract")
        return self._state_valid(state)

    def _zero_real_batch(self) -> CompetentRealEpisodeBatch:
        bh = (self._gauge.rollout_budget, self._gauge.rollout_horizon)
        observations = jnp.zeros(
            (*bh, self._gauge.observation_dim),
            dtype=jnp.float32,
        )
        zero_b = jnp.zeros(bh, dtype=jnp.bool_)
        return CompetentRealEpisodeBatch(
            observations=observations,
            actions=jnp.zeros(bh, dtype=jnp.int32),
            rewards=jnp.zeros(bh, dtype=jnp.float32),
            continuations=jnp.zeros(bh, dtype=jnp.float32),
            next_observations=observations,
            return_targets=jnp.zeros(bh, dtype=jnp.float32),
            terminated=zero_b,
            transition_valid=zero_b,
            competent=zero_b,
            safety_admitted=zero_b,
            protected=zero_b,
            episode_revision_words=jnp.zeros((2,), dtype=jnp.uint32),
            source_revision_words=jnp.zeros((2,), dtype=jnp.uint32),
            source_integrity_tag=jnp.asarray(0, dtype=jnp.uint32),
            batch_integrity_tag=jnp.asarray(0, dtype=jnp.uint32),
        )

    def _real_batch_tag(self, batch: CompetentRealEpisodeBatch) -> UInt[Array, ""]:
        payload = cast(
            CompetentRealEpisodeBatch,
            cast(Any, batch).replace(
                batch_integrity_tag=jnp.asarray(0, dtype=jnp.uint32)
            ),
        )
        return _mix_words(
            _tree_content_words((self._config_fingerprint, payload)),
            salt=_REAL_BATCH_TAG_SALT,
        )

    def bind_competent_real_episode(
        self,
        *,
        observations: Array,
        actions: Array,
        rewards: Array,
        continuations: Array,
        next_observations: Array,
        return_targets: Array,
        terminated: Array,
        transition_valid: Array,
        competent: Array,
        safety_admitted: Array,
        protected: Array,
        episode_revision_words: Array,
        source_revision_words: Array,
        source_integrity_tag: Array,
    ) -> CompetentRealEpisodeBatch:
        """Bind a caller-declared competent-real control batch without claims."""

        bh = (self._gauge.rollout_budget, self._gauge.rollout_horizon)
        for name, value, shape, dtype in (
            (
                "observations",
                observations,
                (*bh, self._gauge.observation_dim),
                jnp.float32,
            ),
            ("actions", actions, bh, jnp.int32),
            ("rewards", rewards, bh, jnp.float32),
            ("continuations", continuations, bh, jnp.float32),
            (
                "next_observations",
                next_observations,
                (*bh, self._gauge.observation_dim),
                jnp.float32,
            ),
            ("return_targets", return_targets, bh, jnp.float32),
            ("terminated", terminated, bh, jnp.bool_),
            ("transition_valid", transition_valid, bh, jnp.bool_),
            ("competent", competent, bh, jnp.bool_),
            ("safety_admitted", safety_admitted, bh, jnp.bool_),
            ("protected", protected, bh, jnp.bool_),
            ("episode_revision_words", episode_revision_words, (2,), jnp.uint32),
            ("source_revision_words", source_revision_words, (2,), jnp.uint32),
            ("source_integrity_tag", source_integrity_tag, (), jnp.uint32),
        ):
            _require_array(value, name=name, shape=shape, dtype=dtype)
        provisional = CompetentRealEpisodeBatch(
            observations=observations,
            actions=actions,
            rewards=rewards,
            continuations=continuations,
            next_observations=next_observations,
            return_targets=return_targets,
            terminated=terminated,
            transition_valid=transition_valid,
            competent=competent,
            safety_admitted=safety_admitted,
            protected=protected,
            episode_revision_words=episode_revision_words,
            source_revision_words=source_revision_words,
            source_integrity_tag=source_integrity_tag,
            batch_integrity_tag=jnp.asarray(0, dtype=jnp.uint32),
        )
        return cast(
            CompetentRealEpisodeBatch,
            cast(Any, provisional).replace(
                batch_integrity_tag=self._real_batch_tag(provisional)
            ),
        )

    def _real_batch_static_valid(self, batch: object) -> bool:
        return (
            isinstance(batch, CompetentRealEpisodeBatch)
            and _tree_static_signature(batch) == self._real_batch_signature
        )

    def _terminal_semantics_valid(
        self,
        rewards: Array,
        continuations: Array,
        return_targets: Array,
        terminated: Array,
        transition_valid: Array,
    ) -> Bool[Array, ""]:
        return _terminal_path_semantics_valid(
            rewards,
            continuations,
            return_targets,
            terminated,
            transition_valid,
        )

    def _real_batch_valid(self, batch: CompetentRealEpisodeBatch) -> Bool[Array, ""]:
        return (
            _tree_finite(batch)
            & _words_nonzero(batch.episode_revision_words)
            & _words_nonzero(batch.source_revision_words)
            & (batch.source_integrity_tag != 0)
            & jnp.any(batch.transition_valid & batch.competent)
            & jnp.all(
                ~batch.transition_valid
                | ((batch.actions >= 0) & (batch.actions < self._gauge.n_actions))
            )
            & self._terminal_semantics_valid(
                batch.rewards,
                batch.continuations,
                batch.return_targets,
                batch.terminated,
                batch.transition_valid,
            )
            & (batch.batch_integrity_tag == self._real_batch_tag(batch))
        )

    def _proposal_tag(
        self,
        proposal: ImaginedRolloutActorCriticUpdateProposal,
    ) -> UInt[Array, ""]:
        payload = cast(
            ImaginedRolloutActorCriticUpdateProposal,
            cast(Any, proposal).replace(
                proposal_integrity_tag=jnp.asarray(0, dtype=jnp.uint32)
            ),
        )
        return _mix_words(
            _tree_content_words((self._config_fingerprint, payload)),
            salt=_LEARNER_PROPOSAL_TAG_SALT,
        )

    def _zero_proposal(
        self,
        state: ImaginedRolloutActorCriticState,
    ) -> ImaginedRolloutActorCriticUpdateProposal:
        provisional = ImaginedRolloutActorCriticUpdateProposal(
            source_mode=jnp.asarray(self.DREAM_SOURCE_MODE, dtype=jnp.int32),
            source_state_integrity_tag=state.state_integrity_tag,
            source_update_count_words=state.update_count_words,
            source_identity_words=jnp.zeros((2,), dtype=jnp.uint32),
            source_content_tag=jnp.asarray(0, dtype=jnp.uint32),
            eligible_transition_count=jnp.asarray(0, dtype=jnp.int32),
            terminal_semantics_valid=jnp.asarray(False),
            source_authorized=jnp.asarray(False),
            update_capacity_available=jnp.asarray(False),
            backward_capacity_available=jnp.asarray(False),
            valid=jnp.asarray(False),
            proposal_integrity_tag=jnp.asarray(0, dtype=jnp.uint32),
        )
        return cast(
            ImaginedRolloutActorCriticUpdateProposal,
            cast(Any, provisional).replace(
                proposal_integrity_tag=self._proposal_tag(provisional)
            ),
        )

    def _proposal_static_valid(self, proposal: object) -> bool:
        return (
            isinstance(proposal, ImaginedRolloutActorCriticUpdateProposal)
            and _tree_static_signature(proposal) == self._proposal_signature
        )

    def _authorization_proposal(
        self,
        state: ImaginedRolloutActorCriticState,
        *,
        source_mode: int,
        source_identity_words: Array,
        source_content_tag: Array,
        rewards: Array,
        continuations: Array,
        return_targets: Array,
        terminated: Array,
        transition_valid: Array,
        training_mask: Array,
        safety_admitted: Array,
        protected: Array,
        source_authorized: Array,
    ) -> ImaginedRolloutActorCriticUpdateProposal:
        """Build source-bound metadata without evaluating a loss or gradient."""

        terminal_semantics_valid = self._terminal_semantics_valid(
            rewards,
            continuations,
            return_targets,
            terminated,
            transition_valid,
        )
        effective_mask = _prefix_closed_transition_mask(
            transition_valid,
            training_mask & safety_admitted & ~protected,
        )
        eligible_count = jnp.sum(effective_mask.astype(jnp.int32))
        _, update_word_capacity = _checked_words_add_small(
            state.update_count_words,
            1,
        )
        proposed_backward_words, backward_word_capacity = _checked_words_add_small(
            state.backward_transition_count_words,
            eligible_count.astype(jnp.uint32),
        )
        update_capacity = update_word_capacity & _words_leq_limit(
            state.update_count_words,
            self._config.max_update_calls - 1,
        )
        backward_capacity = backward_word_capacity & _words_leq_limit(
            proposed_backward_words,
            self._config.max_backward_transitions,
        )
        valid = (
            self._state_valid(state)
            & source_authorized
            & terminal_semantics_valid
            & (eligible_count > 0)
            & update_capacity
            & backward_capacity
        )
        provisional = ImaginedRolloutActorCriticUpdateProposal(
            source_mode=jnp.asarray(source_mode, dtype=jnp.int32),
            source_state_integrity_tag=state.state_integrity_tag,
            source_update_count_words=state.update_count_words,
            source_identity_words=source_identity_words,
            source_content_tag=source_content_tag,
            eligible_transition_count=eligible_count,
            terminal_semantics_valid=terminal_semantics_valid,
            source_authorized=source_authorized,
            update_capacity_available=update_capacity,
            backward_capacity_available=backward_capacity,
            valid=valid,
            proposal_integrity_tag=jnp.asarray(0, dtype=jnp.uint32),
        )
        return cast(
            ImaginedRolloutActorCriticUpdateProposal,
            cast(Any, provisional).replace(
                proposal_integrity_tag=self._proposal_tag(provisional)
            ),
        )

    def propose_dream_update(
        self,
        state: ImaginedRolloutActorCriticState,
        batch: ImaginedRolloutBatch,
        receipt: ImaginedRolloutAuthorizationReceipt,
        gauge_state: ImaginedRolloutSelectionGaugeState,
    ) -> ImaginedRolloutActorCriticUpdateProposal:
        """Form a pure graded dream self-imitation and critic proposal."""

        if not self._state_static_valid(state):
            raise TypeError("state has the wrong actor/critic static contract")
        current_receipt = self._gauge.receipt_valid(gauge_state, batch, receipt)
        fresh = _words_less(
            state.last_dream_authorization_words,
            receipt.authorization_words,
        )
        source_authorized = current_receipt & receipt.authorized & fresh
        return self._authorization_proposal(
            state,
            source_mode=self.DREAM_SOURCE_MODE,
            source_identity_words=receipt.authorization_words,
            source_content_tag=receipt.receipt_integrity_tag,
            rewards=batch.rewards,
            continuations=batch.continuations,
            return_targets=batch.return_targets,
            terminated=batch.terminated,
            transition_valid=batch.transition_valid,
            training_mask=receipt.transition_authorized,
            safety_admitted=receipt.safety_admitted,
            protected=receipt.protected,
            source_authorized=source_authorized,
        )

    def propose_competent_real_update(
        self,
        state: ImaginedRolloutActorCriticState,
        batch: CompetentRealEpisodeBatch,
    ) -> ImaginedRolloutActorCriticUpdateProposal:
        """Form the matched competent-real behavior-cloning control proposal."""

        if not self._state_static_valid(state):
            raise TypeError("state has the wrong actor/critic static contract")
        if not self._real_batch_static_valid(batch):
            raise TypeError("competent-real batch has the wrong static contract")
        fresh = _words_less(
            state.last_real_episode_revision_words,
            batch.episode_revision_words,
        )
        source_authorized = self._real_batch_valid(batch) & fresh
        return self._authorization_proposal(
            state,
            source_mode=self.COMPETENT_REAL_SOURCE_MODE,
            source_identity_words=batch.episode_revision_words,
            source_content_tag=batch.batch_integrity_tag,
            rewards=batch.rewards,
            continuations=batch.continuations,
            return_targets=batch.return_targets,
            terminated=batch.terminated,
            transition_valid=batch.transition_valid,
            training_mask=batch.competent,
            safety_admitted=batch.safety_admitted,
            protected=batch.protected,
            source_authorized=source_authorized,
        )

    def _zero_commit_trace(
        self,
        state: ImaginedRolloutActorCriticState,
        *,
        safety_admitted: Array | None = None,
        protected: Array | None = None,
    ) -> ImaginedRolloutActorCriticCommitTrace:
        shape = (self._gauge.rollout_budget, self._gauge.rollout_horizon)
        zeros = jnp.zeros(shape, dtype=jnp.float32)
        false_mask = jnp.zeros(shape, dtype=jnp.bool_)
        zero_actor = self._zero_actor()
        zero_critic = self._zero_critic()
        return ImaginedRolloutActorCriticCommitTrace(
            actor_gradient=zero_actor,
            critic_gradient=zero_critic,
            actor_momentum_candidate=state.actor_momentum,
            critic_momentum_candidate=state.critic_momentum,
            actor_parameter_update=zero_actor,
            critic_parameter_update=zero_critic,
            training_mask=false_mask,
            safety_admitted=(
                false_mask if safety_admitted is None else safety_admitted
            ),
            protected=false_mask if protected is None else protected,
            critic_targets=zeros,
            advantages=zeros,
            positive_advantages=zeros,
            imitation_weights=zeros,
            actor_loss=jnp.asarray(0.0, dtype=jnp.float32),
            critic_loss=jnp.asarray(0.0, dtype=jnp.float32),
            actor_gradient_norm=jnp.asarray(0.0, dtype=jnp.float32),
            critic_gradient_norm=jnp.asarray(0.0, dtype=jnp.float32),
            backward_transition_count=jnp.asarray(0, dtype=jnp.int32),
            backward_work_performed=jnp.asarray(False),
            candidate_finite=jnp.asarray(False),
        )

    def _autodiff_training_trace(
        self,
        state: ImaginedRolloutActorCriticState,
        *,
        source_mode: int,
        observations: Array,
        actions: Array,
        rewards: Array,
        return_targets: Array,
        terminated: Array,
        effective_mask: Array,
        safety_admitted: Array,
        protected: Array,
    ) -> ImaginedRolloutActorCriticCommitTrace:
        """Execute the one backward pass admitted by commit preflight."""

        mask_float = effective_mask.astype(jnp.float32)
        denominator = jnp.maximum(jnp.sum(mask_float), 1.0)
        critic_targets = jnp.where(terminated, rewards, return_targets)
        safe_actions = jnp.clip(actions, 0, self._gauge.n_actions - 1)

        def loss(
            actor: LinearActorParameters,
            critic: LinearCriticParameters,
        ) -> tuple[Array, tuple[Array, Array, Array, Array, Array]]:
            logits = jnp.einsum("...d,ad->...a", observations, actor.weights)
            logits = logits + actor.bias
            log_probabilities = jax.nn.log_softmax(logits, axis=-1)
            selected_log_probability = jnp.take_along_axis(
                log_probabilities,
                safe_actions[..., None],
                axis=-1,
            )[..., 0]
            values = jnp.einsum("...d,d->...", observations, critic.weights)
            values = values + critic.bias
            advantages = jax.lax.stop_gradient(critic_targets - values)
            positive_advantages = jnp.minimum(
                jnp.maximum(advantages, 0.0),
                jnp.asarray(
                    self._config.max_positive_advantage,
                    dtype=jnp.float32,
                ),
            )
            if source_mode == self.DREAM_SOURCE_MODE:
                imitation_weights = positive_advantages * mask_float
            else:
                imitation_weights = mask_float
            actor_loss = -jnp.sum(
                imitation_weights * selected_log_probability
            ) / denominator
            critic_error = values - jax.lax.stop_gradient(critic_targets)
            critic_loss = 0.5 * jnp.sum(
                mask_float * jnp.square(critic_error)
            ) / denominator
            return actor_loss + critic_loss, (
                actor_loss,
                critic_loss,
                advantages,
                positive_advantages,
                imitation_weights,
            )

        (_, auxiliary), gradients = jax.value_and_grad(
            loss,
            argnums=(0, 1),
            has_aux=True,
        )(state.actor_parameters, state.critic_parameters)
        actor_gradient_raw, critic_gradient_raw = gradients
        actor_gradient, actor_norm = _clip_tree_l2(
            actor_gradient_raw,
            self._config.gradient_clip,
        )
        critic_gradient, critic_norm = _clip_tree_l2(
            critic_gradient_raw,
            self._config.gradient_clip,
        )
        actor_gradient = cast(LinearActorParameters, actor_gradient)
        critic_gradient = cast(LinearCriticParameters, critic_gradient)
        decay = jnp.asarray(self._config.momentum_decay, dtype=jnp.float32)
        injection = 1.0 - decay
        actor_momentum = cast(
            LinearActorParameters,
            jax.tree.map(
                lambda old, gradient: decay * old + injection * gradient,
                state.actor_momentum,
                actor_gradient,
            ),
        )
        critic_momentum = cast(
            LinearCriticParameters,
            jax.tree.map(
                lambda old, gradient: decay * old + injection * gradient,
                state.critic_momentum,
                critic_gradient,
            ),
        )
        actor_update = cast(
            LinearActorParameters,
            jax.tree.map(
                lambda value: -jnp.asarray(
                    self._config.actor_step_size,
                    dtype=jnp.float32,
                )
                * value,
                actor_momentum,
            ),
        )
        critic_update = cast(
            LinearCriticParameters,
            jax.tree.map(
                lambda value: -jnp.asarray(
                    self._config.critic_step_size,
                    dtype=jnp.float32,
                )
                * value,
                critic_momentum,
            ),
        )
        actor_loss, critic_loss, advantages, positives, weights = auxiliary
        candidate_actor = jax.tree.map(
            lambda parameter, update: parameter + update,
            state.actor_parameters,
            actor_update,
        )
        candidate_critic = jax.tree.map(
            lambda parameter, update: parameter + update,
            state.critic_parameters,
            critic_update,
        )
        candidate_finite = (
            _tree_finite(actor_gradient)
            & _tree_finite(critic_gradient)
            & _tree_finite(actor_momentum)
            & _tree_finite(critic_momentum)
            & _tree_finite(actor_update)
            & _tree_finite(critic_update)
            & _tree_finite(candidate_actor)
            & _tree_finite(candidate_critic)
            & jnp.isfinite(actor_loss)
            & jnp.isfinite(critic_loss)
            & jnp.all(jnp.isfinite(critic_targets))
            & jnp.all(jnp.isfinite(advantages))
            & jnp.all(jnp.isfinite(positives))
            & jnp.all(jnp.isfinite(weights))
        )
        return ImaginedRolloutActorCriticCommitTrace(
            actor_gradient=actor_gradient,
            critic_gradient=critic_gradient,
            actor_momentum_candidate=actor_momentum,
            critic_momentum_candidate=critic_momentum,
            actor_parameter_update=actor_update,
            critic_parameter_update=critic_update,
            training_mask=effective_mask,
            safety_admitted=safety_admitted,
            protected=protected,
            critic_targets=critic_targets,
            advantages=advantages,
            positive_advantages=positives,
            imitation_weights=weights,
            actor_loss=actor_loss,
            critic_loss=critic_loss,
            actor_gradient_norm=actor_norm,
            critic_gradient_norm=critic_norm,
            backward_transition_count=jnp.sum(
                effective_mask.astype(jnp.int32)
            ),
            backward_work_performed=jnp.asarray(True),
            candidate_finite=candidate_finite,
        )

    def _commit_expected(
        self,
        destination_state: ImaginedRolloutActorCriticState,
        proposal: ImaginedRolloutActorCriticUpdateProposal,
        expected: ImaginedRolloutActorCriticUpdateProposal,
        *,
        expected_mode: int,
        observations: Array,
        actions: Array,
        rewards: Array,
        return_targets: Array,
        terminated: Array,
        transition_valid: Array,
        training_mask: Array,
        safety_admitted: Array,
        protected: Array,
    ) -> ImaginedRolloutActorCriticCommitResult:
        proposal_integrity = (
            proposal.proposal_integrity_tag == self._proposal_tag(proposal)
        ) & _tree_equal(proposal, expected)
        state_valid = self._state_valid(destination_state)
        source_matches = (
            proposal.source_state_integrity_tag
            == destination_state.state_integrity_tag
        ) & jnp.array_equal(
            proposal.source_update_count_words,
            destination_state.update_count_words,
        )
        mode_matches = proposal.source_mode == jnp.asarray(
            expected_mode,
            dtype=jnp.int32,
        )
        if expected_mode == self.DREAM_SOURCE_MODE:
            fresh = _words_less(
                destination_state.last_dream_authorization_words,
                proposal.source_identity_words,
            )
        else:
            fresh = _words_less(
                destination_state.last_real_episode_revision_words,
                proposal.source_identity_words,
            )
        effective_mask = _prefix_closed_transition_mask(
            transition_valid,
            training_mask & safety_admitted & ~protected,
        )
        backward_count = jnp.sum(effective_mask.astype(jnp.int32))
        proposed_update_words, update_word_capacity = _checked_words_add_small(
            destination_state.update_count_words,
            1,
        )
        update_capacity = update_word_capacity & _words_leq_limit(
            proposed_update_words,
            self._config.max_update_calls,
        )
        proposed_backward_words, backward_word_capacity = _checked_words_add_small(
            destination_state.backward_transition_count_words,
            backward_count.astype(jnp.uint32),
        )
        backward_capacity = backward_word_capacity & _words_leq_limit(
            proposed_backward_words,
            self._config.max_backward_transitions,
        )
        proposed_dream_words, dream_capacity = _checked_words_add_small(
            destination_state.dream_update_count_words,
            1 if expected_mode == self.DREAM_SOURCE_MODE else 0,
        )
        proposed_real_words, real_capacity = _checked_words_add_small(
            destination_state.real_update_count_words,
            1 if expected_mode == self.COMPETENT_REAL_SOURCE_MODE else 0,
        )
        source_authorized = (
            proposal.source_authorized
            & expected.source_authorized
            & proposal.valid
            & expected.valid
        )
        preflight = (
            state_valid
            & proposal_integrity
            & source_matches
            & mode_matches
            & source_authorized
            & fresh
            & update_capacity
            & backward_capacity
            & dream_capacity
            & real_capacity
            & (backward_count == expected.eligible_transition_count)
            & (backward_count > 0)
        )

        if not isinstance(preflight, jax.core.Tracer) and not bool(
            jax.device_get(preflight)
        ):
            trace = self._zero_commit_trace(
                destination_state,
                safety_admitted=safety_admitted,
                protected=protected,
            )
        else:
            trace = cast(
                ImaginedRolloutActorCriticCommitTrace,
                jax.lax.cond(
                    preflight,
                    lambda: self._autodiff_training_trace(
                        destination_state,
                        source_mode=expected_mode,
                        observations=observations,
                        actions=actions,
                        rewards=rewards,
                        return_targets=return_targets,
                        terminated=terminated,
                        effective_mask=effective_mask,
                        safety_admitted=safety_admitted,
                        protected=protected,
                    ),
                    lambda: self._zero_commit_trace(
                        destination_state,
                        safety_admitted=safety_admitted,
                        protected=protected,
                    ),
                ),
            )

        actor_parameters = cast(
            LinearActorParameters,
            jax.tree.map(
                lambda parameter, update: parameter + update,
                destination_state.actor_parameters,
                trace.actor_parameter_update,
            ),
        )
        critic_parameters = cast(
            LinearCriticParameters,
            jax.tree.map(
                lambda parameter, update: parameter + update,
                destination_state.critic_parameters,
                trace.critic_parameter_update,
            ),
        )
        candidate = ImaginedRolloutActorCriticState(
            actor_parameters=actor_parameters,
            critic_parameters=critic_parameters,
            actor_momentum=trace.actor_momentum_candidate,
            critic_momentum=trace.critic_momentum_candidate,
            update_count_words=proposed_update_words,
            dream_update_count_words=proposed_dream_words,
            real_update_count_words=proposed_real_words,
            backward_transition_count_words=proposed_backward_words,
            last_dream_authorization_words=(
                proposal.source_identity_words
                if expected_mode == self.DREAM_SOURCE_MODE
                else destination_state.last_dream_authorization_words
            ),
            last_real_episode_revision_words=(
                proposal.source_identity_words
                if expected_mode == self.COMPETENT_REAL_SOURCE_MODE
                else destination_state.last_real_episode_revision_words
            ),
            state_integrity_tag=jnp.asarray(0, dtype=jnp.uint32),
        )
        candidate = self._seal_state(candidate)
        candidate_valid = trace.candidate_finite & self._state_valid(candidate)
        applied = (
            preflight
            & trace.backward_work_performed
            & trace.candidate_finite
            & candidate_valid
        )
        next_state = cast(
            ImaginedRolloutActorCriticState,
            jax.lax.cond(applied, lambda: candidate, lambda: destination_state),
        )
        return ImaginedRolloutActorCriticCommitResult(
            state=next_state,
            trace=trace,
            diagnostics=ImaginedRolloutActorCriticCommitDiagnostics(
                state_valid=state_valid,
                proposal_integrity_valid=proposal_integrity,
                source_matches=source_matches,
                source_mode_matches=mode_matches,
                source_authorized=source_authorized,
                receipt_or_batch_fresh=fresh,
                update_capacity_available=update_capacity,
                backward_capacity_available=backward_capacity,
                source_truth_authenticated=jnp.asarray(False),
                preflight_valid=preflight,
                backward_work_performed=trace.backward_work_performed,
                autodiff_pass_count=trace.backward_work_performed.astype(jnp.int32),
                backward_transition_count=trace.backward_transition_count,
                candidate_state_valid=candidate_valid,
                applied=applied,
                pre_update_count_words=destination_state.update_count_words,
                post_update_count_words=next_state.update_count_words,
            ),
        )

    def commit_dream_update(
        self,
        destination_state: ImaginedRolloutActorCriticState,
        proposal: ImaginedRolloutActorCriticUpdateProposal,
        batch: ImaginedRolloutBatch,
        receipt: ImaginedRolloutAuthorizationReceipt,
        gauge_state: ImaginedRolloutSelectionGaugeState,
    ) -> ImaginedRolloutActorCriticCommitResult:
        """Recompute and atomically commit a current dream proposal."""

        if not self._state_static_valid(destination_state):
            raise TypeError("destination state has the wrong static contract")
        if not self._proposal_static_valid(proposal):
            raise TypeError("proposal has the wrong static contract")
        expected = self.propose_dream_update(
            destination_state,
            batch,
            receipt,
            gauge_state,
        )
        return self._commit_expected(
            destination_state,
            proposal,
            expected,
            expected_mode=self.DREAM_SOURCE_MODE,
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            return_targets=batch.return_targets,
            terminated=batch.terminated,
            transition_valid=batch.transition_valid,
            training_mask=receipt.transition_authorized,
            safety_admitted=receipt.safety_admitted,
            protected=receipt.protected,
        )

    def commit_competent_real_update(
        self,
        destination_state: ImaginedRolloutActorCriticState,
        proposal: ImaginedRolloutActorCriticUpdateProposal,
        batch: CompetentRealEpisodeBatch,
    ) -> ImaginedRolloutActorCriticCommitResult:
        """Recompute and atomically commit the matched competent-real control."""

        if not self._state_static_valid(destination_state):
            raise TypeError("destination state has the wrong static contract")
        if not self._proposal_static_valid(proposal):
            raise TypeError("proposal has the wrong static contract")
        if not self._real_batch_static_valid(batch):
            raise TypeError("competent-real batch has the wrong static contract")
        expected = self.propose_competent_real_update(destination_state, batch)
        return self._commit_expected(
            destination_state,
            proposal,
            expected,
            expected_mode=self.COMPETENT_REAL_SOURCE_MODE,
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            return_targets=batch.return_targets,
            terminated=batch.terminated,
            transition_valid=batch.transition_valid,
            training_mask=batch.competent,
            safety_admitted=batch.safety_admitted,
            protected=batch.protected,
        )

    @property
    def resource_budget(self) -> ImaginedRolloutActorCriticResourceBudget:
        state = self._zero_state()
        proposal = self._zero_proposal(state)
        trace = self._zero_commit_trace(state)
        state_scalars, state_bytes = _logical_tree_size(state)
        proposal_scalars, proposal_bytes = _logical_tree_size(proposal)
        trace_scalars, trace_bytes = _logical_tree_size(trace)
        actor_scalars, _ = _logical_tree_size(state.actor_parameters)
        critic_scalars, _ = _logical_tree_size(state.critic_parameters)
        return ImaginedRolloutActorCriticResourceBudget(
            persistent_state_scalars=state_scalars,
            persistent_state_bytes=state_bytes,
            proposal_scalars=proposal_scalars,
            proposal_bytes=proposal_bytes,
            commit_trace_scalars=trace_scalars,
            commit_trace_bytes=trace_bytes,
            max_transitions_per_update=self._max_transition_budget,
            max_update_calls=self._config.max_update_calls,
            max_backward_transitions=self._config.max_backward_transitions,
            proposal_autodiff_passes=0,
            max_autodiff_passes_per_preflight_valid_commit=1,
            rejected_preflight_autodiff_passes=0,
            backward_clock_counts_accepted_transitions=True,
            discarded_functional_state_can_repeat_pure_calls=True,
            actor_parameter_scalars=actor_scalars,
            critic_parameter_scalars=critic_scalars,
            planner_model_or_gauge_state_owned=0,
            dispatch_authority=0,
            safety_authority=0,
            output_authority=0,
            scientific_promotion_allowed=False,
        )


def save_imagined_rollout_selection_gauge_checkpoint(
    gauge: ImaginedRolloutSelectionGauge,
    state: ImaginedRolloutSelectionGaugeState,
    path: str | Path,
) -> None:
    """Persist only frozen audit records, summaries, receipts clocks, and tags."""

    if not bool(jax.device_get(gauge.state_valid(state))):
        raise ValueError("refusing to save an invalid imagined-rollout gauge state")
    config = gauge.to_config()
    save_checkpoint(
        state,
        path,
        metadata={
            "schema": IMAGINED_ROLLOUT_GAUGE_CHECKPOINT_SCHEMA,
            "gauge_config": config,
            "config_sha256": _config_digest(config),
            "resource_budget": gauge.resource_budget.to_config(),
            "planner_state_included": False,
            "model_state_included": False,
            "proposal_batches_included": False,
            "dispatch_authority": False,
            "safety_authority": False,
            "output_authority": False,
            "scientific_promotion_allowed": False,
        },
    )


def load_imagined_rollout_selection_gauge_checkpoint(
    path: str | Path,
) -> tuple[ImaginedRolloutSelectionGauge, ImaginedRolloutSelectionGaugeState]:
    """Strictly restore the sole current grounded-audit checkpoint schema."""

    metadata = load_checkpoint_metadata(path)
    if metadata.get("schema") != IMAGINED_ROLLOUT_GAUGE_CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint is not an imagined-rollout gauge v1 checkpoint")
    config = metadata.get("gauge_config")
    if not isinstance(config, Mapping):
        raise ValueError("imagined-rollout gauge checkpoint lacks gauge_config")
    config_dict = dict(config)
    if metadata.get("config_sha256") != _config_digest(config_dict):
        raise ValueError("imagined-rollout gauge config digest does not match")
    gauge = ImaginedRolloutSelectionGauge.from_config(config_dict)
    if metadata.get("resource_budget") != gauge.resource_budget.to_config():
        raise ValueError("imagined-rollout gauge resource budget does not match")
    for name in (
        "planner_state_included",
        "model_state_included",
        "proposal_batches_included",
        "dispatch_authority",
        "safety_authority",
        "output_authority",
        "scientific_promotion_allowed",
    ):
        if metadata.get(name) is not False:
            raise ValueError(f"imagined-rollout gauge checkpoint {name} must be false")
    restored, second_metadata = load_checkpoint(gauge._empty_state(), path)
    if second_metadata != metadata:
        raise ValueError("imagined-rollout gauge metadata changed between reads")
    state = cast(ImaginedRolloutSelectionGaugeState, restored)
    if not bool(jax.device_get(gauge.state_valid(state))):
        raise ValueError("imagined-rollout gauge restored an invalid state")
    if _logical_tree_size(state)[1] != gauge.resource_budget.persistent_state_bytes:
        raise ValueError("imagined-rollout gauge restored state size is invalid")
    return gauge, state


def save_imagined_rollout_actor_critic_checkpoint(
    learner: AuthorizedImaginedRolloutActorCritic,
    state: ImaginedRolloutActorCriticState,
    path: str | Path,
) -> None:
    """Persist only learner-owned parameters, momentum, clocks, and tags."""

    if not bool(jax.device_get(learner.state_valid(state))):
        raise ValueError("refusing to save an invalid rollout actor/critic state")
    config = learner.to_config()
    save_checkpoint(
        state,
        path,
        metadata={
            "schema": IMAGINED_ROLLOUT_ACTOR_CRITIC_CHECKPOINT_SCHEMA,
            "learner_config": config,
            "config_sha256": _config_digest(config),
            "resource_budget": learner.resource_budget.to_config(),
            "gauge_state_included": False,
            "planner_or_model_state_included": False,
            "training_sources_included": False,
            "dispatch_authority": False,
            "safety_authority": False,
            "output_authority": False,
            "scientific_promotion_allowed": False,
        },
    )


def load_imagined_rollout_actor_critic_checkpoint(
    path: str | Path,
) -> tuple[
    AuthorizedImaginedRolloutActorCritic,
    ImaginedRolloutActorCriticState,
]:
    """Strictly restore the sole current bounded learner checkpoint schema."""

    metadata = load_checkpoint_metadata(path)
    if metadata.get("schema") != IMAGINED_ROLLOUT_ACTOR_CRITIC_CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint is not an imagined-rollout learner v1 checkpoint")
    config = metadata.get("learner_config")
    if not isinstance(config, Mapping):
        raise ValueError("imagined-rollout learner checkpoint lacks learner_config")
    config_dict = dict(config)
    if metadata.get("config_sha256") != _config_digest(config_dict):
        raise ValueError("imagined-rollout learner config digest does not match")
    learner = AuthorizedImaginedRolloutActorCritic.from_config(config_dict)
    if metadata.get("resource_budget") != learner.resource_budget.to_config():
        raise ValueError("imagined-rollout learner resource budget does not match")
    for name in (
        "gauge_state_included",
        "planner_or_model_state_included",
        "training_sources_included",
        "dispatch_authority",
        "safety_authority",
        "output_authority",
        "scientific_promotion_allowed",
    ):
        if metadata.get(name) is not False:
            raise ValueError(f"imagined-rollout learner checkpoint {name} must be false")
    restored, second_metadata = load_checkpoint(learner._zero_state(), path)
    if second_metadata != metadata:
        raise ValueError("imagined-rollout learner metadata changed between reads")
    state = cast(ImaginedRolloutActorCriticState, restored)
    if not bool(jax.device_get(learner.state_valid(state))):
        raise ValueError("imagined-rollout learner restored an invalid state")
    if _logical_tree_size(state)[1] != learner.resource_budget.persistent_state_bytes:
        raise ValueError("imagined-rollout learner restored state size is invalid")
    return learner, state


__all__ = [
    "IMAGINED_ROLLOUT_ACTOR_CRITIC_CHECKPOINT_SCHEMA",
    "IMAGINED_ROLLOUT_ACTOR_CRITIC_CONFIG_SCHEMA",
    "IMAGINED_ROLLOUT_COMPETENT_REAL_TRUTH_AUTHENTICATED",
    "IMAGINED_ROLLOUT_CONTENT_INTEGRITY_SCOPE",
    "IMAGINED_ROLLOUT_GAUGE_CHECKPOINT_SCHEMA",
    "IMAGINED_ROLLOUT_GAUGE_CONFIG_SCHEMA",
    "IMAGINED_ROLLOUT_GAUGE_EVIDENCE_LEVEL",
    "IMAGINED_ROLLOUT_GAUGE_MECHANISM_STATUS",
    "IMAGINED_ROLLOUT_PLANNER_ISSUANCE_AUTHENTICATED",
    "IMAGINED_ROLLOUT_SCIENTIFIC_PROMOTION_ALLOWED",
    "AuthorizedImaginedRolloutActorCritic",
    "CompetentRealEpisodeBatch",
    "GroundedRolloutAuditDiagnostics",
    "GroundedRolloutAuditRecord",
    "GroundedRolloutAuditResult",
    "ImaginedRolloutActorCriticCommitDiagnostics",
    "ImaginedRolloutActorCriticCommitResult",
    "ImaginedRolloutActorCriticCommitTrace",
    "ImaginedRolloutActorCriticConfig",
    "ImaginedRolloutActorCriticResourceBudget",
    "ImaginedRolloutActorCriticState",
    "ImaginedRolloutActorCriticUpdateProposal",
    "ImaginedRolloutAuthorizationDiagnostics",
    "ImaginedRolloutAuthorizationReceipt",
    "ImaginedRolloutAuthorizationResult",
    "ImaginedRolloutSelectionGauge",
    "ImaginedRolloutSelectionGaugeConfig",
    "ImaginedRolloutSelectionGaugeResourceBudget",
    "ImaginedRolloutSelectionGaugeState",
    "LinearActorParameters",
    "LinearCriticParameters",
    "load_imagined_rollout_actor_critic_checkpoint",
    "load_imagined_rollout_selection_gauge_checkpoint",
    "save_imagined_rollout_actor_critic_checkpoint",
    "save_imagined_rollout_selection_gauge_checkpoint",
]
