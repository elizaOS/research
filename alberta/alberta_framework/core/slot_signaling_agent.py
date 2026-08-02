# mypy: disable-error-code="call-arg"
"""Fixed-capacity role learners for hidden-regime signaling development.

This is an L1 mechanism substrate, not scientific evidence.  A helper and a
beneficiary each own four physically separate ternary contextual-bandit
tables: scratch slot zero and three durable slots.  The roles have independent
policy keys and values, while an environment-independent bounded search
protocol stays synchronized because both roles receive the same scalar reward.

Durable tables are read-only in the selective configuration.  Reward
relevance statistics remain plastic and are stored separately from values, so
testing a durable slot cannot overwrite or delete it.  A relevant durable slot
stays active; a failed slot starts an exhaustive, vacancy-skipping search of
the other durable slots before scratch resumes learning.  Scratch becomes a
candidate only after consecutive successful leases.  Failed scratch leases
receive a configurable uninterrupted residency before durable retesting.  A
confirmed candidate fills a vacancy or atomically replaces one stale durable
generation.
Independent static axes control durable test writes (selective or writable)
and full-bank replacement (failure evidence or least-recently-successful).
``writable_lru_ablation`` is shorthand for the writable/LRU cell and cannot
be combined with the explicit axis settings; every cell has identical
persistent state.

Three durable slots cannot preserve more than three arbitrary incompatible
conventions.  With no regime observation, retrieval must spend ordinary
experience testing slots; neither bounded memory nor bounded computation can
provide zero-latency recognition of an unbounded convention sequence.
"""

from __future__ import annotations

import dataclasses
import math
from numbers import Real
from typing import Any, Literal

import chex
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, PRNGKeyArray

N_SIGNAL_SYMBOLS = 3
N_SLOT_INPUTS = 3
N_SLOT_ACTIONS = 3
N_DURABLE_SLOTS = 3
N_SLOTS = N_DURABLE_SLOTS + 1
SCRATCH_SLOT = 0
SLOT_VALUE_SHAPE = (N_SLOTS, N_SLOT_INPUTS, N_SLOT_ACTIONS)

SLOT_VACANT = 0
SLOT_SCRATCH = 1
SLOT_DURABLE = 2

_HELPER_RNG_TAG = 0x53484C50  # ASCII "SHLP"
_BENEFICIARY_RNG_TAG = 0x53424E46  # ASCII "SBNF"
_INT32_MAX = np.iinfo(np.int32).max

DURABLE_WRITE_SELECTIVE: Literal["selective"] = "selective"
DURABLE_WRITE_WRITABLE: Literal["writable"] = "writable"
REPLACEMENT_TARGET_EVIDENCE: Literal["evidence"] = "evidence"
REPLACEMENT_TARGET_LRU: Literal["lru"] = "lru"

type DurableWritePolicy = Literal["selective", "writable"]
type ReplacementTargetPolicy = Literal["evidence", "lru"]


@dataclasses.dataclass(frozen=True)
class SlotSignalingConfig:
    """Learner-local lifecycle gates; never evaluator acceptance thresholds.

    ``relevance_mean`` uses a bias-corrected recency update: it is an exact
    sample mean until ``1 / relevance_rate`` observations and an exponential
    mean thereafter.  ``confirmation_steps`` gates its minimum mass.  At each
    lease boundary, ``durable_retrieval_threshold`` controls durable stay and
    failure evidence.  The strictly higher
    ``candidate_confirmation_threshold`` prevents a merely useful stored
    convention from also qualifying as a novel scratch candidate.  Scratch
    must satisfy the higher criterion for
    ``candidate_confirmation_leases`` consecutive complete leases whose raw
    lease mean also crosses the higher threshold before it may fill a vacancy
    or replace a durable generation.  After exhaustive durable retrieval has
    failed, ``scratch_training_leases_before_retest`` gives scratch a bounded
    uninterrupted training residency: only that many consecutive failed
    scratch leases trigger another exhaustive durable search.  A useful
    candidate lease, durable retrieval success, or commit resets the counter.
    """

    learning_rate: float = 0.1
    epsilon: float = 0.1
    relevance_rate: float = 0.1
    lease_length: int = 32
    confirmation_steps: int = 8
    durable_retrieval_threshold: float = 0.5
    candidate_confirmation_threshold: float = 0.75
    candidate_confirmation_leases: int = 2
    scratch_training_leases_before_retest: int = 1
    writable_lru_ablation: bool = False
    durable_write_policy: DurableWritePolicy | None = None
    replacement_target_policy: ReplacementTargetPolicy | None = None

    def __post_init__(self) -> None:
        for name in ("learning_rate", "relevance_rate"):
            value = getattr(self, name)
            if (
                not isinstance(value, Real)
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or not 0.0 < float(value) <= 1.0
            ):
                raise ValueError(f"{name} must lie in (0, 1]")
        if (
            not isinstance(self.epsilon, Real)
            or isinstance(self.epsilon, bool)
            or not math.isfinite(float(self.epsilon))
            or not 0.0 <= float(self.epsilon) <= 1.0
        ):
            raise ValueError("epsilon must lie in [0, 1]")
        if (
            type(self.lease_length) is not int
            or not 1 <= self.lease_length <= _INT32_MAX
        ):
            raise ValueError("lease_length must be a positive int32 integer")
        if (
            type(self.confirmation_steps) is not int
            or self.confirmation_steps < 1
            or self.confirmation_steps > self.lease_length
        ):
            raise ValueError("confirmation_steps must lie in [1, lease_length]")
        for name in ("durable_retrieval_threshold", "candidate_confirmation_threshold"):
            value = getattr(self, name)
            if (
                not isinstance(value, Real)
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.durable_retrieval_threshold >= self.candidate_confirmation_threshold:
            raise ValueError(
                "durable_retrieval_threshold must be below candidate_confirmation_threshold"
            )
        if (
            type(self.candidate_confirmation_leases) is not int
            or not 1 <= self.candidate_confirmation_leases <= _INT32_MAX
        ):
            raise ValueError("candidate_confirmation_leases must be a positive int32 integer")
        if (
            type(self.scratch_training_leases_before_retest) is not int
            or not 1 <= self.scratch_training_leases_before_retest <= _INT32_MAX
        ):
            raise ValueError(
                "scratch_training_leases_before_retest must be a positive int32 integer"
            )
        if type(self.writable_lru_ablation) is not bool:
            raise ValueError("writable_lru_ablation must be boolean")
        if self.durable_write_policy is not None and (
            type(self.durable_write_policy) is not str
            or self.durable_write_policy
            not in (DURABLE_WRITE_SELECTIVE, DURABLE_WRITE_WRITABLE)
        ):
            raise ValueError("durable_write_policy must be 'selective', 'writable', or None")
        if self.replacement_target_policy is not None and (
            type(self.replacement_target_policy) is not str
            or self.replacement_target_policy
            not in (REPLACEMENT_TARGET_EVIDENCE, REPLACEMENT_TARGET_LRU)
        ):
            raise ValueError("replacement_target_policy must be 'evidence', 'lru', or None")
        explicit_axes = (
            self.durable_write_policy is not None,
            self.replacement_target_policy is not None,
        )
        if explicit_axes[0] != explicit_axes[1]:
            raise ValueError(
                "durable-write and replacement-target policies must be explicit together"
            )
        if self.writable_lru_ablation and any(explicit_axes):
            raise ValueError(
                "legacy writable_lru_ablation cannot be combined with explicit policies"
            )

    @property
    def effective_durable_write_policy(self) -> DurableWritePolicy:
        """Resolve the compatibility switch or the explicit durable-write axis."""

        if self.writable_lru_ablation:
            return DURABLE_WRITE_WRITABLE
        return self.durable_write_policy or DURABLE_WRITE_SELECTIVE

    @property
    def effective_replacement_target_policy(self) -> ReplacementTargetPolicy:
        """Resolve the compatibility switch or the explicit replacement axis."""

        if self.writable_lru_ablation:
            return REPLACEMENT_TARGET_LRU
        return self.replacement_target_policy or REPLACEMENT_TARGET_EVIDENCE

    @property
    def durable_writes_enabled(self) -> bool:
        """Whether ordinary learning writes may mutate an active durable table."""

        return self.effective_durable_write_policy == DURABLE_WRITE_WRITABLE

    def to_dict(self) -> dict[str, Any]:
        """Return the exact development mechanism configuration."""

        return {
            "learning_rate": float(self.learning_rate),
            "epsilon": float(self.epsilon),
            "relevance_rate": float(self.relevance_rate),
            "lease_length": self.lease_length,
            "confirmation_steps": self.confirmation_steps,
            "durable_retrieval_threshold": float(self.durable_retrieval_threshold),
            "candidate_confirmation_threshold": float(self.candidate_confirmation_threshold),
            "candidate_confirmation_leases": self.candidate_confirmation_leases,
            "scratch_training_leases_before_retest": (self.scratch_training_leases_before_retest),
            "writable_lru_ablation": self.writable_lru_ablation,
            "requested_durable_write_policy": self.durable_write_policy,
            "requested_replacement_target_policy": self.replacement_target_policy,
            "effective_durable_write_policy": self.effective_durable_write_policy,
            "effective_replacement_target_policy": self.effective_replacement_target_policy,
            "development_only": True,
            "scientific_promotion_allowed": False,
        }


@chex.dataclass(frozen=True)
class SlotSignalingKeys:
    """Named independent policy streams for the two physical roles."""

    helper: PRNGKeyArray
    beneficiary: PRNGKeyArray


def slot_signaling_keys(root_key: Array) -> SlotSignalingKeys:
    """Derive stable role keys without positional split coupling."""

    return SlotSignalingKeys(
        helper=jr.fold_in(root_key, _HELPER_RNG_TAG),
        beneficiary=jr.fold_in(root_key, _BENEFICIARY_RNG_TAG),
    )


@chex.dataclass(frozen=True)
class SlotRoleState:
    """One role's fixed-shape values, relevance memory, lifecycle, and key."""

    values: Float[Array, "4 3 3"]
    relevance_mean: Float[Array, "4"]  # noqa: UP037
    relevance_mass: Float[Array, "4"]  # noqa: UP037
    # Slot zero counts consecutive failed scratch-training leases; durable
    # entries count failed retrieval leases used by replacement selection.
    failed_leases: Int[Array, "4"]  # noqa: UP037
    idle_leases: Int[Array, "4"]  # noqa: UP037
    status: Int[Array, "4"]  # noqa: UP037
    generation: Int[Array, "4"]  # noqa: UP037
    active_slot: Int[Array, ""]
    lease_offset: Int[Array, ""]
    lease_reward_sum: Float[Array, ""]
    remaining_durable_tests: Int[Array, ""]
    search_cursor: Int[Array, ""]
    candidate_successful_leases: Int[Array, ""]
    next_generation: Int[Array, ""]
    key: PRNGKeyArray


@chex.dataclass(frozen=True)
class SlotSignalingState:
    """Joint container with no shared trainable values or policy key."""

    helper: SlotRoleState
    beneficiary: SlotRoleState


@chex.dataclass(frozen=True)
class SlotRoleDecision:
    """Pre-reward decision made from one role's old active slot."""

    slot: Int[Array, ""]
    private_input: Int[Array, ""]
    action: Int[Array, ""]
    selected_value: Float[Array, ""]
    next_key: PRNGKeyArray


@chex.dataclass(frozen=True)
class SlotRoleUpdate:
    """One role's update and auditable lifecycle diagnostics."""

    state: SlotRoleState
    value_pre: Float[Array, ""]
    candidate_value: Float[Array, ""]
    value_post: Float[Array, ""]
    value_write: Bool[Array, ""]
    lease_boundary: Bool[Array, ""]
    lease_reward_mean: Float[Array, ""]
    relevance_ready: Bool[Array, ""]
    durable_relevant: Bool[Array, ""]
    candidate_relevant: Bool[Array, ""]
    candidate_lease_success: Bool[Array, ""]
    scratch_failed_leases_pre: Int[Array, ""]
    scratch_failed_leases_post: Int[Array, ""]
    scratch_retest_started: Bool[Array, ""]
    generation_exhausted: Bool[Array, ""]
    committed_slot: Int[Array, ""]
    committed_generation: Int[Array, ""]
    retired_slot: Int[Array, ""]
    retired_generation: Int[Array, ""]


@chex.dataclass(frozen=True)
class SlotSignalingUpdate:
    """Atomic helper/beneficiary result formed from one old joint state."""

    state: SlotSignalingState
    helper: SlotRoleUpdate
    beneficiary: SlotRoleUpdate
    lifecycle_synchronized: Bool[Array, ""]


@dataclasses.dataclass(frozen=True)
class SlotRoleResourceBudget:
    """Exact persistent resource count for one slot role."""

    value_scalars: int
    relevance_scalars: int
    lifecycle_scalars: int
    key_scalars: int
    state_scalars: int
    state_bytes: int


@dataclasses.dataclass(frozen=True)
class SlotSignalingResourceBudget:
    """Exact persistent state budget for the physically separate dyad."""

    helper: SlotRoleResourceBudget
    beneficiary: SlotRoleResourceBudget
    state_scalars: int
    state_bytes: int


def _zero_role(key: Array) -> SlotRoleState:
    status = jnp.zeros((N_SLOTS,), dtype=jnp.int32).at[SCRATCH_SLOT].set(SLOT_SCRATCH)
    return SlotRoleState(
        values=jnp.zeros(SLOT_VALUE_SHAPE, dtype=jnp.float32),
        relevance_mean=jnp.zeros((N_SLOTS,), dtype=jnp.float32),
        relevance_mass=jnp.zeros((N_SLOTS,), dtype=jnp.float32),
        failed_leases=jnp.zeros((N_SLOTS,), dtype=jnp.int32),
        idle_leases=jnp.zeros((N_SLOTS,), dtype=jnp.int32),
        status=status,
        generation=jnp.zeros((N_SLOTS,), dtype=jnp.int32),
        active_slot=jnp.asarray(SCRATCH_SLOT, dtype=jnp.int32),
        lease_offset=jnp.asarray(0, dtype=jnp.int32),
        lease_reward_sum=jnp.asarray(0.0, dtype=jnp.float32),
        remaining_durable_tests=jnp.asarray(0, dtype=jnp.int32),
        search_cursor=jnp.asarray(1, dtype=jnp.int32),
        candidate_successful_leases=jnp.asarray(0, dtype=jnp.int32),
        next_generation=jnp.asarray(1, dtype=jnp.int32),
        key=key,
    )


def _saturating_increment(value: Array) -> Array:
    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    return jnp.where(value < maximum, value + jnp.asarray(1, dtype=jnp.int32), value)


def _next_durable_slot(
    status: Array,
    cursor: Array,
    excluded_slot: Array,
) -> tuple[Array, Array, Array]:
    """Return the next occupied durable in cyclic order and its candidate count."""

    offsets = jnp.arange(N_DURABLE_SLOTS, dtype=jnp.int32)
    candidates = ((jnp.asarray(cursor, dtype=jnp.int32) - 1 + offsets) % 3) + 1
    valid = jnp.logical_and(
        status[candidates] == SLOT_DURABLE,
        candidates != jnp.asarray(excluded_slot, dtype=jnp.int32),
    )
    first_offset = jnp.argmin(jnp.where(valid, offsets, N_DURABLE_SLOTS))
    selected = candidates[first_offset]
    found = jnp.any(valid)
    selected = jnp.where(found, selected, jnp.asarray(SCRATCH_SLOT, dtype=jnp.int32))
    next_cursor = jnp.where(found, (selected % 3) + 1, cursor).astype(jnp.int32)
    return selected.astype(jnp.int32), next_cursor, jnp.sum(valid).astype(jnp.int32)


def _select_role(
    state: SlotRoleState,
    private_input: Array,
    epsilon: float,
) -> SlotRoleDecision:
    slot = state.active_slot.astype(jnp.int32)
    input_i = jnp.asarray(private_input, dtype=jnp.int32)
    values = state.values[slot, input_i]
    next_key, explore_key, random_action_key, tie_key = jr.split(state.key, 4)
    random_action = jr.randint(
        random_action_key,
        (),
        0,
        N_SLOT_ACTIONS,
        dtype=jnp.int32,
    )
    tie_action = jr.randint(tie_key, (), 0, N_SLOT_ACTIONS, dtype=jnp.int32)
    maximum = jnp.max(values)
    tied = values == maximum
    tie_order = (jnp.arange(N_SLOT_ACTIONS, dtype=jnp.int32) - tie_action) % N_SLOT_ACTIONS
    greedy_action = jnp.argmin(jnp.where(tied, tie_order, N_SLOT_ACTIONS)).astype(jnp.int32)
    explore = jr.uniform(explore_key, (), dtype=jnp.float32) < jnp.float32(epsilon)
    action = jnp.where(explore, random_action, greedy_action).astype(jnp.int32)
    return SlotRoleDecision(
        slot=slot,
        private_input=input_i,
        action=action,
        selected_value=values[action],
        next_key=next_key,
    )


def greedy_slot_action(values: Array, slot: Array, private_input: Array) -> Array:
    """Read one module without consuming policy randomness or mutating state."""

    row = values[
        jnp.asarray(slot, dtype=jnp.int32),
        jnp.asarray(private_input, dtype=jnp.int32),
    ]
    return jnp.argmax(row).astype(jnp.int32)


def _lifecycle_equal(left: SlotRoleState, right: SlotRoleState) -> Array:
    fields = (
        "relevance_mean",
        "relevance_mass",
        "failed_leases",
        "idle_leases",
        "status",
        "generation",
        "active_slot",
        "lease_offset",
        "lease_reward_sum",
        "remaining_durable_tests",
        "search_cursor",
        "candidate_successful_leases",
        "next_generation",
    )
    equal = jnp.asarray(True)
    for name in fields:
        equal = jnp.logical_and(equal, jnp.array_equal(getattr(left, name), getattr(right, name)))
    return equal


class SlotSignalingAgent:
    """Two independent four-slot learners with a synchronized search protocol."""

    def __init__(self, config: SlotSignalingConfig | None = None) -> None:
        self._config = config or SlotSignalingConfig()

    @property
    def config(self) -> SlotSignalingConfig:
        """Return the static development mechanism configuration."""

        return self._config

    def init(self, keys: SlotSignalingKeys) -> SlotSignalingState:
        """Initialize exact-zero role memories with distinct named keys."""

        return SlotSignalingState(
            helper=_zero_role(keys.helper),
            beneficiary=_zero_role(keys.beneficiary),
        )

    def select_helper(
        self,
        state: SlotRoleState,
        private_cue: Array,
    ) -> SlotRoleDecision:
        """Select a message without beneficiary, regime, target, or schedule input."""

        return _select_role(state, private_cue, self._config.epsilon)

    def select_beneficiary(
        self,
        state: SlotRoleState,
        delivered_message: Array,
    ) -> SlotRoleDecision:
        """Select an action without cue, regime, target, or schedule input."""

        return _select_role(state, delivered_message, self._config.epsilon)

    def _update_role(
        self,
        old: SlotRoleState,
        decision: SlotRoleDecision,
        reward: Array,
        external_value_write: Array | bool,
        lifecycle_write: Array | bool,
    ) -> SlotRoleUpdate:
        slot = decision.slot.astype(jnp.int32)
        reward_f = jnp.asarray(reward, dtype=jnp.float32)
        external_value_write_b = jnp.asarray(external_value_write, dtype=jnp.bool_)
        lifecycle_write_b = jnp.asarray(lifecycle_write, dtype=jnp.bool_)
        value_pre = old.values[slot, decision.private_input, decision.action]
        candidate_value = value_pre + jnp.float32(self._config.learning_rate) * (
            reward_f - value_pre
        )
        candidate_values = old.values.at[
            slot,
            decision.private_input,
            decision.action,
        ].set(candidate_value)
        is_scratch = slot == SCRATCH_SLOT
        is_durable = old.status[slot] == SLOT_DURABLE
        ordinary_slot_writable = jnp.logical_or(
            is_scratch,
            jnp.logical_and(
                jnp.asarray(self._config.durable_writes_enabled),
                is_durable,
            ),
        )
        value_write = jnp.logical_and(external_value_write_b, ordinary_slot_writable)
        learned_values = jnp.where(value_write, candidate_values, old.values)

        active_mass = jnp.minimum(
            old.relevance_mass[slot] + jnp.asarray(1.0, dtype=jnp.float32),
            jnp.asarray(16_777_216.0, dtype=jnp.float32),
        )
        relevance_gain = jnp.maximum(
            jnp.float32(self._config.relevance_rate),
            jnp.asarray(1.0, dtype=jnp.float32) / active_mass,
        )
        active_relevance = old.relevance_mean[slot] + relevance_gain * (
            reward_f - old.relevance_mean[slot]
        )
        relevance = old.relevance_mean.at[slot].set(active_relevance)
        relevance_mass = old.relevance_mass.at[slot].set(active_mass)
        lease_sum = old.lease_reward_sum + reward_f
        boundary = old.lease_offset == self._config.lease_length - 1
        lease_mean = lease_sum / jnp.float32(self._config.lease_length)
        relevance_ready = active_mass >= jnp.float32(self._config.confirmation_steps)
        durable_relevant = jnp.logical_and(
            relevance_ready,
            active_relevance >= jnp.float32(self._config.durable_retrieval_threshold),
        )
        candidate_relevant = jnp.logical_and(
            relevance_ready,
            active_relevance >= jnp.float32(self._config.candidate_confirmation_threshold),
        )

        committed_mask = old.status == SLOT_DURABLE
        incremented_idle = jnp.where(
            jnp.logical_and(boundary, committed_mask),
            _saturating_increment(old.idle_leases),
            old.idle_leases,
        )
        boundary_success = jnp.logical_and(
            boundary,
            jnp.logical_and(is_durable, durable_relevant),
        )
        boundary_failure = jnp.logical_and(
            boundary,
            jnp.logical_and(
                is_durable,
                jnp.logical_and(relevance_ready, jnp.logical_not(durable_relevant)),
            ),
        )
        failed_value = jnp.where(
            boundary_success,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.where(
                boundary_failure,
                _saturating_increment(old.failed_leases[slot]),
                old.failed_leases[slot],
            ),
        )
        failed_leases = old.failed_leases.at[slot].set(failed_value)
        idle_value = jnp.where(
            boundary_success,
            jnp.asarray(0, dtype=jnp.int32),
            incremented_idle[slot],
        )
        idle_leases = incremented_idle.at[slot].set(idle_value)
        base_values = learned_values
        base_relevance = relevance
        base_mass = relevance_mass
        base_failed = failed_leases
        base_idle = idle_leases
        base_status = old.status
        base_generation = old.generation

        candidate_lease_success = jnp.logical_and(
            boundary,
            jnp.logical_and(
                is_scratch,
                jnp.logical_and(
                    candidate_relevant,
                    lease_mean >= jnp.float32(self._config.candidate_confirmation_threshold),
                ),
            ),
        )
        scratch_failure = jnp.logical_and(
            boundary,
            jnp.logical_and(
                is_scratch,
                jnp.logical_and(
                    relevance_ready,
                    jnp.logical_not(candidate_lease_success),
                ),
            ),
        )
        scratch_failed_leases_pre = old.failed_leases[SCRATCH_SLOT]
        incremented_scratch_failed_leases = _saturating_increment(scratch_failed_leases_pre)
        scratch_retest_due = jnp.logical_and(
            scratch_failure,
            incremented_scratch_failed_leases >= self._config.scratch_training_leases_before_retest,
        )
        scratch_failed_leases_after_evidence = jnp.where(
            candidate_lease_success,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.where(
                scratch_failure,
                jnp.where(
                    scratch_retest_due,
                    jnp.asarray(0, dtype=jnp.int32),
                    incremented_scratch_failed_leases,
                ),
                scratch_failed_leases_pre,
            ),
        )
        base_failed = failed_leases.at[SCRATCH_SLOT].set(scratch_failed_leases_after_evidence)
        candidate_successful_leases = jnp.where(
            candidate_lease_success,
            _saturating_increment(old.candidate_successful_leases),
            jnp.where(
                scratch_failure,
                jnp.asarray(0, dtype=jnp.int32),
                old.candidate_successful_leases,
            ),
        )
        candidate_confirmed = jnp.logical_and(
            candidate_lease_success,
            candidate_successful_leases >= self._config.candidate_confirmation_leases,
        )
        durable_status = base_status[1:]
        vacant = durable_status == SLOT_VACANT
        has_vacancy = jnp.any(vacant)
        first_vacant = jnp.argmax(vacant.astype(jnp.int32)).astype(jnp.int32) + 1
        lru_slot = jnp.argmax(base_idle[1:]).astype(jnp.int32) + 1
        # Selective replacement first prefers the longest current failure
        # streak, then the longest time since success, then the lowest slot.
        # The writable ablation deliberately uses plain LRU instead.
        maximum_failures = jnp.max(base_failed[1:])
        most_failed = base_failed[1:] == maximum_failures
        evidence_idle = jnp.where(
            most_failed,
            base_idle[1:],
            jnp.asarray(-1, dtype=jnp.int32),
        )
        evidence_slot = jnp.argmax(evidence_idle).astype(jnp.int32) + 1
        full_bank_target = jnp.where(
            jnp.asarray(
                self._config.effective_replacement_target_policy == REPLACEMENT_TARGET_LRU
            ),
            lru_slot,
            evidence_slot,
        )
        target = jnp.where(has_vacancy, first_vacant, full_bank_target)
        commit_requested = jnp.logical_and(
            candidate_confirmed,
            lifecycle_write_b,
        )
        generation_available = old.next_generation < jnp.asarray(_INT32_MAX, dtype=jnp.int32)
        generation_exhausted = jnp.logical_and(
            commit_requested,
            jnp.logical_not(generation_available),
        )
        commit = jnp.logical_and(commit_requested, generation_available)
        replace = jnp.logical_and(commit, jnp.logical_not(has_vacancy))
        replaced_generation = base_generation[target]

        committed_values = base_values.at[target].set(base_values[SCRATCH_SLOT])
        committed_values = committed_values.at[SCRATCH_SLOT].set(
            jnp.zeros((3, 3), dtype=jnp.float32)
        )
        committed_relevance = base_relevance.at[target].set(base_relevance[SCRATCH_SLOT])
        committed_relevance = committed_relevance.at[SCRATCH_SLOT].set(0.0)
        committed_mass = base_mass.at[target].set(base_mass[SCRATCH_SLOT])
        committed_mass = committed_mass.at[SCRATCH_SLOT].set(0.0)
        committed_failed = base_failed.at[target].set(0)
        committed_failed = committed_failed.at[SCRATCH_SLOT].set(0)
        committed_idle = base_idle.at[target].set(0)
        committed_idle = committed_idle.at[SCRATCH_SLOT].set(0)
        committed_status = base_status.at[target].set(SLOT_DURABLE)
        committed_status = committed_status.at[SCRATCH_SLOT].set(SLOT_SCRATCH)
        committed_generation = base_generation.at[target].set(old.next_generation)
        committed_generation = committed_generation.at[SCRATCH_SLOT].set(0)

        final_values = jnp.where(commit, committed_values, base_values)
        final_relevance = jnp.where(commit, committed_relevance, base_relevance)
        final_mass = jnp.where(commit, committed_mass, base_mass)
        final_failed = jnp.where(commit, committed_failed, base_failed)
        final_idle = jnp.where(commit, committed_idle, base_idle)
        final_status = jnp.where(commit, committed_status, base_status)
        final_generation = jnp.where(commit, committed_generation, base_generation)
        final_candidate_successful_leases = jnp.where(
            commit,
            jnp.asarray(0, dtype=jnp.int32),
            candidate_successful_leases,
        )
        next_generation = jnp.where(
            commit,
            _saturating_increment(old.next_generation),
            old.next_generation,
        )

        other_slot, other_cursor, other_count = _next_durable_slot(
            final_status,
            old.search_cursor,
            slot,
        )
        any_slot, any_cursor, any_count = _next_durable_slot(
            final_status,
            old.search_cursor,
            jnp.asarray(-1, dtype=jnp.int32),
        )
        continuing_search = old.remaining_durable_tests > 0
        tests_after_current = jnp.maximum(
            old.remaining_durable_tests - jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )
        can_continue = jnp.logical_and(tests_after_current > 0, other_count > 0)
        continued_slot = jnp.where(can_continue, other_slot, SCRATCH_SLOT)
        continued_remaining = jnp.where(
            can_continue,
            jnp.minimum(tests_after_current, other_count),
            jnp.asarray(0, dtype=jnp.int32),
        )
        continued_cursor = jnp.where(can_continue, other_cursor, old.search_cursor)
        can_start_other_search = other_count > 0
        started_other_slot = jnp.where(can_start_other_search, other_slot, SCRATCH_SLOT)
        started_other_remaining = jnp.where(
            can_start_other_search,
            other_count,
            jnp.asarray(0, dtype=jnp.int32),
        )
        started_other_cursor = jnp.where(
            can_start_other_search,
            other_cursor,
            old.search_cursor,
        )
        failed_durable_slot = jnp.where(
            continuing_search,
            continued_slot,
            started_other_slot,
        )
        failed_durable_remaining = jnp.where(
            continuing_search,
            continued_remaining,
            started_other_remaining,
        )
        failed_durable_cursor = jnp.where(
            continuing_search,
            continued_cursor,
            started_other_cursor,
        )
        can_start_any_search = any_count > 0
        failed_scratch_slot = jnp.where(can_start_any_search, any_slot, SCRATCH_SLOT)
        failed_scratch_remaining = jnp.where(
            can_start_any_search,
            any_count,
            jnp.asarray(0, dtype=jnp.int32),
        )
        failed_scratch_cursor = jnp.where(
            can_start_any_search,
            any_cursor,
            old.search_cursor,
        )
        scratch_retest_started = jnp.logical_and(
            scratch_retest_due,
            can_start_any_search,
        )

        failed_slot = jnp.where(is_durable, failed_durable_slot, failed_scratch_slot)
        failed_remaining = jnp.where(
            is_durable,
            failed_durable_remaining,
            failed_scratch_remaining,
        )
        failed_cursor = jnp.where(
            is_durable,
            failed_durable_cursor,
            failed_scratch_cursor,
        )
        insufficient_evidence = jnp.logical_not(relevance_ready)
        keep_relevant_slot = jnp.where(
            is_scratch,
            jnp.logical_or(
                candidate_lease_success,
                jnp.logical_and(scratch_failure, jnp.logical_not(scratch_retest_due)),
            ),
            durable_relevant,
        )
        boundary_slot = jnp.where(
            commit,
            target,
            jnp.where(
                insufficient_evidence,
                slot,
                jnp.where(keep_relevant_slot, slot, failed_slot),
            ),
        ).astype(jnp.int32)
        boundary_remaining = jnp.where(
            commit,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.where(
                insufficient_evidence,
                old.remaining_durable_tests,
                jnp.where(
                    keep_relevant_slot,
                    jnp.asarray(0, dtype=jnp.int32),
                    failed_remaining,
                ),
            ),
        ).astype(jnp.int32)
        relevant_cursor = jnp.where(is_durable, (slot % 3) + 1, old.search_cursor)
        boundary_cursor = jnp.where(
            commit,
            (target % 3) + 1,
            jnp.where(
                insufficient_evidence,
                old.search_cursor,
                jnp.where(keep_relevant_slot, relevant_cursor, failed_cursor),
            ),
        ).astype(jnp.int32)
        active_slot = jnp.where(boundary, boundary_slot, slot)
        remaining_durable_tests = jnp.where(
            boundary,
            boundary_remaining,
            old.remaining_durable_tests,
        )
        search_cursor = jnp.where(boundary, boundary_cursor, old.search_cursor)
        lease_offset = jnp.where(
            boundary,
            jnp.asarray(0, dtype=jnp.int32),
            old.lease_offset + jnp.asarray(1, dtype=jnp.int32),
        )
        next_lease_sum = jnp.where(
            boundary,
            jnp.asarray(0.0, dtype=jnp.float32),
            lease_sum,
        )
        exhausted_durable_search = jnp.logical_and(
            boundary_failure,
            failed_durable_slot == SCRATCH_SLOT,
        )
        reset_scratch_failed_leases = jnp.logical_or(
            commit,
            jnp.logical_or(boundary_success, exhausted_durable_search),
        )
        next_failed = final_failed.at[SCRATCH_SLOT].set(
            jnp.where(
                reset_scratch_failed_leases,
                jnp.asarray(0, dtype=jnp.int32),
                final_failed[SCRATCH_SLOT],
            )
        )
        next_state = SlotRoleState(
            values=final_values,
            relevance_mean=final_relevance,
            relevance_mass=final_mass,
            failed_leases=next_failed,
            idle_leases=final_idle,
            status=final_status,
            generation=final_generation,
            active_slot=active_slot,
            lease_offset=lease_offset,
            lease_reward_sum=next_lease_sum,
            remaining_durable_tests=remaining_durable_tests,
            search_cursor=search_cursor,
            candidate_successful_leases=final_candidate_successful_leases,
            next_generation=next_generation,
            key=decision.next_key,
        )
        return SlotRoleUpdate(
            state=next_state,
            value_pre=value_pre,
            candidate_value=candidate_value,
            value_post=next_state.values[slot, decision.private_input, decision.action],
            value_write=value_write,
            lease_boundary=boundary,
            lease_reward_mean=lease_mean,
            relevance_ready=relevance_ready,
            durable_relevant=durable_relevant,
            candidate_relevant=candidate_relevant,
            candidate_lease_success=candidate_lease_success,
            scratch_failed_leases_pre=scratch_failed_leases_pre,
            scratch_failed_leases_post=next_failed[SCRATCH_SLOT],
            scratch_retest_started=scratch_retest_started,
            generation_exhausted=generation_exhausted,
            committed_slot=jnp.where(commit, target, jnp.asarray(-1, dtype=jnp.int32)),
            committed_generation=jnp.where(
                commit,
                old.next_generation,
                jnp.asarray(-1, dtype=jnp.int32),
            ),
            retired_slot=jnp.where(replace, target, jnp.asarray(-1, dtype=jnp.int32)),
            retired_generation=jnp.where(
                replace,
                replaced_generation,
                jnp.asarray(-1, dtype=jnp.int32),
            ),
        )

    def update_role(
        self,
        old_state: SlotRoleState,
        decision: SlotRoleDecision,
        reward: Array,
        *,
        value_write: Array | bool = True,
        lifecycle_write: Array | bool = True,
    ) -> SlotRoleUpdate:
        """Apply one decentralized role transition with explicit local permits.

        The transition consumes only the role's old local state, its pre-reward
        decision, the common reward, and caller-supplied value/lifecycle write
        permits.  Calling it once per role with the same lifecycle permit exactly
        reproduces :meth:`update`; it creates no shared or persistent state.
        """

        return self._update_role(
            old_state,
            decision,
            reward,
            value_write,
            lifecycle_write,
        )

    def update(
        self,
        old_state: SlotSignalingState,
        helper_decision: SlotRoleDecision,
        beneficiary_decision: SlotRoleDecision,
        reward: Array,
        *,
        helper_write: Array | bool = True,
        beneficiary_write: Array | bool = True,
    ) -> SlotSignalingUpdate:
        """Form both role candidates from the old joint state and commit atomically."""

        lifecycle_write = jnp.logical_and(
            jnp.asarray(helper_write, dtype=jnp.bool_),
            jnp.asarray(beneficiary_write, dtype=jnp.bool_),
        )
        helper = self.update_role(
            old_state.helper,
            helper_decision,
            reward,
            value_write=helper_write,
            lifecycle_write=lifecycle_write,
        )
        beneficiary = self.update_role(
            old_state.beneficiary,
            beneficiary_decision,
            reward,
            value_write=beneficiary_write,
            lifecycle_write=lifecycle_write,
        )
        state = SlotSignalingState(
            helper=helper.state,
            beneficiary=beneficiary.state,
        )
        return SlotSignalingUpdate(
            state=state,
            helper=helper,
            beneficiary=beneficiary,
            lifecycle_synchronized=_lifecycle_equal(state.helper, state.beneficiary),
        )


def slot_role_resource_budget(state: SlotRoleState) -> SlotRoleResourceBudget:
    """Measure exact persistent state, including separate relevance memory."""

    values = np.asarray(state.values)
    relevance_arrays = (
        np.asarray(state.relevance_mean),
        np.asarray(state.relevance_mass),
    )
    lifecycle_arrays = (
        np.asarray(state.failed_leases),
        np.asarray(state.idle_leases),
        np.asarray(state.status),
        np.asarray(state.generation),
        np.asarray(state.active_slot),
        np.asarray(state.lease_offset),
        np.asarray(state.lease_reward_sum),
        np.asarray(state.remaining_durable_tests),
        np.asarray(state.search_cursor),
        np.asarray(state.candidate_successful_leases),
        np.asarray(state.next_generation),
    )
    key = np.asarray(jr.key_data(state.key))
    value_scalars = int(values.size)
    relevance_scalars = sum(int(array.size) for array in relevance_arrays)
    lifecycle_scalars = sum(int(array.size) for array in lifecycle_arrays)
    key_scalars = int(key.size)
    return SlotRoleResourceBudget(
        value_scalars=value_scalars,
        relevance_scalars=relevance_scalars,
        lifecycle_scalars=lifecycle_scalars,
        key_scalars=key_scalars,
        state_scalars=(value_scalars + relevance_scalars + lifecycle_scalars + key_scalars),
        state_bytes=int(
            values.nbytes
            + sum(array.nbytes for array in relevance_arrays)
            + sum(array.nbytes for array in lifecycle_arrays)
            + key.nbytes
        ),
    )


def slot_signaling_resource_budget(
    state: SlotSignalingState,
) -> SlotSignalingResourceBudget:
    """Return the exact matched resource budget for both physical roles."""

    helper = slot_role_resource_budget(state.helper)
    beneficiary = slot_role_resource_budget(state.beneficiary)
    return SlotSignalingResourceBudget(
        helper=helper,
        beneficiary=beneficiary,
        state_scalars=helper.state_scalars + beneficiary.state_scalars,
        state_bytes=helper.state_bytes + beneficiary.state_bytes,
    )
