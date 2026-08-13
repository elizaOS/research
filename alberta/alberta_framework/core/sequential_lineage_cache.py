# mypy: disable-error-code="call-arg,name-defined"
"""Fixed H=2 sequential evidence for a bounded cross-birth lineage cache.

The mechanism is a small, policy-agnostic sidecar.  One archived reward model
may identify a newly allocated semantic birth only after two consecutive,
post-outcome comparisons against a deterministic fresh prior and every model
from the full pre-allocation live bank.  Confirmation delegates to the shared
pairwise-dominance quarantine law; this module adds birth consistency checks,
one-event frozen comparator snapshots, exact archive lifecycle handling, and
an atomic fail-closed transaction.

Opening evidence can never alter the action, reward, allocation, or eviction
that exposed it.  A confirmed lineage transfer similarly affects only future
eviction-protection scores.  Archived reward weights are an evaluator-side
recurrence fingerprint: this mechanism never transplants them into a live
learner model.

The host supplies ``source_reward_weights`` and the assertion that its context
update, allocation, and eviction have already completed.  This sidecar cannot
authenticate those host facts.  A composing outer transaction must capture the
pre-update weights, finish the host transition, call :meth:`propose`, and commit
both candidates atomically.  ``HOST_TRANSITION_BINDING_CLAIMED`` is therefore
deliberately false.

Every mechanism-produced state also carries an unkeyed SHA256 content token
over its configuration token and every mutable payload field.  This detects a
stale token after field-level mutation and binds pending evidence to the exact
payload covered by that token.  It is integrity checking, not external provenance or
tamper-proof authentication: callers with authority to reproduce the internal
digest can construct another self-consistent state.  Only mechanism-produced
state is supported, and ``EXTERNAL_STATE_PROVENANCE_CLAIMED`` remains false.

Agent namespace is an external identity component.  Every identity stored
here is therefore interpreted as ``(agent_namespace, exact_birth_words)``;
word equality across agents is not semantic equality.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import struct
from numbers import Real
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core.pairwise_dominance_quarantine import (
    PAIRWISE_DOMINANCE_DECISION_SCHEMA,
    PAIRWISE_DOMINANCE_OBSERVATION_SCHEMA,
    TWO_EVENT_PAIRWISE_DOMINANCE_HORIZON,
    pairwise_dominance_observation,
    resolve_two_event_pairwise_dominance,
)

SEQUENTIAL_LINEAGE_CACHE_CONFIG_SCHEMA = "alberta.sequential-lineage-cache.config.v1"
SEQUENTIAL_LINEAGE_CACHE_STATE_SCHEMA = "alberta.sequential-lineage-cache.state.v1"
SEQUENTIAL_LINEAGE_CACHE_EVENT_SCHEMA = "alberta.sequential-lineage-cache.event.v1"
SEQUENTIAL_LINEAGE_CACHE_PROPOSAL_SCHEMA = "alberta.sequential-lineage-cache.proposal.v1"
SEQUENTIAL_LINEAGE_CACHE_RESOURCE_SCHEMA = "alberta.sequential-lineage-cache.resource.v1"
SEQUENTIAL_LINEAGE_CACHE_WORK_SCHEMA = "alberta.sequential-lineage-cache.work.v1"
SEQUENTIAL_LINEAGE_CACHE_CONFIRMATION_HORIZON = TWO_EVENT_PAIRWISE_DOMINANCE_HORIZON
SEQUENTIAL_LINEAGE_CACHE_ARCHIVE_CAPACITY = 1
HOST_TRANSITION_BINDING_CLAIMED = False
STATE_CONTENT_INTEGRITY_CLAIMED = True
EXTERNAL_STATE_PROVENANCE_CLAIMED = False

ARCHIVE_SOURCE_NONE = 0
ARCHIVE_SOURCE_OLD_CACHE = 1
ARCHIVE_SOURCE_OPENING_VICTIM = 2
ARCHIVE_SOURCE_CURRENT_VICTIM = 3

_UINT32_MAX = 2**32 - 1
_STATE_CONTENT_TOKEN_NBYTES = 32


def _float32_bytes(value: Any, *, name: str) -> bytes:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite float32-representable real")
    try:
        packed = struct.pack(">f", float(value))
    except (OverflowError, struct.error, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite float32-representable real") from error
    if not math.isfinite(struct.unpack(">f", packed)[0]):
        raise ValueError(f"{name} must be a finite float32-representable real")
    return packed


@dataclasses.dataclass(frozen=True)
class SequentialLineageCacheConfig:
    """Static fixed-bank geometry and deterministic fresh prior."""

    max_contexts: int
    n_actions: int
    observation_dim: int
    initial_reward_estimate: float

    def __post_init__(self) -> None:
        if type(self.max_contexts) is not int or self.max_contexts < 1:
            raise ValueError("max_contexts must be a positive integer")
        if type(self.n_actions) is not int or self.n_actions < 1:
            raise ValueError("n_actions must be a positive integer")
        if type(self.observation_dim) is not int or self.observation_dim < 1:
            raise ValueError("observation_dim must be a positive integer")
        _float32_bytes(self.initial_reward_estimate, name="initial_reward_estimate")

    @property
    def comparison_bank_size(self) -> int:
        """Candidate, fresh prior, and every pre-allocation live model."""

        return self.max_contexts + 2


@chex.dataclass(frozen=True)
class SequentialLineageArchiveRecord:
    """The only independently usable archive record."""

    valid: Bool[Array, ""]
    source_birth_words: UInt[Array, " 2"]
    lineage_words: UInt[Array, " 2"]
    rescue_words: UInt[Array, " 2"]
    reward_weights: Float[Array, "n_actions observation_dim"]


@chex.dataclass(frozen=True)
class SequentialLineagePendingEvidence:
    """One-event frozen source bank and first relational observation.

    ``candidate`` duplicates the complete opening archive.  A valid pending
    state requires byte-exact array equality between this snapshot and the
    locked root archive, so event two cannot silently use a different model.
    The opening victim's birth and weights are recovered from its integrity-bound
    row in the frozen source arrays; only its lineage and rescue counter require
    separate staging.
    """

    valid: Bool[Array, ""]
    candidate: SequentialLineageArchiveRecord
    target_birth_words: UInt[Array, " 2"]
    source_birth_words: UInt[Array, "max_contexts 2"]
    source_reward_weights: Float[Array, "max_contexts n_actions observation_dim"]
    victim_lineage_words: UInt[Array, " 2"]
    victim_rescue_words: UInt[Array, " 2"]
    first_never_worse: Bool[Array, " comparator_count"]
    first_ever_strict: Bool[Array, " comparator_count"]


@chex.dataclass(frozen=True)
class SequentialLineageCacheState:
    """Integrity-bound live lineages, one archive, and at most one quarantine."""

    config_token: UInt[Array, " 32"]
    content_token: UInt[Array, " 32"]
    bound_birth_words: UInt[Array, "max_contexts 2"]
    live_lineage_words: UInt[Array, "max_contexts 2"]
    live_rescue_words: UInt[Array, "max_contexts 2"]
    archive: SequentialLineageArchiveRecord
    pending: SequentialLineagePendingEvidence


@chex.dataclass(frozen=True)
class SequentialLineageCacheEvent:
    """One complete post-outcome context transaction supplied by the caller.

    ``source_reward_weights`` must be the finite pre-update, pre-allocation
    reward-model bank.  ``context_update_applied`` reports host completion but
    is not independently authenticated by this policy-agnostic sidecar.
    """

    source_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    source_birth_words: UInt[Array, "max_contexts 2"]
    post_birth_words: UInt[Array, "max_contexts 2"]
    source_in_use: Bool[Array, " max_contexts"]
    post_in_use: Bool[Array, " max_contexts"]
    source_reward_weights: Float[Array, "max_contexts n_actions observation_dim"]
    observation: Float[Array, " observation_dim"]
    action: Int[Array, ""]
    reward: Float[Array, ""]
    allocated: Bool[Array, ""]
    evicted: Bool[Array, ""]
    target_slot: Int[Array, ""]
    context_update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class SequentialLineageCacheProposal:
    """Atomic successor and complete causal diagnostics for one event."""

    state: SequentialLineageCacheState
    source_state_valid: Bool[Array, ""]
    event_valid: Bool[Array, ""]
    predictive_inputs_finite: Bool[Array, ""]
    evidence_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    rescue_capacity_available: Bool[Array, ""]
    full_bank_birth: Bool[Array, ""]
    cache_tested: Bool[Array, ""]
    quarantine_opened: Bool[Array, ""]
    quarantine_second_evidence: Bool[Array, ""]
    quarantine_confirmed: Bool[Array, ""]
    quarantine_rejected: Bool[Array, ""]
    target_identity_matched: Bool[Array, ""]
    target_survived: Bool[Array, ""]
    confirmation_commit_abstained: Bool[Array, ""]
    lineage_transferred: Bool[Array, ""]
    rescue_incremented: Bool[Array, ""]
    victim_staged: Bool[Array, ""]
    overlap_full_bank_birth: Bool[Array, ""]
    new_quarantine_suppressed: Bool[Array, ""]
    archive_locked_during_pending: Bool[Array, ""]
    archive_selected_source: Int[Array, ""]
    archive_old_retained: Bool[Array, ""]
    archive_opening_victim_selected: Bool[Array, ""]
    archive_current_victim_selected: Bool[Array, ""]
    parameter_transplanted: Bool[Array, ""]
    predictions: Float[Array, " bank_size"]
    losses: Float[Array, " bank_size"]
    comparator_mask: Bool[Array, " bank_size"]
    never_worse: Bool[Array, " bank_size"]
    ever_strict: Bool[Array, " bank_size"]
    update_applied: Bool[Array, ""]


@dataclasses.dataclass(frozen=True)
class SequentialLineageCacheResourceRecord:
    """Exact persistent bytes and named selected logical proposal outputs.

    Logical output fields do not estimate branch intermediates, compiler
    fusion, allocator residency, or a physical transient-memory peak.
    """

    schema: str
    config_schema: str
    state_schema: str
    event_schema: str
    proposal_schema: str
    pairwise_observation_schema: str
    pairwise_decision_schema: str
    confirmation_horizon: int
    max_contexts: int
    n_actions: int
    observation_dim: int
    comparison_bank_size: int
    archive_capacity_per_agent: int
    config_token_nbytes_per_agent: int
    content_token_nbytes_per_agent: int
    base_lineage_archive_nbytes_per_agent: int
    pending_evidence_nbytes_per_agent: int
    frozen_source_snapshot_nbytes_per_agent: int
    frozen_candidate_snapshot_nbytes_per_agent: int
    per_agent_state_nbytes: int
    n_agents: int
    joint_state_nbytes: int
    base_scan_carry_nbytes: int
    total_scan_carry_nbytes: int
    logical_prediction_and_error_nbytes: int
    logical_selected_pairwise_diagnostic_nbytes: int
    logical_atomic_candidate_nbytes: int
    replay_capacity: int
    parameter_transplant_allowed: bool
    host_transition_binding_claimed: bool
    state_content_integrity_claimed: bool
    external_state_provenance_claimed: bool
    persistent_capacity_growth: int


@dataclasses.dataclass(frozen=True)
class SequentialLineageCacheWorkRecord:
    """Exact named logical counts for one fixed standalone invocation schedule.

    This record does not claim matched outer work.  An evaluator must invoke
    the sidecar equally in every condition and compare these records exactly.
    The named counts are not an exhaustive primitive-operation or compiled-FLOP
    total: state-contract booleans, finite checks, SHA bit operations, branch
    intermediates, compiler fusion, and allocator behavior are outside scope.
    SHA message words, compression blocks, and rounds are exact algorithmic
    units, not counts of the primitive operations inside a round.
    """

    schema: str
    confirmation_horizon: int
    total_steps: int
    n_agents: int
    prediction_bank_calls: int
    scalar_predictions: int
    absolute_losses: int
    pairwise_observation_calls: int
    pairwise_resolution_calls: int
    eligible_candidate_comparator_cells: int
    eligible_relational_comparisons: int
    executed_relational_vector_cells: int
    executed_relational_scalar_comparisons: int
    coefficient_products: int
    dot_additions: int
    archive_compare_selects: int
    state_audits: int
    content_digest_evaluations: int
    content_digest_message_words_per_evaluation: int
    content_digest_compression_blocks_per_evaluation: int
    content_digest_compression_blocks: int
    content_digest_rounds: int
    transaction_proposals: int
    snapshot_candidate_constructions: int
    replay_updates: int
    random_draws: int
    reset_callbacks: int
    maximum_lineage_transfers_per_agent_event: int
    standalone_schedule_fixed_per_invocation: bool
    outer_invocation_parity_required: bool
    matched_outer_work_claimed: bool
    exhaustive_primitive_operation_count_claimed: bool
    compiled_flop_count_claimed: bool


@chex.dataclass(frozen=True)
class _RelationalEvidence:
    comparator_mask: Bool[Array, " bank_size"]
    never_worse: Bool[Array, " bank_size"]
    ever_strict: Bool[Array, " bank_size"]
    valid: Bool[Array, ""]
    confirmed: Bool[Array, ""]
    rejected: Bool[Array, ""]


_SHA256_INITIAL = (
    0x6A09E667,
    0xBB67AE85,
    0x3C6EF372,
    0xA54FF53A,
    0x510E527F,
    0x9B05688C,
    0x1F83D9AB,
    0x5BE0CD19,
)
_SHA256_ROUND_CONSTANTS = (
    0x428A2F98,
    0x71374491,
    0xB5C0FBCF,
    0xE9B5DBA5,
    0x3956C25B,
    0x59F111F1,
    0x923F82A4,
    0xAB1C5ED5,
    0xD807AA98,
    0x12835B01,
    0x243185BE,
    0x550C7DC3,
    0x72BE5D74,
    0x80DEB1FE,
    0x9BDC06A7,
    0xC19BF174,
    0xE49B69C1,
    0xEFBE4786,
    0x0FC19DC6,
    0x240CA1CC,
    0x2DE92C6F,
    0x4A7484AA,
    0x5CB0A9DC,
    0x76F988DA,
    0x983E5152,
    0xA831C66D,
    0xB00327C8,
    0xBF597FC7,
    0xC6E00BF3,
    0xD5A79147,
    0x06CA6351,
    0x14292967,
    0x27B70A85,
    0x2E1B2138,
    0x4D2C6DFC,
    0x53380D13,
    0x650A7354,
    0x766A0ABB,
    0x81C2C92E,
    0x92722C85,
    0xA2BFE8A1,
    0xA81A664B,
    0xC24B8B70,
    0xC76C51A3,
    0xD192E819,
    0xD6990624,
    0xF40E3585,
    0x106AA070,
    0x19A4C116,
    0x1E376C08,
    0x2748774C,
    0x34B0BCB5,
    0x391C0CB3,
    0x4ED8AA4A,
    0x5B9CCA4F,
    0x682E6FF3,
    0x748F82EE,
    0x78A5636F,
    0x84C87814,
    0x8CC70208,
    0x90BEFFFA,
    0xA4506CEB,
    0xBEF9A3F7,
    0xC67178F2,
)


def _rotate_right(value: Array, amount: int) -> Array:
    return cast(
        Array,
        (value >> jnp.uint32(amount)) | (value << jnp.uint32(32 - amount)),
    )


@jax.jit
def _sha256_word_message(words: Array) -> Array:
    """Hash a canonical big-endian uint32-word message with SHA256."""

    message_words = words.shape[0]
    padded_words = ((message_words + 3 + 15) // 16) * 16
    bit_length = message_words * 32
    padded = jnp.zeros((padded_words,), dtype=jnp.uint32)
    padded = padded.at[:message_words].set(words)
    padded = padded.at[message_words].set(jnp.uint32(0x80000000))
    padded = padded.at[-2].set(jnp.uint32((bit_length >> 32) & _UINT32_MAX))
    padded = padded.at[-1].set(jnp.uint32(bit_length & _UINT32_MAX))
    blocks = padded.reshape((-1, 16))
    round_constants = jnp.asarray(_SHA256_ROUND_CONSTANTS, dtype=jnp.uint32)

    def compress(hash_words: Array, block: Array) -> tuple[Array, None]:
        schedule = jnp.zeros((64,), dtype=jnp.uint32).at[:16].set(block)

        def extend(index: int, current: Array) -> Array:
            left = current[index - 15]
            right = current[index - 2]
            sigma_zero = _rotate_right(left, 7) ^ _rotate_right(left, 18) ^ (left >> 3)
            sigma_one = _rotate_right(right, 17) ^ _rotate_right(right, 19) ^ (right >> 10)
            value = current[index - 16] + sigma_zero + current[index - 7] + sigma_one
            return current.at[index].set(value)

        schedule = jax.lax.fori_loop(16, 64, extend, schedule)

        def round_step(index: int, working: tuple[Array, ...]) -> tuple[Array, ...]:
            a, b, c, d, e, f, g, h = working
            sum_one = _rotate_right(e, 6) ^ _rotate_right(e, 11) ^ _rotate_right(e, 25)
            choice = (e & f) ^ ((~e) & g)
            temporary_one = h + sum_one + choice + round_constants[index] + schedule[index]
            sum_zero = _rotate_right(a, 2) ^ _rotate_right(a, 13) ^ _rotate_right(a, 22)
            majority = (a & b) ^ (a & c) ^ (b & c)
            temporary_two = sum_zero + majority
            return (
                temporary_one + temporary_two,
                a,
                b,
                c,
                d + temporary_one,
                e,
                f,
                g,
            )

        working = tuple(hash_words[index] for index in range(8))
        working = jax.lax.fori_loop(0, 64, round_step, working)
        return hash_words + jnp.stack(working), None

    initial = jnp.asarray(_SHA256_INITIAL, dtype=jnp.uint32)
    digest_words, _ = jax.lax.scan(compress, initial, blocks)
    shifts = jnp.asarray((24, 16, 8, 0), dtype=jnp.uint32)
    return cast(
        Array,
        ((digest_words[:, None] >> shifts[None, :]) & jnp.uint32(0xFF))
        .astype(jnp.uint8)
        .reshape((_STATE_CONTENT_TOKEN_NBYTES,)),
    )


def _packed_token_words(token: Array) -> Array:
    values = token.reshape((-1, 4)).astype(jnp.uint32)
    return cast(
        Array,
        (values[:, 0] << jnp.uint32(24))
        | (values[:, 1] << jnp.uint32(16))
        | (values[:, 2] << jnp.uint32(8))
        | values[:, 3],
    )


def _float_words(values: Array) -> Array:
    return jax.lax.bitcast_convert_type(values.reshape((-1,)), jnp.uint32)


def _archive_content_words(record: SequentialLineageArchiveRecord) -> Array:
    return jnp.concatenate(
        (
            record.valid.reshape((1,)).astype(jnp.uint32),
            record.source_birth_words.reshape((-1,)),
            record.lineage_words.reshape((-1,)),
            record.rescue_words.reshape((-1,)),
            _float_words(record.reward_weights),
        )
    )


def _state_content_words(state: SequentialLineageCacheState) -> Array:
    pending = state.pending
    return jnp.concatenate(
        (
            _packed_token_words(state.config_token),
            state.bound_birth_words.reshape((-1,)),
            state.live_lineage_words.reshape((-1,)),
            state.live_rescue_words.reshape((-1,)),
            _archive_content_words(state.archive),
            pending.valid.reshape((1,)).astype(jnp.uint32),
            _archive_content_words(pending.candidate),
            pending.target_birth_words.reshape((-1,)),
            pending.source_birth_words.reshape((-1,)),
            _float_words(pending.source_reward_weights),
            pending.victim_lineage_words.reshape((-1,)),
            pending.victim_rescue_words.reshape((-1,)),
            pending.first_never_worse.reshape((-1,)).astype(jnp.uint32),
            pending.first_ever_strict.reshape((-1,)).astype(jnp.uint32),
        )
    )


@jax.jit
def _state_content_token(state: SequentialLineageCacheState) -> Array:
    return cast(Array, _sha256_word_message(_state_content_words(state)))


def _state_content_word_count(config: SequentialLineageCacheConfig) -> int:
    return 31 + 10 * config.max_contexts + config.n_actions * config.observation_dim * (
        config.max_contexts + 2
    )


def _config_token(config: SequentialLineageCacheConfig) -> Array:
    """Return a stable SHA256 token over every behavior-affecting field."""

    geometry = (
        f"{SEQUENTIAL_LINEAGE_CACHE_CONFIG_SCHEMA}\n"
        f"{config.max_contexts}\n{config.n_actions}\n{config.observation_dim}\n"
    ).encode("ascii")
    digest = hashlib.sha256(
        geometry
        + _float32_bytes(
            config.initial_reward_estimate,
            name="initial_reward_estimate",
        )
    ).digest()
    return jnp.asarray(tuple(digest), dtype=jnp.uint8)


def _require_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    array = jnp.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    expected_dtype = jnp.dtype(dtype)
    if array.dtype != expected_dtype:
        raise TypeError(f"{name} must have dtype {expected_dtype}, got {array.dtype}")
    return array


def _tree_nbytes(tree: Any) -> int:
    return sum(
        int(jnp.asarray(leaf).size) * int(jnp.asarray(leaf).dtype.itemsize)
        for leaf in jax.tree.leaves(tree)
    )


def _tree_finite(tree: Any) -> Bool[Array, ""]:
    valid = jnp.asarray(True, dtype=jnp.bool_)
    for leaf in jax.tree.leaves(tree):
        array = jnp.asarray(leaf)
        if jnp.issubdtype(array.dtype, jnp.inexact):
            valid = valid & jnp.all(jnp.isfinite(array))
    return valid


def _words_equal(left: Array, right: Array) -> Bool[Array, ""]:
    return jnp.all(left == right)


def _words_le(left: Array, right: Array) -> Array:
    return (left[..., 0] < right[..., 0]) | (
        (left[..., 0] == right[..., 0]) & (left[..., 1] <= right[..., 1])
    )


def _words_lt(left: Array, right: Array) -> Array:
    return (left[..., 0] < right[..., 0]) | (
        (left[..., 0] == right[..., 0]) & (left[..., 1] < right[..., 1])
    )


def _checked_words_increment(words: Array) -> tuple[Array, Array]:
    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    capacity_available = ~jnp.all(words == maximum)
    low = words[1] + jnp.asarray(1, dtype=jnp.uint32)
    carry = (low == jnp.asarray(0, dtype=jnp.uint32)).astype(jnp.uint32)
    proposed = jnp.stack((words[0] + carry, low)).astype(jnp.uint32)
    return jnp.where(capacity_available, proposed, words), capacity_available


def _rows_unique(words: Array, mask: Array) -> Bool[Array, ""]:
    same = jnp.all(words[:, None, :] == words[None, :, :], axis=-1)
    used_pairs = mask[:, None] & mask[None, :]
    off_diagonal = ~jnp.eye(words.shape[0], dtype=jnp.bool_)
    return ~jnp.any(same & used_pairs & off_diagonal)


def _zero_archive(config: SequentialLineageCacheConfig) -> SequentialLineageArchiveRecord:
    return SequentialLineageArchiveRecord(
        valid=jnp.asarray(False, dtype=jnp.bool_),
        source_birth_words=jnp.zeros((2,), dtype=jnp.uint32),
        lineage_words=jnp.zeros((2,), dtype=jnp.uint32),
        rescue_words=jnp.zeros((2,), dtype=jnp.uint32),
        reward_weights=jnp.zeros(
            (config.n_actions, config.observation_dim),
            dtype=jnp.float32,
        ),
    )


def _zero_pending(
    config: SequentialLineageCacheConfig,
) -> SequentialLineagePendingEvidence:
    return SequentialLineagePendingEvidence(
        valid=jnp.asarray(False, dtype=jnp.bool_),
        candidate=_zero_archive(config),
        target_birth_words=jnp.zeros((2,), dtype=jnp.uint32),
        source_birth_words=jnp.zeros((config.max_contexts, 2), dtype=jnp.uint32),
        source_reward_weights=jnp.zeros(
            (config.max_contexts, config.n_actions, config.observation_dim),
            dtype=jnp.float32,
        ),
        victim_lineage_words=jnp.zeros((2,), dtype=jnp.uint32),
        victim_rescue_words=jnp.zeros((2,), dtype=jnp.uint32),
        first_never_worse=jnp.zeros((config.max_contexts + 1,), dtype=jnp.bool_),
        first_ever_strict=jnp.zeros((config.max_contexts + 1,), dtype=jnp.bool_),
    )


def _archive_payload_zero(record: SequentialLineageArchiveRecord) -> Bool[Array, ""]:
    return (
        jnp.all(record.source_birth_words == jnp.asarray(0, dtype=jnp.uint32))
        & jnp.all(record.lineage_words == jnp.asarray(0, dtype=jnp.uint32))
        & jnp.all(record.rescue_words == jnp.asarray(0, dtype=jnp.uint32))
        & jnp.all(record.reward_weights == jnp.float32(0.0))
    )


def _archives_equal(
    left: SequentialLineageArchiveRecord,
    right: SequentialLineageArchiveRecord,
) -> Bool[Array, ""]:
    return (
        jnp.array_equal(left.valid, right.valid)
        & jnp.array_equal(left.source_birth_words, right.source_birth_words)
        & jnp.array_equal(left.lineage_words, right.lineage_words)
        & jnp.array_equal(left.rescue_words, right.rescue_words)
        & jnp.array_equal(left.reward_weights, right.reward_weights)
    )


def _pending_payload_zero(
    pending: SequentialLineagePendingEvidence,
) -> Bool[Array, ""]:
    return (
        ~pending.candidate.valid
        & _archive_payload_zero(pending.candidate)
        & jnp.all(pending.target_birth_words == jnp.asarray(0, dtype=jnp.uint32))
        & jnp.all(pending.source_birth_words == jnp.asarray(0, dtype=jnp.uint32))
        & jnp.all(pending.source_reward_weights == jnp.float32(0.0))
        & jnp.all(pending.victim_lineage_words == jnp.asarray(0, dtype=jnp.uint32))
        & jnp.all(pending.victim_rescue_words == jnp.asarray(0, dtype=jnp.uint32))
        & ~jnp.any(pending.first_never_worse)
        & ~jnp.any(pending.first_ever_strict)
    )


def _record_with_valid(
    record: SequentialLineageArchiveRecord,
    valid: Array,
) -> SequentialLineageArchiveRecord:
    return SequentialLineageArchiveRecord(
        valid=jnp.asarray(valid, dtype=jnp.bool_),
        source_birth_words=record.source_birth_words,
        lineage_words=record.lineage_words,
        rescue_words=record.rescue_words,
        reward_weights=record.reward_weights,
    )


def _select_archive(
    left: SequentialLineageArchiveRecord,
    left_source: Array,
    right: SequentialLineageArchiveRecord,
    right_source: int,
) -> tuple[SequentialLineageArchiveRecord, Array]:
    """Choose greater exact rescue count, then newer source birth."""

    rescue_greater = _words_lt(left.rescue_words, right.rescue_words)
    rescue_equal = _words_equal(left.rescue_words, right.rescue_words)
    right_newer_or_equal = _words_le(left.source_birth_words, right.source_birth_words)
    choose_right = right.valid & (
        ~left.valid | rescue_greater | (rescue_equal & right_newer_or_equal)
    )
    selected = cast(
        SequentialLineageArchiveRecord,
        jax.tree_util.tree_map(
            lambda right_value, left_value: jnp.where(
                choose_right,
                right_value,
                left_value,
            ),
            right,
            left,
        ),
    )
    source = jnp.where(
        choose_right,
        jnp.asarray(right_source, dtype=jnp.int32),
        left_source,
    )
    return selected, source


class SequentialLineageCache:
    """Pure fixed-capacity H=2 cross-birth lineage-cache transaction."""

    def __init__(self, config: SequentialLineageCacheConfig):
        self._config = config
        self._config_token = _config_token(config)

    @property
    def config(self) -> SequentialLineageCacheConfig:
        return self._config

    def init(self) -> SequentialLineageCacheState:
        """Return exact empty sidecar state."""

        c = self._config
        zero_rows = jnp.zeros((c.max_contexts, 2), dtype=jnp.uint32)
        unsigned = SequentialLineageCacheState(
            config_token=self._config_token,
            content_token=jnp.zeros((_STATE_CONTENT_TOKEN_NBYTES,), dtype=jnp.uint8),
            bound_birth_words=zero_rows,
            live_lineage_words=zero_rows,
            live_rescue_words=zero_rows,
            archive=_zero_archive(c),
            pending=_zero_pending(c),
        )
        return self._with_content_token(unsigned)

    def _with_content_token(
        self,
        state: SequentialLineageCacheState,
    ) -> SequentialLineageCacheState:
        """Reseal trusted state content; this private helper proves no provenance."""

        return cast(
            SequentialLineageCacheState,
            cast(Any, state).replace(content_token=_state_content_token(state)),
        )

    def _require_archive_contract(self, record: SequentialLineageArchiveRecord) -> None:
        c = self._config
        _require_array(record.valid, name="archive.valid", shape=(), dtype=jnp.bool_)
        for name in ("source_birth_words", "lineage_words", "rescue_words"):
            _require_array(
                getattr(record, name),
                name=f"archive.{name}",
                shape=(2,),
                dtype=jnp.uint32,
            )
        _require_array(
            record.reward_weights,
            name="archive.reward_weights",
            shape=(c.n_actions, c.observation_dim),
            dtype=jnp.float32,
        )

    def _require_pending_contract(
        self,
        pending: SequentialLineagePendingEvidence,
    ) -> None:
        c = self._config
        _require_array(pending.valid, name="pending.valid", shape=(), dtype=jnp.bool_)
        self._require_archive_contract(pending.candidate)
        _require_array(
            pending.target_birth_words,
            name="pending.target_birth_words",
            shape=(2,),
            dtype=jnp.uint32,
        )
        _require_array(
            pending.source_birth_words,
            name="pending.source_birth_words",
            shape=(c.max_contexts, 2),
            dtype=jnp.uint32,
        )
        _require_array(
            pending.source_reward_weights,
            name="pending.source_reward_weights",
            shape=(c.max_contexts, c.n_actions, c.observation_dim),
            dtype=jnp.float32,
        )
        for name in ("victim_lineage_words", "victim_rescue_words"):
            _require_array(
                getattr(pending, name),
                name=f"pending.{name}",
                shape=(2,),
                dtype=jnp.uint32,
            )
        for name in ("first_never_worse", "first_ever_strict"):
            _require_array(
                getattr(pending, name),
                name=f"pending.{name}",
                shape=(c.max_contexts + 1,),
                dtype=jnp.bool_,
            )

    def _require_state_contract(self, state: SequentialLineageCacheState) -> None:
        c = self._config
        _require_array(
            state.config_token,
            name="config_token",
            shape=(32,),
            dtype=jnp.uint8,
        )
        _require_array(
            state.content_token,
            name="content_token",
            shape=(_STATE_CONTENT_TOKEN_NBYTES,),
            dtype=jnp.uint8,
        )
        for name in (
            "bound_birth_words",
            "live_lineage_words",
            "live_rescue_words",
        ):
            _require_array(
                getattr(state, name),
                name=name,
                shape=(c.max_contexts, 2),
                dtype=jnp.uint32,
            )
        self._require_archive_contract(state.archive)
        self._require_pending_contract(state.pending)

    def _require_external_state_contract(
        self,
        step_words: Any,
        birth_words: Any,
        in_use: Any,
    ) -> tuple[Array, Array, Array]:
        c = self._config
        return (
            _require_array(
                step_words,
                name="step_words",
                shape=(2,),
                dtype=jnp.uint32,
            ),
            _require_array(
                birth_words,
                name="birth_words",
                shape=(c.max_contexts, 2),
                dtype=jnp.uint32,
            ),
            _require_array(
                in_use,
                name="in_use",
                shape=(c.max_contexts,),
                dtype=jnp.bool_,
            ),
        )

    def _require_event_contract(self, event: SequentialLineageCacheEvent) -> None:
        c = self._config
        for name in ("source_step_words", "post_step_words"):
            _require_array(
                getattr(event, name),
                name=f"event.{name}",
                shape=(2,),
                dtype=jnp.uint32,
            )
        for name in ("source_birth_words", "post_birth_words"):
            _require_array(
                getattr(event, name),
                name=f"event.{name}",
                shape=(c.max_contexts, 2),
                dtype=jnp.uint32,
            )
        for name in ("source_in_use", "post_in_use"):
            _require_array(
                getattr(event, name),
                name=f"event.{name}",
                shape=(c.max_contexts,),
                dtype=jnp.bool_,
            )
        _require_array(
            event.source_reward_weights,
            name="event.source_reward_weights",
            shape=(c.max_contexts, c.n_actions, c.observation_dim),
            dtype=jnp.float32,
        )
        _require_array(
            event.observation,
            name="event.observation",
            shape=(c.observation_dim,),
            dtype=jnp.float32,
        )
        _require_array(event.action, name="event.action", shape=(), dtype=jnp.int32)
        _require_array(event.reward, name="event.reward", shape=(), dtype=jnp.float32)
        for name in ("allocated", "evicted", "context_update_applied"):
            _require_array(
                getattr(event, name),
                name=f"event.{name}",
                shape=(),
                dtype=jnp.bool_,
            )
        _require_array(
            event.target_slot,
            name="event.target_slot",
            shape=(),
            dtype=jnp.int32,
        )

    def state_valid(
        self,
        state: SequentialLineageCacheState,
        step_words: UInt[Array, " 2"],
        birth_words: UInt[Array, "max_contexts 2"],
        in_use: Bool[Array, " max_contexts"],
    ) -> Bool[Array, ""]:
        """Check trusted-state integrity and semantic consistency.

        The unkeyed content token detects stale field mutation but does not
        establish that an external caller obtained the state from this
        mechanism.  See ``EXTERNAL_STATE_PROVENANCE_CLAIMED``.
        """

        self._require_state_contract(state)
        checked_step, checked_births, checked_in_use = self._require_external_state_contract(
            step_words, birth_words, in_use
        )
        return self._state_valid_unchecked(
            state,
            checked_step,
            checked_births,
            checked_in_use,
        )

    def _state_valid_unchecked(
        self,
        state: SequentialLineageCacheState,
        step_words: Array,
        birth_words: Array,
        in_use: Array,
    ) -> Bool[Array, ""]:
        c = self._config
        zero_words = jnp.zeros((2,), dtype=jnp.uint32)
        config_bound = jnp.array_equal(state.config_token, self._config_token)
        content_bound = jnp.array_equal(state.content_token, _state_content_token(state))
        bound = jnp.all(state.bound_birth_words == birth_words, axis=1)
        live_rows_valid = (
            bound
            & _words_le(
                state.live_lineage_words,
                state.bound_birth_words,
            )
            & _words_le(
                state.live_rescue_words,
                state.bound_birth_words,
            )
        )
        unused_rows_zero = (
            jnp.all(birth_words == zero_words, axis=1)
            & jnp.all(state.bound_birth_words == zero_words, axis=1)
            & jnp.all(state.live_lineage_words == zero_words, axis=1)
            & jnp.all(state.live_rescue_words == zero_words, axis=1)
        )
        live_valid = (
            jnp.all(jnp.where(in_use, live_rows_valid, unused_rows_zero))
            & _rows_unique(birth_words, in_use)
            & _rows_unique(state.live_lineage_words, in_use)
            & jnp.all(jnp.where(in_use, _words_le(birth_words, step_words), True))
        )

        archive = state.archive
        archive_distinct = ~jnp.any(
            in_use
            & jnp.all(
                state.live_lineage_words == archive.lineage_words[None, :],
                axis=1,
            )
        )
        archive_source_absent = ~jnp.any(
            in_use
            & jnp.all(
                birth_words == archive.source_birth_words[None, :],
                axis=1,
            )
        )
        valid_archive_payload = (
            _words_le(archive.lineage_words, archive.source_birth_words)
            & _words_le(archive.rescue_words, archive.source_birth_words)
            & _words_le(archive.source_birth_words, step_words)
            & archive_distinct
            & archive_source_absent
            & jnp.all(jnp.isfinite(archive.reward_weights))
        )
        archive_valid = jnp.where(
            archive.valid,
            valid_archive_payload,
            _archive_payload_zero(archive),
        )

        pending = state.pending
        target_matches = in_use & jnp.all(
            birth_words == pending.target_birth_words[None, :],
            axis=1,
        )
        target_count = jnp.sum(target_matches.astype(jnp.int32))
        safe_target = jnp.argmax(target_matches.astype(jnp.int32)).astype(jnp.int32)
        expected_current_births = pending.source_birth_words.at[safe_target].set(
            pending.target_birth_words
        )
        all_pending_rows = jnp.ones((c.max_contexts,), dtype=jnp.bool_)
        pending_victim_distinct = ~jnp.any(
            in_use
            & jnp.all(
                state.live_lineage_words == pending.victim_lineage_words[None, :],
                axis=1,
            )
        )
        candidate_source_absent = ~jnp.any(
            jnp.all(
                pending.source_birth_words
                == pending.candidate.source_birth_words[None, :],
                axis=1,
            )
        )
        target_lineage_fresh = _words_equal(
            state.live_lineage_words[safe_target],
            pending.target_birth_words,
        ) & _words_equal(
            state.live_rescue_words[safe_target],
            zero_words,
        )
        pending_payload_valid = (
            archive.valid
            & pending.candidate.valid
            & _archives_equal(pending.candidate, archive)
            & jnp.all(in_use)
            & _words_equal(pending.target_birth_words, step_words)
            & (target_count == 1)
            & target_lineage_fresh
            & jnp.all(expected_current_births == birth_words)
            & jnp.all(
                _words_lt(
                    pending.source_birth_words,
                    pending.target_birth_words[None, :],
                )
            )
            & _rows_unique(pending.source_birth_words, all_pending_rows)
            & _words_le(
                pending.victim_lineage_words,
                pending.source_birth_words[safe_target],
            )
            & _words_le(
                pending.victim_rescue_words,
                pending.source_birth_words[safe_target],
            )
            & candidate_source_absent
            & ~_words_equal(
                pending.candidate.lineage_words,
                pending.victim_lineage_words,
            )
            & pending_victim_distinct
            & jnp.all(jnp.isfinite(pending.source_reward_weights))
            & jnp.all((~pending.first_ever_strict) | pending.first_never_worse)
        )
        pending_valid = jnp.where(
            pending.valid,
            pending_payload_valid,
            _pending_payload_zero(pending),
        )
        return config_bound & content_bound & live_valid & archive_valid & pending_valid

    def _event_valid_unchecked(self, event: SequentialLineageCacheEvent) -> Bool[Array, ""]:
        c = self._config
        proposed_step, step_capacity = _checked_words_increment(event.source_step_words)
        clock_valid = step_capacity & _words_equal(proposed_step, event.post_step_words)
        target_valid = (event.target_slot >= 0) & (event.target_slot < c.max_contexts)
        safe_target = jnp.clip(event.target_slot, 0, c.max_contexts - 1)
        row_is_target = jnp.arange(c.max_contexts, dtype=jnp.int32) == safe_target
        other_births_same = jnp.all(
            jnp.where(
                row_is_target[:, None],
                jnp.asarray(True, dtype=jnp.bool_),
                event.source_birth_words == event.post_birth_words,
            )
        )
        other_use_same = jnp.all(
            jnp.where(
                row_is_target,
                jnp.asarray(True, dtype=jnp.bool_),
                event.source_in_use == event.post_in_use,
            )
        )
        source_full = jnp.all(event.source_in_use)
        allocation_valid = (
            target_valid
            & other_births_same
            & other_use_same
            & event.post_in_use[safe_target]
            & _words_equal(
                event.post_birth_words[safe_target],
                event.post_step_words,
            )
            & jnp.where(
                event.evicted,
                source_full & event.source_in_use[safe_target],
                (~source_full) & (~event.source_in_use[safe_target]),
            )
        )
        no_allocation_valid = (
            ~event.evicted
            & jnp.all(event.source_birth_words == event.post_birth_words)
            & jnp.all(event.source_in_use == event.post_in_use)
        )
        lifecycle_valid = jnp.where(
            event.allocated,
            allocation_valid,
            no_allocation_valid,
        )
        zero_words = jnp.zeros((2,), dtype=jnp.uint32)
        source_unused_zero = jnp.all(
            jnp.where(
                event.source_in_use[:, None],
                jnp.asarray(True, dtype=jnp.bool_),
                event.source_birth_words == zero_words,
            )
        )
        post_unused_zero = jnp.all(
            jnp.where(
                event.post_in_use[:, None],
                jnp.asarray(True, dtype=jnp.bool_),
                event.post_birth_words == zero_words,
            )
        )
        predictive_inputs_finite = (
            jnp.all(jnp.isfinite(event.source_reward_weights))
            & jnp.all(jnp.isfinite(event.observation))
            & jnp.isfinite(event.reward)
            & (event.action >= 0)
            & (event.action < c.n_actions)
        )
        return (
            event.context_update_applied
            & clock_valid
            & target_valid
            & lifecycle_valid
            & source_unused_zero
            & post_unused_zero
            & predictive_inputs_finite
        )

    def _prediction_bank(
        self,
        state: SequentialLineageCacheState,
        event: SequentialLineageCacheEvent,
    ) -> tuple[Array, Array, Array]:
        c = self._config
        safe_action = jnp.clip(event.action, 0, c.n_actions - 1)
        comparator_weights = jnp.where(
            state.pending.valid,
            state.pending.source_reward_weights,
            event.source_reward_weights,
        )
        candidate_weights = jnp.where(
            state.pending.valid,
            state.pending.candidate.reward_weights,
            state.archive.reward_weights,
        )
        archive_prediction = candidate_weights[safe_action] @ event.observation
        fresh_prediction = (
            jnp.full(
                (c.observation_dim,),
                jnp.float32(c.initial_reward_estimate),
                dtype=jnp.float32,
            )
            @ event.observation
        )
        live_predictions = comparator_weights[:, safe_action, :] @ event.observation
        predictions = jnp.concatenate(
            (
                archive_prediction[None],
                fresh_prediction[None],
                live_predictions,
            )
        ).astype(jnp.float32)
        losses = jnp.abs(event.reward - predictions).astype(jnp.float32)
        finite = (
            jnp.all(jnp.isfinite(event.source_reward_weights))
            & jnp.all(jnp.isfinite(event.observation))
            & jnp.isfinite(event.reward)
            & (event.action >= 0)
            & (event.action < c.n_actions)
            & jnp.all(jnp.isfinite(predictions))
            & jnp.all(jnp.isfinite(losses))
        )
        return predictions, losses, finite

    def _relational_evidence(
        self,
        state: SequentialLineageCacheState,
        losses: Array,
    ) -> _RelationalEvidence:
        c = self._config
        candidate = jnp.asarray(0, dtype=jnp.int32)
        false = jnp.asarray(False, dtype=jnp.bool_)
        opening = pairwise_dominance_observation(losses, candidate)
        comparator_mask = jnp.concatenate(
            (
                jnp.asarray((False,), dtype=jnp.bool_),
                jnp.ones((c.max_contexts + 1,), dtype=jnp.bool_),
            )
        )
        first_never = jnp.concatenate(
            (
                jnp.asarray((False,), dtype=jnp.bool_),
                state.pending.first_never_worse,
            )
        )
        first_strict = jnp.concatenate(
            (
                jnp.asarray((False,), dtype=jnp.bool_),
                state.pending.first_ever_strict,
            )
        )
        resolved = resolve_two_event_pairwise_dominance(
            comparator_mask,
            first_never,
            first_strict,
            losses,
            candidate,
        )
        use_resolved = state.pending.valid
        return _RelationalEvidence(
            comparator_mask=jnp.where(
                use_resolved,
                resolved.comparator_mask,
                opening.comparator_mask,
            ),
            never_worse=jnp.where(
                use_resolved,
                resolved.never_worse,
                opening.never_worse,
            ),
            ever_strict=jnp.where(
                use_resolved,
                resolved.ever_strict,
                opening.ever_strict,
            ),
            valid=jnp.where(use_resolved, resolved.valid, opening.valid),
            confirmed=jnp.where(use_resolved, resolved.confirmed, false),
            rejected=jnp.where(use_resolved, resolved.rejected, false),
        )

    def propose(
        self,
        state: SequentialLineageCacheState,
        event: SequentialLineageCacheEvent,
    ) -> SequentialLineageCacheProposal:
        """Propose one integrity-checked post-outcome transaction.

        Static shape or dtype mismatches raise immediately.  Every semantic,
        timing, evidence, capacity, and candidate-state failure is fail-closed:
        no partial sidecar mutation is returned.  A valid H=2 confirmation
        whose target disappeared is a recorded abstention, not a partial
        transfer.  Reachability bounds reject terminal rescue counters before
        they can participate in a transaction.
        """

        self._require_state_contract(state)
        self._require_event_contract(event)
        c = self._config
        source_state_valid = self._state_valid_unchecked(
            state,
            event.source_step_words,
            event.source_birth_words,
            event.source_in_use,
        )
        event_valid = self._event_valid_unchecked(event)
        predictions, losses, predictive_inputs_finite = self._prediction_bank(
            state,
            event,
        )
        relation = self._relational_evidence(state, losses)
        evidence_valid = predictive_inputs_finite & relation.valid

        full_bank_birth = event.allocated & event.evicted
        cache_tested = (~state.pending.valid) & full_bank_birth & state.archive.valid
        quarantine_opened = cache_tested & evidence_valid
        quarantine_second = state.pending.valid
        quarantine_confirmed = quarantine_second & evidence_valid & relation.confirmed
        quarantine_rejected = quarantine_second & evidence_valid & relation.rejected
        overlap = quarantine_second & full_bank_birth

        source_target_matches = event.source_in_use & jnp.all(
            event.source_birth_words == state.pending.target_birth_words[None, :],
            axis=1,
        )
        source_target_count = jnp.sum(source_target_matches.astype(jnp.int32))
        pending_target_slot = jnp.argmax(source_target_matches.astype(jnp.int32)).astype(jnp.int32)
        target_identity_matched = quarantine_second & (source_target_count == 1)
        post_target_matches = event.post_in_use & jnp.all(
            event.post_birth_words == state.pending.target_birth_words[None, :],
            axis=1,
        )
        post_target_count = jnp.sum(post_target_matches.astype(jnp.int32))
        post_target_slot = jnp.argmax(post_target_matches.astype(jnp.int32)).astype(jnp.int32)
        target_survived = target_identity_matched & (post_target_count == 1)

        active_candidate = SequentialLineageArchiveRecord(
            valid=jnp.where(
                quarantine_second,
                state.pending.candidate.valid,
                state.archive.valid,
            ),
            source_birth_words=jnp.where(
                quarantine_second,
                state.pending.candidate.source_birth_words,
                state.archive.source_birth_words,
            ),
            lineage_words=jnp.where(
                quarantine_second,
                state.pending.candidate.lineage_words,
                state.archive.lineage_words,
            ),
            rescue_words=jnp.where(
                quarantine_second,
                state.pending.candidate.rescue_words,
                state.archive.rescue_words,
            ),
            reward_weights=jnp.where(
                quarantine_second,
                state.pending.candidate.reward_weights,
                state.archive.reward_weights,
            ),
        )
        incremented_rescue, rescue_capacity = _checked_words_increment(
            active_candidate.rescue_words
        )
        confirmation_committable = quarantine_confirmed & target_survived & rescue_capacity
        confirmation_abstained = quarantine_confirmed & ~confirmation_committable

        safe_current_target = jnp.clip(event.target_slot, 0, c.max_contexts - 1)
        allocated_bound = state.bound_birth_words.at[safe_current_target].set(
            event.post_birth_words[safe_current_target]
        )
        allocated_lineage = state.live_lineage_words.at[safe_current_target].set(
            event.post_birth_words[safe_current_target]
        )
        allocated_rescue = state.live_rescue_words.at[safe_current_target].set(
            jnp.zeros((2,), dtype=jnp.uint32)
        )
        bound_birth_words = jnp.where(
            event.allocated,
            allocated_bound,
            state.bound_birth_words,
        )
        live_lineage_words = jnp.where(
            event.allocated,
            allocated_lineage,
            state.live_lineage_words,
        )
        live_rescue_words = jnp.where(
            event.allocated,
            allocated_rescue,
            state.live_rescue_words,
        )
        live_lineage_words = jnp.where(
            confirmation_committable,
            live_lineage_words.at[post_target_slot].set(active_candidate.lineage_words),
            live_lineage_words,
        )
        live_rescue_words = jnp.where(
            confirmation_committable,
            live_rescue_words.at[post_target_slot].set(incremented_rescue),
            live_rescue_words,
        )

        opening_victim = SequentialLineageArchiveRecord(
            valid=state.pending.valid,
            source_birth_words=state.pending.source_birth_words[pending_target_slot],
            lineage_words=state.pending.victim_lineage_words,
            rescue_words=state.pending.victim_rescue_words,
            reward_weights=state.pending.source_reward_weights[pending_target_slot],
        )
        current_victim_allowed = full_bank_birth & (state.pending.valid | ~quarantine_opened)
        current_victim = SequentialLineageArchiveRecord(
            valid=current_victim_allowed,
            source_birth_words=event.source_birth_words[safe_current_target],
            lineage_words=state.live_lineage_words[safe_current_target],
            rescue_words=state.live_rescue_words[safe_current_target],
            reward_weights=event.source_reward_weights[safe_current_target],
        )
        old_archive = _record_with_valid(
            active_candidate,
            active_candidate.valid & ~confirmation_committable,
        )
        old_source = jnp.where(
            old_archive.valid,
            jnp.asarray(ARCHIVE_SOURCE_OLD_CACHE, dtype=jnp.int32),
            jnp.asarray(ARCHIVE_SOURCE_NONE, dtype=jnp.int32),
        )
        selected_archive, selected_source = _select_archive(
            old_archive,
            old_source,
            opening_victim,
            ARCHIVE_SOURCE_OPENING_VICTIM,
        )
        selected_archive, selected_source = _select_archive(
            selected_archive,
            selected_source,
            current_victim,
            ARCHIVE_SOURCE_CURRENT_VICTIM,
        )

        zero_pending = _zero_pending(c)
        proposed_pending = SequentialLineagePendingEvidence(
            valid=quarantine_opened,
            candidate=SequentialLineageArchiveRecord(
                valid=jnp.where(
                    quarantine_opened,
                    state.archive.valid,
                    zero_pending.candidate.valid,
                ),
                source_birth_words=jnp.where(
                    quarantine_opened,
                    state.archive.source_birth_words,
                    zero_pending.candidate.source_birth_words,
                ),
                lineage_words=jnp.where(
                    quarantine_opened,
                    state.archive.lineage_words,
                    zero_pending.candidate.lineage_words,
                ),
                rescue_words=jnp.where(
                    quarantine_opened,
                    state.archive.rescue_words,
                    zero_pending.candidate.rescue_words,
                ),
                reward_weights=jnp.where(
                    quarantine_opened,
                    state.archive.reward_weights,
                    zero_pending.candidate.reward_weights,
                ),
            ),
            target_birth_words=jnp.where(
                quarantine_opened,
                event.post_birth_words[safe_current_target],
                zero_pending.target_birth_words,
            ),
            source_birth_words=jnp.where(
                quarantine_opened,
                event.source_birth_words,
                zero_pending.source_birth_words,
            ),
            source_reward_weights=jnp.where(
                quarantine_opened,
                event.source_reward_weights,
                zero_pending.source_reward_weights,
            ),
            victim_lineage_words=jnp.where(
                quarantine_opened,
                state.live_lineage_words[safe_current_target],
                zero_pending.victim_lineage_words,
            ),
            victim_rescue_words=jnp.where(
                quarantine_opened,
                state.live_rescue_words[safe_current_target],
                zero_pending.victim_rescue_words,
            ),
            first_never_worse=jnp.where(
                quarantine_opened,
                relation.never_worse[1:],
                zero_pending.first_never_worse,
            ),
            first_ever_strict=jnp.where(
                quarantine_opened,
                relation.ever_strict[1:],
                zero_pending.first_ever_strict,
            ),
        )
        unsigned_candidate = SequentialLineageCacheState(
            config_token=state.config_token,
            content_token=state.content_token,
            bound_birth_words=bound_birth_words,
            live_lineage_words=live_lineage_words,
            live_rescue_words=live_rescue_words,
            archive=selected_archive,
            pending=proposed_pending,
        )
        candidate_state = self._with_content_token(unsigned_candidate)
        candidate_state_valid = self._state_valid_unchecked(
            candidate_state,
            event.post_step_words,
            event.post_birth_words,
            event.post_in_use,
        ) & _tree_finite(candidate_state)
        update_applied = source_state_valid & event_valid & evidence_valid & candidate_state_valid
        committed = cast(
            SequentialLineageCacheState,
            jax.tree_util.tree_map(
                lambda proposed, current: jnp.where(
                    update_applied,
                    proposed,
                    current,
                ),
                candidate_state,
                state,
            ),
        )

        applied_source = jnp.where(
            update_applied,
            selected_source,
            jnp.asarray(ARCHIVE_SOURCE_NONE, dtype=jnp.int32),
        )
        applied_lifecycle = update_applied & (
            quarantine_second | (full_bank_birth & ~quarantine_opened)
        )
        zero_bank = jnp.zeros((c.comparison_bank_size,), dtype=jnp.float32)
        zero_masks = jnp.zeros((c.comparison_bank_size,), dtype=jnp.bool_)
        return SequentialLineageCacheProposal(
            state=committed,
            source_state_valid=source_state_valid,
            event_valid=event_valid,
            predictive_inputs_finite=predictive_inputs_finite,
            evidence_valid=evidence_valid,
            candidate_state_valid=candidate_state_valid,
            rescue_capacity_available=rescue_capacity,
            full_bank_birth=update_applied & full_bank_birth,
            cache_tested=update_applied & cache_tested,
            quarantine_opened=update_applied & quarantine_opened,
            quarantine_second_evidence=update_applied & quarantine_second,
            quarantine_confirmed=update_applied & quarantine_confirmed,
            quarantine_rejected=update_applied & quarantine_rejected,
            target_identity_matched=update_applied & target_identity_matched,
            target_survived=update_applied & target_survived,
            confirmation_commit_abstained=update_applied & confirmation_abstained,
            lineage_transferred=update_applied & confirmation_committable,
            rescue_incremented=update_applied & confirmation_committable,
            victim_staged=update_applied & quarantine_opened,
            overlap_full_bank_birth=update_applied & overlap,
            new_quarantine_suppressed=update_applied & overlap,
            archive_locked_during_pending=update_applied & quarantine_opened,
            archive_selected_source=applied_source,
            archive_old_retained=(applied_lifecycle & (applied_source == ARCHIVE_SOURCE_OLD_CACHE)),
            archive_opening_victim_selected=(
                applied_lifecycle & (applied_source == ARCHIVE_SOURCE_OPENING_VICTIM)
            ),
            archive_current_victim_selected=(
                applied_lifecycle & (applied_source == ARCHIVE_SOURCE_CURRENT_VICTIM)
            ),
            parameter_transplanted=jnp.asarray(False, dtype=jnp.bool_),
            predictions=jnp.where(update_applied, predictions, zero_bank),
            losses=jnp.where(update_applied, losses, zero_bank),
            comparator_mask=jnp.where(
                update_applied,
                relation.comparator_mask,
                zero_masks,
            ),
            never_worse=jnp.where(update_applied, relation.never_worse, zero_masks),
            ever_strict=jnp.where(update_applied, relation.ever_strict, zero_masks),
            update_applied=update_applied,
        )

    def resource_record(
        self,
        *,
        n_agents: int = 1,
        base_scan_carry_nbytes: int = 0,
    ) -> SequentialLineageCacheResourceRecord:
        """Return exact persistent storage and selected-output formulas."""

        if type(n_agents) is not int or n_agents < 1:
            raise ValueError("n_agents must be a positive integer")
        if type(base_scan_carry_nbytes) is not int or base_scan_carry_nbytes < 0:
            raise ValueError("base_scan_carry_nbytes must be a nonnegative integer")
        c = self._config
        config_token = 32
        content_token = _STATE_CONTENT_TOKEN_NBYTES
        archive_record = 25 + 4 * c.n_actions * c.observation_dim
        base = 24 * c.max_contexts + config_token + content_token + archive_record
        pending = (
            27
            + 10 * c.max_contexts
            + 4 * c.max_contexts * c.n_actions * c.observation_dim
            + archive_record
        )
        per_agent = base + pending
        measured = _tree_nbytes(self.init())
        if measured != per_agent:
            raise AssertionError(
                f"state byte formula {per_agent} disagrees with measured {measured}"
            )
        joint = n_agents * per_agent
        bank_size = c.comparison_bank_size
        return SequentialLineageCacheResourceRecord(
            schema=SEQUENTIAL_LINEAGE_CACHE_RESOURCE_SCHEMA,
            config_schema=SEQUENTIAL_LINEAGE_CACHE_CONFIG_SCHEMA,
            state_schema=SEQUENTIAL_LINEAGE_CACHE_STATE_SCHEMA,
            event_schema=SEQUENTIAL_LINEAGE_CACHE_EVENT_SCHEMA,
            proposal_schema=SEQUENTIAL_LINEAGE_CACHE_PROPOSAL_SCHEMA,
            pairwise_observation_schema=PAIRWISE_DOMINANCE_OBSERVATION_SCHEMA,
            pairwise_decision_schema=PAIRWISE_DOMINANCE_DECISION_SCHEMA,
            confirmation_horizon=SEQUENTIAL_LINEAGE_CACHE_CONFIRMATION_HORIZON,
            max_contexts=c.max_contexts,
            n_actions=c.n_actions,
            observation_dim=c.observation_dim,
            comparison_bank_size=bank_size,
            archive_capacity_per_agent=SEQUENTIAL_LINEAGE_CACHE_ARCHIVE_CAPACITY,
            config_token_nbytes_per_agent=config_token,
            content_token_nbytes_per_agent=content_token,
            base_lineage_archive_nbytes_per_agent=base,
            pending_evidence_nbytes_per_agent=pending,
            frozen_source_snapshot_nbytes_per_agent=(
                8 * c.max_contexts + 4 * c.max_contexts * c.n_actions * c.observation_dim
            ),
            frozen_candidate_snapshot_nbytes_per_agent=archive_record,
            per_agent_state_nbytes=per_agent,
            n_agents=n_agents,
            joint_state_nbytes=joint,
            base_scan_carry_nbytes=base_scan_carry_nbytes,
            total_scan_carry_nbytes=base_scan_carry_nbytes + joint,
            logical_prediction_and_error_nbytes=(n_agents * 2 * bank_size * 4),
            logical_selected_pairwise_diagnostic_nbytes=n_agents * 3 * bank_size,
            logical_atomic_candidate_nbytes=joint,
            replay_capacity=0,
            parameter_transplant_allowed=False,
            host_transition_binding_claimed=HOST_TRANSITION_BINDING_CLAIMED,
            state_content_integrity_claimed=STATE_CONTENT_INTEGRITY_CLAIMED,
            external_state_provenance_claimed=EXTERNAL_STATE_PROVENANCE_CLAIMED,
            persistent_capacity_growth=0,
        )

    def work_record(
        self,
        *,
        total_steps: int,
        n_agents: int = 1,
    ) -> SequentialLineageCacheWorkRecord:
        """Return exact named logical counts; outer invocation parity is required."""

        if type(total_steps) is not int or total_steps < 0:
            raise ValueError("total_steps must be a nonnegative integer")
        if type(n_agents) is not int or n_agents < 1:
            raise ValueError("n_agents must be a positive integer")
        c = self._config
        calls = total_steps * n_agents
        predictions = calls * c.comparison_bank_size
        observation_calls = 2 * calls
        resolution_calls = calls
        eligible_cells = observation_calls * (c.max_contexts + 1)
        executed_cells = observation_calls * c.comparison_bank_size
        digest_words = _state_content_word_count(c)
        measured_digest_words = _state_content_words(self.init()).shape[0]
        if measured_digest_words != digest_words:
            raise AssertionError(
                f"content word formula {digest_words} disagrees with measured "
                f"{measured_digest_words}"
            )
        digest_blocks = (digest_words + 3 + 15) // 16
        digest_evaluations = 3 * calls
        return SequentialLineageCacheWorkRecord(
            schema=SEQUENTIAL_LINEAGE_CACHE_WORK_SCHEMA,
            confirmation_horizon=SEQUENTIAL_LINEAGE_CACHE_CONFIRMATION_HORIZON,
            total_steps=total_steps,
            n_agents=n_agents,
            prediction_bank_calls=calls,
            scalar_predictions=predictions,
            absolute_losses=predictions,
            pairwise_observation_calls=observation_calls,
            pairwise_resolution_calls=resolution_calls,
            eligible_candidate_comparator_cells=eligible_cells,
            eligible_relational_comparisons=2 * eligible_cells,
            executed_relational_vector_cells=executed_cells,
            executed_relational_scalar_comparisons=2 * executed_cells,
            coefficient_products=predictions * c.observation_dim,
            dot_additions=predictions * max(c.observation_dim - 1, 0),
            archive_compare_selects=2 * calls,
            state_audits=2 * calls,
            content_digest_evaluations=digest_evaluations,
            content_digest_message_words_per_evaluation=digest_words,
            content_digest_compression_blocks_per_evaluation=digest_blocks,
            content_digest_compression_blocks=digest_evaluations * digest_blocks,
            content_digest_rounds=digest_evaluations * digest_blocks * 64,
            transaction_proposals=calls,
            snapshot_candidate_constructions=calls,
            replay_updates=0,
            random_draws=0,
            reset_callbacks=0,
            maximum_lineage_transfers_per_agent_event=1,
            standalone_schedule_fixed_per_invocation=True,
            outer_invocation_parity_required=True,
            matched_outer_work_claimed=False,
            exhaustive_primitive_operation_count_claimed=False,
            compiled_flop_count_claimed=False,
        )


def initialize_sequential_lineage_cache(
    config: SequentialLineageCacheConfig,
) -> SequentialLineageCacheState:
    """Convenience initializer for one exact empty sidecar."""

    return SequentialLineageCache(config).init()


def measure_sequential_lineage_cache_state_nbytes(
    state: SequentialLineageCacheState,
) -> int:
    """Measure all persistent JAX-array leaves in one sidecar."""

    return _tree_nbytes(state)


__all__ = [
    "ARCHIVE_SOURCE_CURRENT_VICTIM",
    "ARCHIVE_SOURCE_NONE",
    "ARCHIVE_SOURCE_OLD_CACHE",
    "ARCHIVE_SOURCE_OPENING_VICTIM",
    "EXTERNAL_STATE_PROVENANCE_CLAIMED",
    "HOST_TRANSITION_BINDING_CLAIMED",
    "SEQUENTIAL_LINEAGE_CACHE_ARCHIVE_CAPACITY",
    "SEQUENTIAL_LINEAGE_CACHE_CONFIG_SCHEMA",
    "SEQUENTIAL_LINEAGE_CACHE_CONFIRMATION_HORIZON",
    "SEQUENTIAL_LINEAGE_CACHE_EVENT_SCHEMA",
    "SEQUENTIAL_LINEAGE_CACHE_PROPOSAL_SCHEMA",
    "SEQUENTIAL_LINEAGE_CACHE_RESOURCE_SCHEMA",
    "SEQUENTIAL_LINEAGE_CACHE_STATE_SCHEMA",
    "SEQUENTIAL_LINEAGE_CACHE_WORK_SCHEMA",
    "STATE_CONTENT_INTEGRITY_CLAIMED",
    "SequentialLineageArchiveRecord",
    "SequentialLineageCache",
    "SequentialLineageCacheConfig",
    "SequentialLineageCacheEvent",
    "SequentialLineageCacheProposal",
    "SequentialLineageCacheResourceRecord",
    "SequentialLineageCacheState",
    "SequentialLineageCacheWorkRecord",
    "SequentialLineagePendingEvidence",
    "initialize_sequential_lineage_cache",
    "measure_sequential_lineage_cache_state_nbytes",
]
