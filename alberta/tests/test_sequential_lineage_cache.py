# mypy: disable-error-code="call-arg"
"""Contracts for the fixed H=2 sequential cross-birth lineage cache."""

from __future__ import annotations

import hashlib
import struct

import chex
import jax.numpy as jnp
import pytest

from alberta_framework.core import sequential_lineage_cache as lineage_cache_module
from alberta_framework.core.sequential_lineage_cache import (
    ARCHIVE_SOURCE_CURRENT_VICTIM,
    ARCHIVE_SOURCE_OLD_CACHE,
    ARCHIVE_SOURCE_OPENING_VICTIM,
    EXTERNAL_STATE_PROVENANCE_CLAIMED,
    HOST_TRANSITION_BINDING_CLAIMED,
    SEQUENTIAL_LINEAGE_CACHE_CONFIRMATION_HORIZON,
    STATE_CONTENT_INTEGRITY_CLAIMED,
    SequentialLineageArchiveRecord,
    SequentialLineageCache,
    SequentialLineageCacheConfig,
    SequentialLineageCacheEvent,
    SequentialLineageCacheProposal,
    SequentialLineageCacheState,
    measure_sequential_lineage_cache_state_nbytes,
)

pytestmark = pytest.mark.unit

CONFIG = SequentialLineageCacheConfig(
    max_contexts=3,
    n_actions=4,
    observation_dim=4,
    initial_reward_estimate=0.5,
)


def _words(value: int) -> jnp.ndarray:
    return jnp.asarray((0, value), dtype=jnp.uint32)


def _cache_weights(*, first: float = 0.5, second: float = 0.0) -> jnp.ndarray:
    weights = jnp.zeros((CONFIG.n_actions, CONFIG.observation_dim), dtype=jnp.float32)
    return weights.at[0, 0].set(first).at[0, 1].set(second)


def _source_weights(
    *,
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> jnp.ndarray:
    weights = jnp.zeros(
        (CONFIG.max_contexts, CONFIG.n_actions, CONFIG.observation_dim),
        dtype=jnp.float32,
    )
    weights = weights.at[:, 0, 0].set(jnp.asarray(first, dtype=jnp.float32))
    return weights.at[:, 0, 1].set(jnp.asarray(second, dtype=jnp.float32))


def _full_source(
    *,
    cache_weights: jnp.ndarray | None = None,
    source_weights: jnp.ndarray | None = None,
    cache_rescue_words: jnp.ndarray | None = None,
) -> tuple[
    SequentialLineageCache,
    SequentialLineageCacheState,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    mechanism = SequentialLineageCache(CONFIG)
    births = jnp.asarray(((0, 1), (0, 2), (0, 3)), dtype=jnp.uint32)
    in_use = jnp.ones((CONFIG.max_contexts,), dtype=jnp.bool_)
    rewards = (
        _source_weights(first=(0.5, 0.5, 0.5), second=(1.0, 1.0, 1.0))
        if source_weights is None
        else source_weights
    )
    archive = SequentialLineageArchiveRecord(
        valid=jnp.asarray(True, dtype=jnp.bool_),
        source_birth_words=_words(4),
        lineage_words=_words(4),
        rescue_words=(
            jnp.zeros((2,), dtype=jnp.uint32) if cache_rescue_words is None else cache_rescue_words
        ),
        reward_weights=(_cache_weights() if cache_weights is None else cache_weights),
    )
    initial = mechanism.init().replace(  # type: ignore[attr-defined]
        bound_birth_words=births,
        live_lineage_words=births,
        archive=archive,
    )
    # Synthetic trusted fixture only.  Resealing proves integrity, not provenance.
    initial = mechanism._with_content_token(initial)  # noqa: SLF001
    assert bool(mechanism.state_valid(initial, _words(10), births, in_use))
    return mechanism, initial, births, in_use, rewards


def _opening_event(
    births: jnp.ndarray,
    in_use: jnp.ndarray,
    source_weights: jnp.ndarray,
    *,
    observation_index: int = 0,
    reward: float = 0.0,
) -> SequentialLineageCacheEvent:
    post_births = births.at[2].set(_words(11))
    return SequentialLineageCacheEvent(
        source_step_words=_words(10),
        post_step_words=_words(11),
        source_birth_words=births,
        post_birth_words=post_births,
        source_in_use=in_use,
        post_in_use=in_use,
        source_reward_weights=source_weights,
        observation=jax_one_hot(observation_index),
        action=jnp.asarray(0, dtype=jnp.int32),
        reward=jnp.asarray(reward, dtype=jnp.float32),
        allocated=jnp.asarray(True, dtype=jnp.bool_),
        evicted=jnp.asarray(True, dtype=jnp.bool_),
        target_slot=jnp.asarray(2, dtype=jnp.int32),
        context_update_applied=jnp.asarray(True, dtype=jnp.bool_),
    )


def _second_event(
    opening: SequentialLineageCacheProposal,
    source_weights: jnp.ndarray,
    *,
    observation_index: int = 1,
    reward: float = 0.0,
    allocation_target: int | None = None,
) -> SequentialLineageCacheEvent:
    state = opening.state
    births = state.bound_birth_words
    in_use = jnp.ones((CONFIG.max_contexts,), dtype=jnp.bool_)
    allocated = allocation_target is not None
    post_births = (
        births.at[allocation_target].set(_words(12)) if allocation_target is not None else births
    )
    return SequentialLineageCacheEvent(
        source_step_words=_words(11),
        post_step_words=_words(12),
        source_birth_words=births,
        post_birth_words=post_births,
        source_in_use=in_use,
        post_in_use=in_use,
        source_reward_weights=source_weights,
        observation=jax_one_hot(observation_index),
        action=jnp.asarray(0, dtype=jnp.int32),
        reward=jnp.asarray(reward, dtype=jnp.float32),
        allocated=jnp.asarray(allocated, dtype=jnp.bool_),
        evicted=jnp.asarray(allocated, dtype=jnp.bool_),
        target_slot=jnp.asarray(
            0 if allocation_target is None else allocation_target,
            dtype=jnp.int32,
        ),
        context_update_applied=jnp.asarray(True, dtype=jnp.bool_),
    )


def jax_one_hot(index: int) -> jnp.ndarray:
    return jnp.eye(CONFIG.observation_dim, dtype=jnp.float32)[index]


def test_resource_and_work_contracts_are_exact_and_bounded() -> None:
    mechanism = SequentialLineageCache(CONFIG)
    state = mechanism.init()
    resources = mechanism.resource_record(n_agents=2, base_scan_carry_nbytes=978)
    work = mechanism.work_record(total_steps=4_000, n_agents=2)

    assert SEQUENTIAL_LINEAGE_CACHE_CONFIRMATION_HORIZON == 2
    assert measure_sequential_lineage_cache_state_nbytes(state) == 563
    assert resources.config_token_nbytes_per_agent == 32
    assert resources.content_token_nbytes_per_agent == 32
    assert resources.base_lineage_archive_nbytes_per_agent == 225
    assert resources.pending_evidence_nbytes_per_agent == 338
    assert resources.frozen_source_snapshot_nbytes_per_agent == 216
    assert resources.frozen_candidate_snapshot_nbytes_per_agent == 89
    assert resources.per_agent_state_nbytes == 563
    assert resources.joint_state_nbytes == 1_126
    assert resources.total_scan_carry_nbytes == 2_104
    assert resources.logical_prediction_and_error_nbytes == 80
    assert resources.logical_selected_pairwise_diagnostic_nbytes == 30
    assert resources.logical_atomic_candidate_nbytes == 1_126
    assert resources.archive_capacity_per_agent == 1
    assert resources.replay_capacity == 0
    assert resources.parameter_transplant_allowed is False
    assert resources.host_transition_binding_claimed is False
    assert resources.state_content_integrity_claimed is True
    assert resources.external_state_provenance_claimed is False

    assert work.prediction_bank_calls == 8_000
    assert work.scalar_predictions == work.absolute_losses == 40_000
    assert work.pairwise_observation_calls == 16_000
    assert work.pairwise_resolution_calls == 8_000
    assert work.eligible_candidate_comparator_cells == 64_000
    assert work.eligible_relational_comparisons == 128_000
    assert work.executed_relational_vector_cells == 80_000
    assert work.executed_relational_scalar_comparisons == 160_000
    assert work.coefficient_products == 160_000
    assert work.dot_additions == 120_000
    assert work.archive_compare_selects == 16_000
    assert work.state_audits == 16_000
    assert work.content_digest_evaluations == 24_000
    assert work.content_digest_message_words_per_evaluation == 141
    assert work.content_digest_compression_blocks_per_evaluation == 9
    assert work.content_digest_compression_blocks == 216_000
    assert work.content_digest_rounds == 13_824_000
    assert work.replay_updates == work.random_draws == work.reset_callbacks == 0
    assert work.standalone_schedule_fixed_per_invocation is True
    assert work.outer_invocation_parity_required is True
    assert work.matched_outer_work_claimed is False
    assert work.exhaustive_primitive_operation_count_claimed is False
    assert work.compiled_flop_count_claimed is False
    assert HOST_TRANSITION_BINDING_CLAIMED is False
    assert STATE_CONTENT_INTEGRITY_CLAIMED is True
    assert EXTERNAL_STATE_PROVENANCE_CLAIMED is False


@pytest.mark.parametrize(
    "raw_words",
    [
        (),
        (0,),
        tuple(range(20)),
    ],
)
def test_jax_sha256_word_framing_matches_hashlib(raw_words: tuple[int, ...]) -> None:
    words = jnp.asarray(raw_words, dtype=jnp.uint32)
    expected = hashlib.sha256(
        b"".join(struct.pack(">I", value) for value in raw_words)
    ).digest()

    observed = lineage_cache_module._sha256_word_message(words)  # noqa: SLF001

    assert bytes(int(value) for value in observed.tolist()) == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_contexts", True),
        ("max_contexts", 3.0),
        ("n_actions", 4.0),
        ("observation_dim", 4.0),
    ],
)
def test_config_rejects_nonordinary_integer_geometry(field: str, value: object) -> None:
    kwargs: dict[str, object] = {
        "max_contexts": 3,
        "n_actions": 4,
        "observation_dim": 4,
        "initial_reward_estimate": 0.5,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match="positive integer"):
        SequentialLineageCacheConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), 1e300])
def test_config_rejects_nonfinite_or_unrepresentable_prior(value: object) -> None:
    with pytest.raises(ValueError, match="float32-representable"):
        SequentialLineageCacheConfig(
            max_contexts=3,
            n_actions=4,
            observation_dim=4,
            initial_reward_estimate=value,  # type: ignore[arg-type]
        )


def test_resource_and_work_arguments_are_strict_ordinary_integers() -> None:
    mechanism = SequentialLineageCache(CONFIG)

    for kwargs in ({"n_agents": True}, {"n_agents": 2.0}, {"base_scan_carry_nbytes": 1.5}):
        with pytest.raises(ValueError):
            mechanism.resource_record(**kwargs)
    for kwargs in (
        {"total_steps": True},
        {"total_steps": 1.5},
        {"total_steps": 1, "n_agents": 2.0},
    ):
        with pytest.raises(ValueError):
            mechanism.work_record(**kwargs)


def test_invalid_pending_payload_is_exactly_zero() -> None:
    mechanism = SequentialLineageCache(CONFIG)
    state = mechanism.init()
    births = jnp.zeros((CONFIG.max_contexts, 2), dtype=jnp.uint32)
    in_use = jnp.asarray((True, False, False), dtype=jnp.bool_)

    assert bool(mechanism.state_valid(state, _words(0), births, in_use))
    assert not bool(state.pending.valid)
    assert not bool(state.pending.candidate.valid)
    for leaf in (
        state.pending.candidate.source_birth_words,
        state.pending.candidate.lineage_words,
        state.pending.candidate.rescue_words,
        state.pending.candidate.reward_weights,
        state.pending.target_birth_words,
        state.pending.source_birth_words,
        state.pending.source_reward_weights,
        state.pending.victim_lineage_words,
        state.pending.victim_rescue_words,
        state.pending.first_never_worse,
        state.pending.first_ever_strict,
    ):
        assert not bool(jnp.any(leaf))

    malformed = state.replace(  # type: ignore[attr-defined]
        pending=state.pending.replace(  # type: ignore[attr-defined]
            source_reward_weights=state.pending.source_reward_weights.at[0, 0, 0].set(1.0)
        )
    )
    malformed = mechanism._with_content_token(malformed)  # noqa: SLF001
    assert not bool(mechanism.state_valid(malformed, _words(0), births, in_use))


def test_live_births_and_archive_source_birth_are_unique_and_disjoint() -> None:
    mechanism, state, births, in_use, _ = _full_source()
    duplicate_births = births.at[1].set(births[2])
    duplicate_state = state.replace(  # type: ignore[attr-defined]
        bound_birth_words=duplicate_births
    )
    colliding_archive = state.archive.replace(  # type: ignore[attr-defined]
        source_birth_words=births[2],
        lineage_words=_words(0),
    )
    collision_state = state.replace(archive=colliding_archive)  # type: ignore[attr-defined]
    duplicate_state = mechanism._with_content_token(duplicate_state)  # noqa: SLF001
    collision_state = mechanism._with_content_token(collision_state)  # noqa: SLF001

    assert not bool(
        mechanism.state_valid(
            duplicate_state,
            _words(10),
            duplicate_births,
            in_use,
        )
    )
    assert not bool(mechanism.state_valid(collision_state, _words(10), births, in_use))


def test_live_archive_and_pending_rescue_counts_are_birth_bounded() -> None:
    mechanism, state, births, in_use, source_weights = _full_source()
    live_overflow = state.replace(  # type: ignore[attr-defined]
        live_rescue_words=state.live_rescue_words.at[2].set(_words(4))
    )
    archive_overflow = state.replace(  # type: ignore[attr-defined]
        archive=state.archive.replace(rescue_words=_words(5))  # type: ignore[attr-defined]
    )
    opening = mechanism.propose(state, _opening_event(births, in_use, source_weights))
    pending_overflow = opening.state.replace(  # type: ignore[attr-defined]
        pending=opening.state.pending.replace(  # type: ignore[attr-defined]
            victim_rescue_words=_words(4)
        )
    )
    live_overflow = mechanism._with_content_token(live_overflow)  # noqa: SLF001
    archive_overflow = mechanism._with_content_token(archive_overflow)  # noqa: SLF001
    pending_overflow = mechanism._with_content_token(pending_overflow)  # noqa: SLF001

    assert not bool(mechanism.state_valid(live_overflow, _words(10), births, in_use))
    assert not bool(mechanism.state_valid(archive_overflow, _words(10), births, in_use))
    assert not bool(
        mechanism.state_valid(
            pending_overflow,
            _words(11),
            opening.state.bound_birth_words,
            in_use,
        )
    )


def test_pending_target_and_candidate_reachability_tampering_is_rejected() -> None:
    mechanism, state, births, in_use, source_weights = _full_source()
    opening = mechanism.propose(state, _opening_event(births, in_use, source_weights))
    target = 2
    target_lineage_changed = opening.state.replace(  # type: ignore[attr-defined]
        live_lineage_words=opening.state.live_lineage_words.at[target].set(_words(5))
    )
    target_rescue_changed = opening.state.replace(  # type: ignore[attr-defined]
        live_rescue_words=opening.state.live_rescue_words.at[target].set(_words(1))
    )
    victim_lineage_candidate = opening.state.archive.replace(  # type: ignore[attr-defined]
        lineage_words=opening.state.pending.victim_lineage_words
    )
    candidate_lineage_changed = opening.state.replace(  # type: ignore[attr-defined]
        archive=victim_lineage_candidate,
        pending=opening.state.pending.replace(  # type: ignore[attr-defined]
            candidate=victim_lineage_candidate
        ),
    )
    frozen_birth_candidate = opening.state.archive.replace(  # type: ignore[attr-defined]
        source_birth_words=births[2],
        lineage_words=_words(0),
    )
    candidate_source_changed = opening.state.replace(  # type: ignore[attr-defined]
        archive=frozen_birth_candidate,
        pending=opening.state.pending.replace(  # type: ignore[attr-defined]
            candidate=frozen_birth_candidate
        ),
    )

    for malformed in (
        target_lineage_changed,
        target_rescue_changed,
        candidate_lineage_changed,
        candidate_source_changed,
    ):
        malformed = mechanism._with_content_token(malformed)  # noqa: SLF001
        assert not bool(
            mechanism.state_valid(
                malformed,
                _words(11),
                opening.state.bound_birth_words,
                in_use,
            )
        )


def test_all_tie_opening_stages_exact_snapshot_with_zero_rescue_commit() -> None:
    mechanism, state, births, in_use, source_weights = _full_source()
    result = mechanism.propose(state, _opening_event(births, in_use, source_weights))

    assert bool(result.update_applied)
    assert bool(result.cache_tested)
    assert bool(result.quarantine_opened)
    assert bool(result.victim_staged)
    assert bool(result.archive_locked_during_pending)
    assert not bool(result.lineage_transferred)
    assert not bool(result.rescue_incremented)
    assert not bool(result.parameter_transplanted)
    chex.assert_trees_all_equal(result.losses, jnp.full((5,), 0.5, dtype=jnp.float32))
    chex.assert_trees_all_equal(result.state.archive, state.archive)
    chex.assert_trees_all_equal(result.state.pending.candidate, state.archive)
    chex.assert_trees_all_equal(result.state.pending.target_birth_words, _words(11))
    chex.assert_trees_all_equal(result.state.pending.source_birth_words, births)
    chex.assert_trees_all_equal(
        result.state.pending.source_reward_weights,
        source_weights,
    )
    chex.assert_trees_all_equal(
        result.state.pending.victim_lineage_words,
        state.live_lineage_words[2],
    )
    chex.assert_trees_all_equal(
        result.state.live_lineage_words[2],
        _words(11),
    )
    assert not bool(jnp.any(result.state.live_rescue_words[2]))


@pytest.mark.parametrize(
    "field",
    [
        "valid",
        "candidate.valid",
        "candidate.source_birth_words",
        "candidate.lineage_words",
        "candidate.rescue_words",
        "candidate.reward_weights",
        "target_birth_words",
        "source_birth_words",
        "source_reward_weights",
        "victim_lineage_words",
        "victim_rescue_words",
        "first_never_worse",
        "first_ever_strict",
    ],
)
def test_every_pending_field_rejects_a_stale_content_token(field: str) -> None:
    mechanism, state, births, in_use, source_weights = _full_source()
    opening = mechanism.propose(state, _opening_event(births, in_use, source_weights))
    pending = opening.state.pending
    candidate_mutations = {
        "candidate.valid": pending.candidate.replace(  # type: ignore[attr-defined]
            valid=jnp.asarray(False, dtype=jnp.bool_)
        ),
        "candidate.source_birth_words": pending.candidate.replace(  # type: ignore[attr-defined]
            source_birth_words=_words(5)
        ),
        "candidate.lineage_words": pending.candidate.replace(  # type: ignore[attr-defined]
            lineage_words=_words(0)
        ),
        "candidate.rescue_words": pending.candidate.replace(  # type: ignore[attr-defined]
            rescue_words=_words(1)
        ),
        "candidate.reward_weights": pending.candidate.replace(  # type: ignore[attr-defined]
            reward_weights=pending.candidate.reward_weights.at[0, 0].set(0.25)
        ),
    }
    mutations = {
        "valid": pending.replace(  # type: ignore[attr-defined]
            valid=jnp.asarray(False, dtype=jnp.bool_)
        ),
        **{
            name: pending.replace(candidate=candidate)  # type: ignore[attr-defined]
            for name, candidate in candidate_mutations.items()
        },
        "target_birth_words": pending.replace(  # type: ignore[attr-defined]
            target_birth_words=_words(10)
        ),
        "source_birth_words": pending.replace(  # type: ignore[attr-defined]
            source_birth_words=pending.source_birth_words.at[0].set(_words(5))
        ),
        "source_reward_weights": pending.replace(  # type: ignore[attr-defined]
            source_reward_weights=pending.source_reward_weights.at[0, 0, 0].set(0.25)
        ),
        "victim_lineage_words": pending.replace(  # type: ignore[attr-defined]
            victim_lineage_words=_words(0)
        ),
        "victim_rescue_words": pending.replace(  # type: ignore[attr-defined]
            victim_rescue_words=_words(1)
        ),
        "first_never_worse": pending.replace(  # type: ignore[attr-defined]
            first_never_worse=pending.first_never_worse.at[0].set(False)
        ),
        "first_ever_strict": pending.replace(  # type: ignore[attr-defined]
            first_ever_strict=pending.first_ever_strict.at[0].set(True)
        ),
    }
    tampered = opening.state.replace(pending=mutations[field])  # type: ignore[attr-defined]

    chex.assert_trees_all_equal(tampered.content_token, opening.state.content_token)
    assert not bool(
        mechanism.state_valid(
            tampered,
            _words(11),
            opening.state.bound_birth_words,
            in_use,
        )
    )
    rejected = mechanism.propose(
        tampered,
        _second_event(opening, source_weights),
    )
    assert not bool(rejected.source_state_valid)
    assert not bool(rejected.update_applied)
    chex.assert_trees_all_equal(rejected.state, tampered)


def test_stale_token_rejects_bounded_invented_live_and_archive_content() -> None:
    mechanism, state, births, in_use, _ = _full_source()
    invented_births = births.at[2].set(_words(5))
    cases = (
        (
            state.replace(bound_birth_words=invented_births),  # type: ignore[attr-defined]
            invented_births,
        ),
        (
            state.replace(  # type: ignore[attr-defined]
                live_lineage_words=state.live_lineage_words.at[2].set(_words(0))
            ),
            births,
        ),
        (
            state.replace(  # type: ignore[attr-defined]
                live_rescue_words=state.live_rescue_words.at[2].set(_words(1))
            ),
            births,
        ),
        (
            state.replace(  # type: ignore[attr-defined]
                archive=state.archive.replace(lineage_words=_words(0))  # type: ignore[attr-defined]
            ),
            births,
        ),
        (
            state.replace(  # type: ignore[attr-defined]
                archive=state.archive.replace(rescue_words=_words(1))  # type: ignore[attr-defined]
            ),
            births,
        ),
        (
            state.replace(  # type: ignore[attr-defined]
                archive=state.archive.replace(source_birth_words=_words(5))  # type: ignore[attr-defined]
            ),
            births,
        ),
        (
            state.replace(  # type: ignore[attr-defined]
                archive=state.archive.replace(  # type: ignore[attr-defined]
                    reward_weights=state.archive.reward_weights.at[0, 0].set(0.25)
                )
            ),
            births,
        ),
    )

    for tampered, external_births in cases:
        assert not bool(mechanism.state_valid(tampered, _words(10), external_births, in_use))


def test_private_reseal_demonstrates_integrity_without_external_provenance() -> None:
    mechanism, state, births, in_use, _ = _full_source()
    invented = state.replace(  # type: ignore[attr-defined]
        live_rescue_words=state.live_rescue_words.at[2].set(_words(1))
    )

    assert not bool(mechanism.state_valid(invented, _words(10), births, in_use))
    resealed = mechanism._with_content_token(invented)  # noqa: SLF001

    assert bool(mechanism.state_valid(resealed, _words(10), births, in_use))
    assert STATE_CONTENT_INTEGRITY_CLAIMED is True
    assert EXTERNAL_STATE_PROVENANCE_CLAIMED is False


def test_tie_then_strict_confirmation_transfers_only_lineage_and_rescue() -> None:
    mechanism, state, births, in_use, source_weights = _full_source()
    opening = mechanism.propose(state, _opening_event(births, in_use, source_weights))
    mutated_live = source_weights.at[:, 0, 1].set(jnp.float32(-100.0))
    second = mechanism.propose(
        opening.state,
        _second_event(opening, mutated_live),
    )

    assert bool(second.update_applied)
    assert bool(second.quarantine_second_evidence)
    assert bool(second.quarantine_confirmed)
    assert not bool(second.quarantine_rejected)
    assert bool(second.target_survived)
    assert bool(second.lineage_transferred)
    assert bool(second.rescue_incremented)
    assert not bool(second.confirmation_commit_abstained)
    assert not bool(second.parameter_transplanted)
    chex.assert_trees_all_equal(
        second.losses,
        jnp.asarray((0.0, 0.5, 1.0, 1.0, 1.0), dtype=jnp.float32),
    )
    chex.assert_trees_all_equal(
        second.state.live_lineage_words[2],
        state.archive.lineage_words,
    )
    chex.assert_trees_all_equal(second.state.live_rescue_words[2], _words(1))
    assert int(second.archive_selected_source) == ARCHIVE_SOURCE_OPENING_VICTIM
    chex.assert_trees_all_equal(
        second.state.archive.source_birth_words,
        births[2],
    )
    chex.assert_trees_all_equal(
        second.state.archive.reward_weights,
        source_weights[2],
    )
    assert not bool(second.state.pending.valid)


def test_pending_state_rejects_a_different_fresh_prior_config() -> None:
    mechanism, state, births, in_use, source_weights = _full_source()
    opening = mechanism.propose(state, _opening_event(births, in_use, source_weights))
    other = SequentialLineageCache(
        SequentialLineageCacheConfig(
            max_contexts=CONFIG.max_contexts,
            n_actions=CONFIG.n_actions,
            observation_dim=CONFIG.observation_dim,
            initial_reward_estimate=0.25,
        )
    )

    assert not bool(
        other.state_valid(
            opening.state,
            _words(11),
            opening.state.bound_birth_words,
            in_use,
        )
    )
    assert not bool(jnp.array_equal(mechanism.init().config_token, other.init().config_token))
    rejected = other.propose(
        opening.state,
        _second_event(opening, source_weights),
    )

    assert not bool(rejected.source_state_valid)
    assert not bool(rejected.update_applied)
    chex.assert_trees_all_equal(rejected.state, opening.state)


def test_pending_state_rejects_archive_candidate_swap() -> None:
    mechanism, state, births, in_use, source_weights = _full_source()
    opening = mechanism.propose(state, _opening_event(births, in_use, source_weights))
    swapped_archive = opening.state.archive.replace(  # type: ignore[attr-defined]
        reward_weights=opening.state.archive.reward_weights.at[0, 0].set(-0.25)
    )
    swapped = opening.state.replace(archive=swapped_archive)  # type: ignore[attr-defined]
    swapped = mechanism._with_content_token(swapped)  # noqa: SLF001

    assert not bool(
        mechanism.state_valid(
            swapped,
            _words(11),
            opening.state.bound_birth_words,
            in_use,
        )
    )
    rejected = mechanism.propose(
        swapped,
        _second_event(opening, source_weights),
    )

    assert not bool(rejected.source_state_valid)
    assert not bool(rejected.update_applied)
    chex.assert_trees_all_equal(rejected.state, swapped)


def test_fresh_or_live_comparator_tied_on_both_events_forces_valid_rejection() -> None:
    mechanism, state, births, in_use, source_weights = _full_source()
    opening = mechanism.propose(state, _opening_event(births, in_use, source_weights))
    second = mechanism.propose(
        opening.state,
        _second_event(opening, source_weights, observation_index=0),
    )

    assert bool(second.evidence_valid)
    assert not bool(second.quarantine_confirmed)
    assert bool(second.quarantine_rejected)
    assert not bool(second.lineage_transferred)
    assert int(second.archive_selected_source) == ARCHIVE_SOURCE_OLD_CACHE


@pytest.mark.parametrize(
    ("cache_second", "live_first", "live_second", "confirmed"),
    [
        (0.5, (1.0, 1.0, 1.0), (0.5, 0.5, 0.5), True),
        (0.5, (1.0, 1.0, 1.0), (0.0, 1.0, 1.0), False),
        (0.0, (1.0, 1.0, 0.0), (1.0, 1.0, 0.0), False),
    ],
)
def test_fixed_h2_law_covers_strict_then_tie_worse_and_double_tie(
    cache_second: float,
    live_first: tuple[float, float, float],
    live_second: tuple[float, float, float],
    confirmed: bool,
) -> None:
    cache = _cache_weights(first=0.0, second=cache_second)
    sources = _source_weights(first=live_first, second=live_second)
    mechanism, state, births, in_use, _ = _full_source(
        cache_weights=cache,
        source_weights=sources,
    )
    opening = mechanism.propose(state, _opening_event(births, in_use, sources))
    second = mechanism.propose(opening.state, _second_event(opening, sources))

    assert bool(second.quarantine_confirmed) is confirmed
    assert bool(second.quarantine_rejected) is (not confirmed)
    if confirmed:
        assert bool(second.lineage_transferred)
    else:
        assert not bool(second.lineage_transferred)
        assert int(second.archive_selected_source) == ARCHIVE_SOURCE_OLD_CACHE


def test_exact_timing_and_birth_consistency_fail_closed_atomically() -> None:
    mechanism, state, births, in_use, source_weights = _full_source()
    opening = mechanism.propose(state, _opening_event(births, in_use, source_weights))
    valid_second = _second_event(opening, source_weights)
    stale_clock = valid_second.replace(  # type: ignore[attr-defined]
        source_step_words=_words(12),
        post_step_words=_words(13),
    )
    stale_birth = valid_second.replace(  # type: ignore[attr-defined]
        source_birth_words=valid_second.source_birth_words.at[0].set(_words(99))
    )

    for event in (stale_clock, stale_birth):
        rejected = mechanism.propose(opening.state, event)
        assert not bool(rejected.update_applied)
        assert not bool(rejected.source_state_valid)
        chex.assert_trees_all_equal(rejected.state, opening.state)


def test_non_full_allocation_initializes_a_new_lineage_without_archiving() -> None:
    mechanism = SequentialLineageCache(CONFIG)
    state = mechanism.init()
    births = jnp.zeros((CONFIG.max_contexts, 2), dtype=jnp.uint32)
    source_in_use = jnp.asarray((True, False, False), dtype=jnp.bool_)
    post_in_use = jnp.asarray((True, True, False), dtype=jnp.bool_)
    post_births = births.at[1].set(_words(1))
    source_weights = jnp.zeros(
        (CONFIG.max_contexts, CONFIG.n_actions, CONFIG.observation_dim),
        dtype=jnp.float32,
    )
    event = SequentialLineageCacheEvent(
        source_step_words=_words(0),
        post_step_words=_words(1),
        source_birth_words=births,
        post_birth_words=post_births,
        source_in_use=source_in_use,
        post_in_use=post_in_use,
        source_reward_weights=source_weights,
        observation=jax_one_hot(0),
        action=jnp.asarray(0, dtype=jnp.int32),
        reward=jnp.asarray(0.0, dtype=jnp.float32),
        allocated=jnp.asarray(True, dtype=jnp.bool_),
        evicted=jnp.asarray(False, dtype=jnp.bool_),
        target_slot=jnp.asarray(1, dtype=jnp.int32),
        context_update_applied=jnp.asarray(True, dtype=jnp.bool_),
    )

    assert bool(mechanism.state_valid(state, _words(0), births, source_in_use))
    result = mechanism.propose(state, event)

    assert bool(result.update_applied)
    assert not bool(result.full_bank_birth)
    assert not bool(result.cache_tested)
    assert not bool(result.state.archive.valid)
    assert not bool(result.state.pending.valid)
    chex.assert_trees_all_equal(result.state.bound_birth_words[1], _words(1))
    chex.assert_trees_all_equal(result.state.live_lineage_words[1], _words(1))
    chex.assert_trees_all_equal(result.state.live_rescue_words[1], _words(0))


def test_nonfinite_or_rejected_child_transaction_rolls_back_every_field() -> None:
    mechanism, state, births, in_use, source_weights = _full_source()
    event = _opening_event(births, in_use, source_weights)
    nonfinite = event.replace(  # type: ignore[attr-defined]
        reward=jnp.asarray(jnp.nan, dtype=jnp.float32)
    )
    rejected_child = event.replace(  # type: ignore[attr-defined]
        context_update_applied=jnp.asarray(False, dtype=jnp.bool_)
    )

    for invalid in (nonfinite, rejected_child):
        result = mechanism.propose(state, invalid)
        assert not bool(result.update_applied)
        chex.assert_trees_all_equal(result.state, state)


def test_target_eviction_records_confirmation_but_abstains_and_keeps_top_archive() -> None:
    mechanism, state, births, in_use, source_weights = _full_source()
    opening = mechanism.propose(state, _opening_event(births, in_use, source_weights))
    second = mechanism.propose(
        opening.state,
        _second_event(opening, source_weights, allocation_target=2),
    )

    assert bool(second.update_applied)
    assert bool(second.overlap_full_bank_birth)
    assert bool(second.new_quarantine_suppressed)
    assert bool(second.quarantine_confirmed)
    assert not bool(second.target_survived)
    assert bool(second.confirmation_commit_abstained)
    assert not bool(second.lineage_transferred)
    assert not bool(second.rescue_incremented)
    assert int(second.archive_selected_source) == ARCHIVE_SOURCE_CURRENT_VICTIM
    chex.assert_trees_all_equal(second.state.archive.source_birth_words, _words(11))
    chex.assert_trees_all_equal(second.state.archive.lineage_words, _words(11))


def test_overlap_on_another_slot_commits_then_reduces_both_victims_to_top_one() -> None:
    mechanism, state, births, in_use, source_weights = _full_source()
    opening = mechanism.propose(state, _opening_event(births, in_use, source_weights))
    second = mechanism.propose(
        opening.state,
        _second_event(opening, source_weights, allocation_target=0),
    )

    assert bool(second.quarantine_confirmed)
    assert bool(second.target_survived)
    assert bool(second.lineage_transferred)
    assert bool(second.overlap_full_bank_birth)
    assert bool(second.new_quarantine_suppressed)
    assert int(second.archive_selected_source) == ARCHIVE_SOURCE_OPENING_VICTIM
    chex.assert_trees_all_equal(second.state.archive.source_birth_words, _words(3))
    chex.assert_trees_all_equal(second.state.live_lineage_words[2], _words(4))


def test_unreachable_rescue_counter_fails_closed_without_wrap() -> None:
    maximum = jnp.asarray((2**32 - 1, 2**32 - 1), dtype=jnp.uint32)
    mechanism, state, births, in_use, source_weights = _full_source()
    malformed = state.replace(  # type: ignore[attr-defined]
        archive=state.archive.replace(rescue_words=maximum)  # type: ignore[attr-defined]
    )
    malformed = mechanism._with_content_token(malformed)  # noqa: SLF001

    assert not bool(mechanism.state_valid(malformed, _words(10), births, in_use))
    result = mechanism.propose(
        malformed,
        _opening_event(births, in_use, source_weights),
    )

    assert not bool(result.source_state_valid)
    assert not bool(result.rescue_capacity_available)
    assert not bool(result.update_applied)
    assert not bool(result.lineage_transferred)
    assert not bool(result.rescue_incremented)
    chex.assert_trees_all_equal(result.state, malformed)


def test_clock_and_rescue_increment_cross_the_low_word_boundary_exactly() -> None:
    low_max = jnp.asarray((0, 2**32 - 1), dtype=jnp.uint32)
    high_one = jnp.asarray((1, 0), dtype=jnp.uint32)
    high_one_next = jnp.asarray((1, 1), dtype=jnp.uint32)
    mechanism, state, births, in_use, source_weights = _full_source()
    boundary_archive = state.archive.replace(  # type: ignore[attr-defined]
        source_birth_words=low_max,
        rescue_words=low_max,
    )
    state = mechanism._with_content_token(  # noqa: SLF001
        state.replace(archive=boundary_archive)  # type: ignore[attr-defined]
    )
    opening_event = _opening_event(births, in_use, source_weights).replace(  # type: ignore[attr-defined]
        source_step_words=low_max,
        post_step_words=high_one,
        post_birth_words=births.at[2].set(high_one),
    )

    assert bool(mechanism.state_valid(state, low_max, births, in_use))
    opening = mechanism.propose(state, opening_event)
    second_event = _second_event(opening, source_weights).replace(  # type: ignore[attr-defined]
        source_step_words=high_one,
        post_step_words=high_one_next,
    )
    second = mechanism.propose(opening.state, second_event)

    assert bool(opening.update_applied)
    assert bool(second.update_applied)
    assert bool(second.quarantine_confirmed)
    assert bool(second.lineage_transferred)
    chex.assert_trees_all_equal(second.state.live_rescue_words[2], high_one)
    chex.assert_trees_all_equal(second.state.bound_birth_words[2], high_one)


def test_rejected_candidate_retains_higher_rescue_victim_before_recency() -> None:
    mechanism, state, births, in_use, source_weights = _full_source(cache_rescue_words=_words(1))
    state = state.replace(  # type: ignore[attr-defined]
        live_rescue_words=state.live_rescue_words.at[2].set(_words(2))
    )
    state = mechanism._with_content_token(state)  # noqa: SLF001
    opening = mechanism.propose(state, _opening_event(births, in_use, source_weights))
    second = mechanism.propose(
        opening.state,
        _second_event(opening, source_weights, observation_index=0),
    )

    assert bool(second.quarantine_rejected)
    assert int(second.archive_selected_source) == ARCHIVE_SOURCE_OPENING_VICTIM
    chex.assert_trees_all_equal(second.state.archive.source_birth_words, _words(3))
    chex.assert_trees_all_equal(second.state.archive.rescue_words, _words(2))


def test_invalid_archive_is_replaced_by_current_victim_without_opening() -> None:
    mechanism, state, births, in_use, source_weights = _full_source()
    empty = mechanism.init().archive
    state = mechanism._with_content_token(  # noqa: SLF001
        state.replace(archive=empty)  # type: ignore[attr-defined]
    )
    result = mechanism.propose(state, _opening_event(births, in_use, source_weights))

    assert bool(result.update_applied)
    assert not bool(result.cache_tested)
    assert not bool(result.quarantine_opened)
    assert int(result.archive_selected_source) == ARCHIVE_SOURCE_CURRENT_VICTIM
    chex.assert_trees_all_equal(result.state.archive.source_birth_words, _words(3))
    chex.assert_trees_all_equal(result.state.archive.reward_weights, source_weights[2])
