"""Contracts for the fixed two-event pairwise-dominance law."""

from __future__ import annotations

import chex
import jax.numpy as jnp
import pytest

from alberta_framework.core.pairwise_dominance_quarantine import (
    TWO_EVENT_PAIRWISE_DOMINANCE_HORIZON,
    pairwise_dominance_observation,
    resolve_two_event_pairwise_dominance,
)

pytestmark = pytest.mark.unit


def test_horizon_is_structurally_fixed_and_candidate_is_not_a_comparator() -> None:
    assert TWO_EVENT_PAIRWISE_DOMINANCE_HORIZON == 2
    first = pairwise_dominance_observation(
        jnp.asarray([0.0, 1.0, 2.0], dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
    )

    assert bool(first.valid)
    chex.assert_trees_all_equal(
        first.comparator_mask,
        jnp.asarray([False, True, True]),
    )
    chex.assert_trees_all_equal(first.never_worse, first.comparator_mask)
    chex.assert_trees_all_equal(first.ever_strict, first.comparator_mask)


@pytest.mark.parametrize(
    ("first_losses", "second_losses"),
    [
        ([0.0, 0.0], [0.0, 1.0]),
        ([0.0, 1.0], [0.0, 0.0]),
    ],
)
def test_one_tie_and_one_strict_comparison_confirms(
    first_losses: list[float],
    second_losses: list[float],
) -> None:
    first = pairwise_dominance_observation(
        jnp.asarray(first_losses, dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    decision = resolve_two_event_pairwise_dominance(
        first.comparator_mask,
        first.never_worse,
        first.ever_strict,
        jnp.asarray(second_losses, dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
    )

    assert bool(decision.valid)
    assert bool(decision.confirmed)
    assert not bool(decision.rejected)


@pytest.mark.parametrize(
    ("first_losses", "second_losses"),
    [
        ([0.0, 0.0], [0.0, 0.0]),
        ([0.0, 1.0], [2.0, 1.0]),
    ],
)
def test_two_ties_or_one_worse_comparison_rejects(
    first_losses: list[float],
    second_losses: list[float],
) -> None:
    first = pairwise_dominance_observation(
        jnp.asarray(first_losses, dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    decision = resolve_two_event_pairwise_dominance(
        first.comparator_mask,
        first.never_worse,
        first.ever_strict,
        jnp.asarray(second_losses, dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
    )

    assert bool(decision.valid)
    assert not bool(decision.confirmed)
    assert bool(decision.rejected)


def test_nonfinite_evidence_fails_closed() -> None:
    first = pairwise_dominance_observation(
        jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    decision = resolve_two_event_pairwise_dominance(
        first.comparator_mask,
        first.never_worse,
        first.ever_strict,
        jnp.asarray([jnp.nan, 1.0], dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
    )

    assert not bool(decision.valid)
    assert not bool(decision.confirmed)
    assert not bool(decision.rejected)


def test_impossible_first_masks_and_different_candidate_identity_are_invalid() -> None:
    losses = jnp.asarray([0.0, 1.0, 2.0], dtype=jnp.float32)
    first = pairwise_dominance_observation(
        losses,
        jnp.asarray(0, dtype=jnp.int32),
    )
    impossible = resolve_two_event_pairwise_dominance(
        first.comparator_mask,
        jnp.asarray([False, False, True]),
        jnp.asarray([False, True, True]),
        losses,
        jnp.asarray(0, dtype=jnp.int32),
    )
    wrong_candidate = resolve_two_event_pairwise_dominance(
        first.comparator_mask,
        first.never_worse,
        first.ever_strict,
        losses,
        jnp.asarray(1, dtype=jnp.int32),
    )

    assert not bool(impossible.valid)
    assert not bool(impossible.confirmed)
    assert not bool(impossible.rejected)
    assert not bool(wrong_candidate.valid)
    assert not bool(wrong_candidate.confirmed)
    assert not bool(wrong_candidate.rejected)
