# mypy: disable-error-code="call-arg,name-defined"
"""Fixed two-event pairwise-dominance evidence for online quarantine.

The helper is deliberately small and policy-agnostic.  A candidate is compared
with every *other* member of a fixed bank on two consecutive observations.  It
confirms only when it was never worse than each comparator and was strictly
better than each comparator at least once.  The candidate's own slot is masked
out explicitly; its unavoidable self-tie is not evidence against it.

There is no margin, dwell parameter, score, random draw, or configurable
horizon.  Callers own the causal transaction boundary and must ensure the two
observations are consecutive and authenticated.
"""

from __future__ import annotations

from typing import Any

import chex
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, Int

TWO_EVENT_PAIRWISE_DOMINANCE_HORIZON = 2
PAIRWISE_DOMINANCE_OBSERVATION_SCHEMA = (
    "alberta.pairwise-dominance-quarantine.observation.v1"
)
PAIRWISE_DOMINANCE_DECISION_SCHEMA = "alberta.pairwise-dominance-quarantine.decision.v1"


@chex.dataclass(frozen=True)
class PairwiseDominanceObservation:
    """One relational observation with the candidate excluded."""

    comparator_mask: Bool[Array, " bank_size"]
    never_worse: Bool[Array, " bank_size"]
    ever_strict: Bool[Array, " bank_size"]
    valid: Bool[Array, ""]


@chex.dataclass(frozen=True)
class TwoEventPairwiseDominanceDecision:
    """The exact decision after combining two observations."""

    comparator_mask: Bool[Array, " bank_size"]
    never_worse: Bool[Array, " bank_size"]
    ever_strict: Bool[Array, " bank_size"]
    valid: Bool[Array, ""]
    confirmed: Bool[Array, ""]
    rejected: Bool[Array, ""]


def _require_losses(losses: Any) -> Array:
    array = jnp.asarray(losses)
    if array.ndim != 1 or array.shape[0] < 2:
        raise ValueError("losses must have shape (bank_size,) with bank_size >= 2")
    if array.dtype != jnp.dtype(jnp.float32):
        raise TypeError(f"losses must have dtype float32, got {array.dtype}")
    return array


def _require_candidate(candidate: Any) -> Array:
    array = jnp.asarray(candidate)
    if array.shape != ():
        raise ValueError(f"candidate must be scalar, got {array.shape}")
    if array.dtype != jnp.dtype(jnp.int32):
        raise TypeError(f"candidate must have dtype int32, got {array.dtype}")
    return array


def _require_mask(mask: Any, *, name: str, bank_size: int) -> Array:
    array = jnp.asarray(mask)
    if array.shape != (bank_size,):
        raise ValueError(f"{name} must have shape ({bank_size},), got {array.shape}")
    if array.dtype != jnp.dtype(jnp.bool_):
        raise TypeError(f"{name} must have dtype bool, got {array.dtype}")
    return array


def pairwise_dominance_observation(
    losses: Float[Array, " bank_size"],
    candidate: Int[Array, ""],
) -> PairwiseDominanceObservation:
    """Return exact relational masks for one observation.

    The candidate position is always ``False`` in all three masks.  Callers can
    therefore authenticate which slot was excluded without relying on a
    reduction convention.
    """

    checked_losses = _require_losses(losses)
    checked_candidate = _require_candidate(candidate)
    bank_size = checked_losses.shape[0]
    candidate_valid = (checked_candidate >= 0) & (checked_candidate < bank_size)
    safe_candidate = jnp.clip(checked_candidate, 0, bank_size - 1)
    raw_comparator_mask = jnp.arange(bank_size, dtype=jnp.int32) != safe_candidate
    evidence_valid = candidate_valid & jnp.all(jnp.isfinite(checked_losses))
    candidate_loss = checked_losses[safe_candidate]
    comparator_mask = evidence_valid & raw_comparator_mask
    return PairwiseDominanceObservation(
        comparator_mask=comparator_mask,
        never_worse=comparator_mask & (candidate_loss <= checked_losses),
        ever_strict=comparator_mask & (candidate_loss < checked_losses),
        valid=evidence_valid,
    )


def resolve_two_event_pairwise_dominance(
    first_comparator_mask: Bool[Array, " bank_size"],
    first_never_worse: Bool[Array, " bank_size"],
    first_ever_strict: Bool[Array, " bank_size"],
    second_losses: Float[Array, " bank_size"],
    candidate: Int[Array, ""],
) -> TwoEventPairwiseDominanceDecision:
    """Resolve the frozen two-event law without a tunable horizon."""

    checked_losses = _require_losses(second_losses)
    bank_size = checked_losses.shape[0]
    checked_comparators = _require_mask(
        first_comparator_mask,
        name="first_comparator_mask",
        bank_size=bank_size,
    )
    checked_never = _require_mask(
        first_never_worse,
        name="first_never_worse",
        bank_size=bank_size,
    )
    checked_strict = _require_mask(
        first_ever_strict,
        name="first_ever_strict",
        bank_size=bank_size,
    )
    second = pairwise_dominance_observation(checked_losses, candidate)
    prior_valid = (
        jnp.array_equal(checked_comparators, second.comparator_mask)
        & jnp.all((~checked_never) | checked_comparators)
        & jnp.all((~checked_strict) | checked_comparators)
        & jnp.all((~checked_strict) | checked_never)
    )
    valid = second.valid & prior_valid
    never_worse = second.comparator_mask & checked_never & second.never_worse
    ever_strict = second.comparator_mask & (checked_strict | second.ever_strict)
    confirmed = valid & jnp.all(
        (~second.comparator_mask) | (never_worse & ever_strict)
    )
    return TwoEventPairwiseDominanceDecision(
        comparator_mask=second.comparator_mask,
        never_worse=never_worse,
        ever_strict=ever_strict,
        valid=valid,
        confirmed=confirmed,
        rejected=valid & ~confirmed,
    )


__all__ = [
    "PAIRWISE_DOMINANCE_DECISION_SCHEMA",
    "PAIRWISE_DOMINANCE_OBSERVATION_SCHEMA",
    "TWO_EVENT_PAIRWISE_DOMINANCE_HORIZON",
    "PairwiseDominanceObservation",
    "TwoEventPairwiseDominanceDecision",
    "pairwise_dominance_observation",
    "resolve_two_event_pairwise_dominance",
]
