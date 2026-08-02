# mypy: disable-error-code="call-arg"
"""Non-mutating experiential-memory to discrete-policy proposal boundary.

The boundary performs a real :class:`ExperientialMemory` query internally and
interprets the retrieved action vector only as categorical score mass, one
channel per discrete action.  It never treats stored values as integer action
identifiers and never averages or rounds identifiers.  Selection is a
deterministic, lowest-index-tie-broken argmax over caller-declared safe actions
with positive mass.

Raw mass, normalized mass, effective reliability, and retrieval provenance are
kept separate.  None is a calibrated confidence or evidence of benefit.  The
boundary owns no state, uses no randomness, and never mutates the supplied
memory state.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Mapping
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, Int

from alberta_framework.core.experiential_memory import (
    ExperientialMemory,
    ExperientialMemoryRetrieval,
    ExperientialMemoryState,
)

EXPERIENTIAL_MEMORY_POLICY_SCHEMA = "alberta.experiential-memory-policy.v1"
_POLICY_TYPE = "ExperientialMemoryPolicy"
_ACTION_SEMANTICS = "categorical-score-mass-not-action-identifiers"
_SELECTION_SEMANTICS = "lowest-index-argmax-over-safe-positive-mass"


def _require_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> None:
    """Reject shape or dtype drift before eager execution and JAX tracing."""
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must be an array with shape and dtype metadata")
    actual_shape = tuple(cast(Any, value).shape)
    if actual_shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {actual_shape}")
    expected_dtype = jnp.dtype(dtype)
    actual_dtype = jnp.dtype(cast(Any, value).dtype)
    if actual_dtype != expected_dtype:
        raise TypeError(f"{name} must have dtype {expected_dtype}; got {actual_dtype}")


@dataclasses.dataclass(frozen=True)
class ExperientialMemoryPolicyResourceDeclaration:
    """Exact owned state and bounded logical work for one proposal.

    The external memory byte count is the fixed JAX allocation declared by the
    supplied memory.  Logical values interpreted and argmax candidates are not
    FLOP, latency, energy, or compiler-workspace measurements.
    """

    n_actions: int
    owned_trainable_float32_scalars: int
    owned_persistent_state_bytes: int
    external_memory_persistent_state_bytes: int
    memory_queries_per_proposal: int
    random_draws_per_proposal: int
    score_mass_values_interpreted_per_proposal: int
    hard_safety_values_interpreted_per_proposal: int
    argmax_candidates_per_proposal: int

    def to_config(self) -> dict[str, int]:
        """Return the exact JSON-compatible resource declaration."""
        return dataclasses.asdict(self)


@chex.dataclass(frozen=True)
class ExperientialMemoryPolicyProposal:
    """One read-only discrete proposal with separate auditable channels."""

    available: Bool[Array, ""]
    action: Int[Array, ""]
    action_mass: Float[Array, " n_actions"]
    normalized_action_mass: Float[Array, " n_actions"]
    total_action_mass: Float[Array, ""]
    selected_action_mass: Float[Array, ""]
    selected_normalized_action_mass: Float[Array, ""]
    hard_safety_mask: Bool[Array, " n_actions"]
    effective_reliability: Float[Array, ""]
    action_mass_valid: Bool[Array, ""]
    safe_positive_mass_available: Bool[Array, ""]
    retrieval: ExperientialMemoryRetrieval


class ExperientialMemoryPolicy:
    """Stateless categorical proposal boundary over a real bounded memory."""

    def __init__(self, memory: ExperientialMemory):
        if not isinstance(memory, ExperientialMemory):
            raise TypeError("memory must be ExperientialMemory")
        self._memory = memory

    @property
    def memory(self) -> ExperientialMemory:
        """The exact memory construction queried by this boundary."""
        return self._memory

    def to_config(self) -> dict[str, object]:
        """Serialize fixed semantics and the complete memory construction."""
        return {
            "schema": EXPERIENTIAL_MEMORY_POLICY_SCHEMA,
            "type": _POLICY_TYPE,
            "mechanism_status": "development-l0",
            "action_semantics": _ACTION_SEMANTICS,
            "selection_semantics": _SELECTION_SEMANTICS,
            "calibrated_confidence_claimed": False,
            "benefit_claimed": False,
            "scientific_promotion_allowed": False,
            "memory": self._memory.to_config(),
        }

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> ExperientialMemoryPolicy:
        """Strictly reconstruct the policy and reject semantic drift."""
        expected = {
            "schema",
            "type",
            "mechanism_status",
            "action_semantics",
            "selection_semantics",
            "calibrated_confidence_claimed",
            "benefit_claimed",
            "scientific_promotion_allowed",
            "memory",
        }
        if set(config) != expected:
            raise ValueError("experiential-memory policy config fields do not match v1")
        fixed: dict[str, object] = {
            "schema": EXPERIENTIAL_MEMORY_POLICY_SCHEMA,
            "type": _POLICY_TYPE,
            "mechanism_status": "development-l0",
            "action_semantics": _ACTION_SEMANTICS,
            "selection_semantics": _SELECTION_SEMANTICS,
            "calibrated_confidence_claimed": False,
            "benefit_claimed": False,
            "scientific_promotion_allowed": False,
        }
        for name, expected_value in fixed.items():
            if type(config.get(name)) is not type(expected_value) or config.get(
                name
            ) != expected_value:
                if name == "calibrated_confidence_claimed":
                    raise ValueError("experiential-memory policy cannot claim confidence")
                raise ValueError(f"experiential-memory policy {name} is invalid")
        memory_payload = config.get("memory")
        if not isinstance(memory_payload, Mapping) or any(
            type(key) is not str for key in memory_payload
        ):
            raise ValueError("experiential-memory policy memory must be an object")
        memory = ExperientialMemory.from_config(
            cast(dict[str, Any], dict(memory_payload))
        )
        result = cls(memory)
        if result.to_config() != dict(config):
            raise ValueError("experiential-memory policy config is noncanonical")
        return result

    def resource_declaration(self) -> ExperientialMemoryPolicyResourceDeclaration:
        """Declare exact owned state and maximum logical work per proposal."""
        n_actions = self._memory.config.action_dim
        return ExperientialMemoryPolicyResourceDeclaration(
            n_actions=n_actions,
            owned_trainable_float32_scalars=0,
            owned_persistent_state_bytes=0,
            external_memory_persistent_state_bytes=self._memory.persistent_bytes,
            memory_queries_per_proposal=1,
            random_draws_per_proposal=0,
            score_mass_values_interpreted_per_proposal=n_actions,
            hard_safety_values_interpreted_per_proposal=n_actions,
            argmax_candidates_per_proposal=n_actions,
        )

    def state_valid(
        self,
        state: ExperientialMemoryState,
    ) -> Bool[Array, ""]:
        """Return the exact full memory invariant after static validation."""
        self._memory._validate_state_static_contract(state)
        return cast(Bool[Array, ""], self._state_valid_jit(state))

    @functools.partial(jax.jit, static_argnums=(0,))
    def _state_valid_jit(self, state: ExperientialMemoryState) -> Array:
        return self._memory._state_is_valid(state)

    def propose(
        self,
        state: ExperientialMemoryState,
        query_key: Float[Array, " key_dim"],
        representation_version: Int[Array, ""],
        query_uncertainty: Float[Array, ""],
        query_uncertainty_available: Bool[Array, ""],
        hard_safety_mask: Bool[Array, " n_actions"],
    ) -> ExperientialMemoryPolicyProposal:
        """Query memory and propose a safe categorical action without mutation."""
        _require_array(
            hard_safety_mask,
            name="hard_safety_mask",
            shape=(self._memory.config.action_dim,),
            dtype=jnp.bool_,
        )
        # The memory public boundary establishes state and query static
        # contracts.  No caller-supplied retrieval can bypass this query.
        retrieval = self._memory.query(
            state,
            query_key,
            representation_version,
            query_uncertainty,
            query_uncertainty_available,
        )
        return cast(
            ExperientialMemoryPolicyProposal,
            self._interpret_jit(retrieval, hard_safety_mask),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _interpret_jit(
        self,
        retrieval: ExperientialMemoryRetrieval,
        hard_safety_mask: Array,
    ) -> ExperientialMemoryPolicyProposal:
        action_mass = retrieval.action
        finite = jnp.all(jnp.isfinite(action_mass))
        nonnegative = jnp.all(action_mass >= 0.0)
        safe_finite_mass = jnp.where(jnp.isfinite(action_mass), action_mass, 0.0)
        total_action_mass = jnp.sum(safe_finite_mass)
        action_mass_valid = (
            finite
            & nonnegative
            & jnp.isfinite(total_action_mass)
            & (total_action_mass > 0.0)
        )
        denominator = jnp.where(action_mass_valid, total_action_mass, 1.0)
        normalized = jnp.where(
            action_mass_valid,
            safe_finite_mass / denominator,
            jnp.zeros_like(action_mass),
        )
        eligible = hard_safety_mask & (safe_finite_mass > 0.0)
        safe_positive_mass_available = action_mass_valid & jnp.any(eligible)
        selection_scores = jnp.where(eligible, safe_finite_mass, -jnp.inf)
        candidate_action = jnp.argmax(selection_scores).astype(jnp.int32)
        available = retrieval.accepted & safe_positive_mass_available
        action = jnp.where(available, candidate_action, -1).astype(jnp.int32)
        selected_action_mass = jnp.where(
            available,
            safe_finite_mass[candidate_action],
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        selected_normalized_action_mass = jnp.where(
            available,
            normalized[candidate_action],
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        return ExperientialMemoryPolicyProposal(
            available=available,
            action=action,
            action_mass=action_mass,
            normalized_action_mass=normalized,
            total_action_mass=total_action_mass,
            selected_action_mass=selected_action_mass,
            selected_normalized_action_mass=selected_normalized_action_mass,
            hard_safety_mask=hard_safety_mask,
            effective_reliability=retrieval.effective_reliability,
            action_mass_valid=action_mass_valid,
            safe_positive_mass_available=safe_positive_mass_available,
            retrieval=retrieval,
        )


__all__ = [
    "EXPERIENTIAL_MEMORY_POLICY_SCHEMA",
    "ExperientialMemoryPolicy",
    "ExperientialMemoryPolicyProposal",
    "ExperientialMemoryPolicyResourceDeclaration",
]
