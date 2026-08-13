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
import math
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
EXPERIENTIAL_MEMORY_ADVANTAGE_GATE_CONFIG_SCHEMA = (
    "alberta.experiential-memory-advantage-gate.config.v1"
)
_POLICY_TYPE = "ExperientialMemoryPolicy"
_ACTION_SEMANTICS = "categorical-score-mass-not-action-identifiers"
_SELECTION_SEMANTICS = "lowest-index-argmax-over-safe-positive-mass"


@dataclasses.dataclass(frozen=True, slots=True)
class ExperientialMemoryAdvantageGateConfig:
    """Opt-in conservative authority over a base action.

    The gate interprets only exact one-hot neighbor actions as
    action-conditioned reward evidence.  A memory proposal may replace a
    different base action only when both actions have at least
    ``min_action_support`` selected neighbors and the proposal's local
    similarity-weighted mean reward exceeds the base mean by strictly more
    than ``min_reward_advantage``. Both actions must also own at least
    ``min_action_weight_mass`` of the normalized selected-neighbor mass.
    Missing, fractional, malformed, or non-finite evidence abstains.

    ``min_action_support`` is a raw selected-neighbor count, not an effective
    sample size. The v1 gate has no uncertainty interval, delayed-return
    credit, causal attribution, context inference, or nonstationarity model.
    It is only a conservative authority boundary over local immediate-reward
    evidence and cannot make aliased contexts identifiable.

    This configuration is deliberately separate from the memory and policy
    v1 schemas.  Callers that do not construct the gate retain their exact
    historical proposal semantics.
    """

    min_action_support: int = 1
    min_action_weight_mass: float = 0.1
    min_reward_advantage: float = 0.0

    def __post_init__(self) -> None:
        if type(self.min_action_support) is not int or not (
            1 <= self.min_action_support <= 2**31 - 1
        ):
            raise ValueError("min_action_support must be a positive exact int32")
        for name, value in (
            ("min_action_weight_mass", self.min_action_weight_mass),
            ("min_reward_advantage", self.min_reward_advantage),
        ):
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite exact float")
        represented_mass = float(
            jnp.asarray(self.min_action_weight_mass, dtype=jnp.float32)
        )
        if not math.isfinite(represented_mass) or not 0.0 < represented_mass <= 1.0:
            raise ValueError(
                "min_action_weight_mass must remain in (0, 1] in float32"
            )
        represented_advantage = float(
            jnp.asarray(self.min_reward_advantage, dtype=jnp.float32)
        )
        if not math.isfinite(represented_advantage) or represented_advantage < 0.0:
            raise ValueError(
                "min_reward_advantage must remain finite and non-negative in float32"
            )

    def to_config(self) -> dict[str, object]:
        """Return the exact standalone v1 gate configuration."""

        return {
            "schema": EXPERIENTIAL_MEMORY_ADVANTAGE_GATE_CONFIG_SCHEMA,
            "type": type(self).__name__,
            "mechanism_status": "development-l0",
            "scientific_promotion_allowed": False,
            "min_action_support": self.min_action_support,
            "min_action_weight_mass": self.min_action_weight_mass,
            "min_reward_advantage": self.min_reward_advantage,
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> ExperientialMemoryAdvantageGateConfig:
        """Strictly reconstruct one standalone v1 gate configuration."""

        expected = {
            "schema",
            "type",
            "mechanism_status",
            "scientific_promotion_allowed",
            "min_action_support",
            "min_action_weight_mass",
            "min_reward_advantage",
        }
        if set(config) != expected:
            raise ValueError("experiential-memory advantage gate fields do not match v1")
        fixed = {
            "schema": EXPERIENTIAL_MEMORY_ADVANTAGE_GATE_CONFIG_SCHEMA,
            "type": cls.__name__,
            "mechanism_status": "development-l0",
            "scientific_promotion_allowed": False,
        }
        if any(
            type(config.get(name)) is not type(value)
            or config.get(name) != value
            for name, value in fixed.items()
        ):
            raise ValueError("experiential-memory advantage gate fixed fields are invalid")
        if type(config.get("min_action_support")) is not int:
            raise ValueError("min_action_support must be a JSON integer")
        for name in ("min_action_weight_mass", "min_reward_advantage"):
            if type(config.get(name)) is not float:
                raise ValueError(f"{name} must be a JSON float")
        result = cls(
            min_action_support=cast(int, config["min_action_support"]),
            min_action_weight_mass=cast(float, config["min_action_weight_mass"]),
            min_reward_advantage=cast(float, config["min_reward_advantage"]),
        )
        if result.to_config() != dict(config):
            raise ValueError("experiential-memory advantage gate config is noncanonical")
        return result


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


@chex.dataclass(frozen=True)
class ExperientialMemoryAdvantageGateDiagnostics:
    """Exact local evidence and authority decision for one proposed override."""

    evidence_valid: Bool[Array, ""]
    neighbor_action_one_hot: Bool[Array, " top_k"]
    neighbor_reward_finite: Bool[Array, " top_k"]
    neighbor_weight_contract_valid: Bool[Array, ""]
    action_support_counts: Int[Array, " n_actions"]
    action_weight_mass: Float[Array, " n_actions"]
    action_reward_means: Float[Array, " n_actions"]
    base_action: Int[Array, ""]
    proposed_action: Int[Array, ""]
    actions_differ: Bool[Array, ""]
    base_support_count: Int[Array, ""]
    proposed_support_count: Int[Array, ""]
    min_action_support: Int[Array, ""]
    base_action_weight_mass: Float[Array, ""]
    proposed_action_weight_mass: Float[Array, ""]
    min_action_weight_mass: Float[Array, ""]
    weight_mass_ready: Bool[Array, ""]
    support_ready: Bool[Array, ""]
    base_reward_mean: Float[Array, ""]
    proposed_reward_mean: Float[Array, ""]
    reward_advantage: Float[Array, ""]
    min_reward_advantage: Float[Array, ""]
    advantage_ready: Bool[Array, ""]
    replacement_allowed: Bool[Array, ""]


@dataclasses.dataclass(frozen=True)
class ExperientialMemoryAdvantageGateResourceDeclaration:
    """Fixed logical work and zero persistent state for one assessment."""

    n_actions: int
    top_k: int
    neighbor_action_values_interpreted: int
    neighbor_reward_values_interpreted: int
    neighbor_weight_values_interpreted: int
    owned_persistent_state_bytes: int
    random_draws_per_assessment: int

    def to_config(self) -> dict[str, int]:
        """Return the exact JSON-compatible resource declaration."""

        return dataclasses.asdict(self)


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


class ExperientialMemoryAdvantageGate:
    """Stateless, fail-closed local reward-advantage authority gate.

    The gate consumes a genuine policy proposal together with the exact
    pre-write memory state that produced it.  It neither queries nor mutates
    memory.  Only selected neighbors with exact one-hot stored actions are
    accepted as action-conditioned reward evidence.  This keeps arbitrary
    categorical score vectors from being misrepresented as executed actions.

    Selected-row counts are raw counts; similarity weights are checked through
    a separate minimum-mass floor. Neither is an effective sample size or an
    uncertainty interval. Immediate observed reward is associational evidence,
    not causal intervention credit, so this gate deliberately makes no claim
    to solve state aliasing or changing latent contexts.
    """

    def __init__(
        self,
        memory: ExperientialMemory,
        config: ExperientialMemoryAdvantageGateConfig,
    ) -> None:
        if not isinstance(memory, ExperientialMemory):
            raise TypeError("memory must be ExperientialMemory")
        if type(config) is not ExperientialMemoryAdvantageGateConfig:
            raise TypeError(
                "config must be an exact ExperientialMemoryAdvantageGateConfig"
            )
        if config.min_action_support > memory.config.top_k:
            raise ValueError("min_action_support must not exceed memory top_k")
        self._memory = memory
        self._config = config

    @property
    def memory(self) -> ExperientialMemory:
        """The exact bounded memory whose pre-write state is assessed."""

        return self._memory

    @property
    def config(self) -> ExperientialMemoryAdvantageGateConfig:
        """The exact static authority thresholds."""

        return self._config

    def to_config(self) -> dict[str, object]:
        """Serialize only the standalone gate construction."""

        return self._config.to_config()

    @classmethod
    def from_config(
        cls,
        memory: ExperientialMemory,
        config: Mapping[str, object],
    ) -> ExperientialMemoryAdvantageGate:
        """Reconstruct a gate bound to an already-constructed memory."""

        return cls(
            memory,
            ExperientialMemoryAdvantageGateConfig.from_config(config),
        )

    def resource_declaration(
        self,
    ) -> ExperientialMemoryAdvantageGateResourceDeclaration:
        """Declare exact bounded logical reads and zero owned state."""

        n_actions = self._memory.config.action_dim
        top_k = self._memory.config.top_k
        return ExperientialMemoryAdvantageGateResourceDeclaration(
            n_actions=n_actions,
            top_k=top_k,
            neighbor_action_values_interpreted=top_k * n_actions,
            neighbor_reward_values_interpreted=top_k,
            neighbor_weight_values_interpreted=top_k,
            owned_persistent_state_bytes=0,
            random_draws_per_assessment=0,
        )

    def assess(
        self,
        state: ExperientialMemoryState,
        proposal: ExperientialMemoryPolicyProposal,
        base_action: Int[Array, ""],
    ) -> ExperientialMemoryAdvantageGateDiagnostics:
        """Return whether local evidence authorizes changing ``base_action``."""

        self._memory._validate_state_static_contract(state)
        if not isinstance(proposal, ExperientialMemoryPolicyProposal):
            raise TypeError("proposal must be ExperientialMemoryPolicyProposal")
        _require_array(
            base_action,
            name="base_action",
            shape=(),
            dtype=jnp.int32,
        )
        retrieval = proposal.retrieval
        top_k = self._memory.config.top_k
        n_actions = self._memory.config.action_dim
        for name, shape, dtype in (
            ("proposal.available", (), jnp.bool_),
            ("proposal.action", (), jnp.int32),
            ("retrieval.neighbor_indices", (top_k,), jnp.int32),
            ("retrieval.neighbor_mask", (top_k,), jnp.bool_),
            ("retrieval.neighbor_weights", (top_k,), jnp.float32),
            ("retrieval.accepted", (), jnp.bool_),
            ("retrieval.state_valid", (), jnp.bool_),
        ):
            owner, field = name.split(".", maxsplit=1)
            value = getattr(proposal if owner == "proposal" else retrieval, field)
            _require_array(value, name=name, shape=shape, dtype=dtype)
        _require_array(
            proposal.action_mass,
            name="proposal.action_mass",
            shape=(n_actions,),
            dtype=jnp.float32,
        )
        return cast(
            ExperientialMemoryAdvantageGateDiagnostics,
            self._assess_jit(state, proposal, base_action),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _assess_jit(
        self,
        state: ExperientialMemoryState,
        proposal: ExperientialMemoryPolicyProposal,
        base_action: Array,
    ) -> ExperientialMemoryAdvantageGateDiagnostics:
        cfg = self._config
        n_actions = self._memory.config.action_dim
        capacity = self._memory.config.capacity
        retrieval = proposal.retrieval
        indices = retrieval.neighbor_indices
        neighbor_mask = retrieval.neighbor_mask
        indices_valid = jnp.all((indices >= 0) & (indices < capacity))
        safe_indices = jnp.clip(indices, 0, capacity - 1)
        actions = state.entries.actions[safe_indices]
        rewards = state.entries.rewards[safe_indices]
        rows_valid = state.entries.valid[safe_indices]

        action_finite = jnp.all(jnp.isfinite(actions), axis=1)
        action_binary = jnp.all((actions == 0.0) | (actions == 1.0), axis=1)
        action_one_hot = (
            action_finite
            & action_binary
            & (jnp.sum(actions, axis=1) == 1.0)
        )
        reward_finite = jnp.isfinite(rewards)
        weights = retrieval.neighbor_weights
        weights_finite_nonnegative = jnp.all(jnp.isfinite(weights)) & jnp.all(
            weights >= 0.0
        )
        masked_weights_zero = jnp.all(jnp.where(neighbor_mask, True, weights == 0.0))
        positive_normalized_weight = jnp.isclose(
            jnp.sum(weights),
            jnp.asarray(1.0, dtype=jnp.float32),
            rtol=1.0e-5,
            atol=1.0e-6,
        )
        weight_contract_valid = (
            weights_finite_nonnegative
            & masked_weights_zero
            & positive_normalized_weight
        )
        selected_rows_valid = jnp.all(
            (~neighbor_mask) | (rows_valid & action_one_hot & reward_finite)
        )
        proposal_action_valid = (
            proposal.available
            & (proposal.action >= 0)
            & (proposal.action < n_actions)
        )
        base_action_valid = (base_action >= 0) & (base_action < n_actions)
        evidence_valid = (
            proposal_action_valid
            & base_action_valid
            & retrieval.accepted
            & retrieval.state_valid
            & indices_valid
            & weight_contract_valid
            & selected_rows_valid
        )

        safe_actions = jnp.where(
            action_one_hot,
            jnp.argmax(actions, axis=1).astype(jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )
        evidence_mask = neighbor_mask & rows_valid & action_one_hot & reward_finite
        support_counts = (
            jnp.zeros((n_actions,), dtype=jnp.int32)
            .at[safe_actions]
            .add(evidence_mask.astype(jnp.int32))
        )
        evidence_weights = jnp.where(evidence_mask, weights, 0.0)
        weight_mass = (
            jnp.zeros((n_actions,), dtype=jnp.float32)
            .at[safe_actions]
            .add(evidence_weights)
        )
        weighted_reward_sum = (
            jnp.zeros((n_actions,), dtype=jnp.float32)
            .at[safe_actions]
            .add(evidence_weights * jnp.where(reward_finite, rewards, 0.0))
        )
        reward_means = jnp.where(
            weight_mass > 0.0,
            weighted_reward_sum / jnp.where(weight_mass > 0.0, weight_mass, 1.0),
            jnp.zeros_like(weight_mass),
        )

        safe_base = jnp.clip(base_action, 0, n_actions - 1)
        safe_proposed = jnp.clip(proposal.action, 0, n_actions - 1)
        base_support = support_counts[safe_base]
        proposed_support = support_counts[safe_proposed]
        minimum_support = jnp.asarray(cfg.min_action_support, dtype=jnp.int32)
        base_weight_mass = weight_mass[safe_base]
        proposed_weight_mass = weight_mass[safe_proposed]
        minimum_weight_mass = jnp.asarray(
            cfg.min_action_weight_mass,
            dtype=jnp.float32,
        )
        weight_mass_ready = (
            evidence_valid
            & (base_weight_mass >= minimum_weight_mass)
            & (proposed_weight_mass >= minimum_weight_mass)
        )
        support_ready = (
            evidence_valid
            & (base_support >= minimum_support)
            & (proposed_support >= minimum_support)
            & weight_mass_ready
        )
        base_mean = jnp.where(evidence_valid, reward_means[safe_base], 0.0)
        proposed_mean = jnp.where(evidence_valid, reward_means[safe_proposed], 0.0)
        advantage = jnp.where(
            evidence_valid,
            proposed_mean - base_mean,
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        minimum_advantage = jnp.asarray(
            cfg.min_reward_advantage,
            dtype=jnp.float32,
        )
        advantage_ready = (
            support_ready
            & jnp.isfinite(advantage)
            & (advantage > minimum_advantage)
        )
        actions_differ = proposal_action_valid & base_action_valid & (
            proposal.action != base_action
        )
        replacement_allowed = actions_differ & advantage_ready
        return ExperientialMemoryAdvantageGateDiagnostics(
            evidence_valid=evidence_valid,
            neighbor_action_one_hot=action_one_hot,
            neighbor_reward_finite=reward_finite,
            neighbor_weight_contract_valid=weight_contract_valid,
            action_support_counts=support_counts,
            action_weight_mass=weight_mass,
            action_reward_means=reward_means,
            base_action=base_action,
            proposed_action=proposal.action,
            actions_differ=actions_differ,
            base_support_count=base_support,
            proposed_support_count=proposed_support,
            min_action_support=minimum_support,
            base_action_weight_mass=base_weight_mass,
            proposed_action_weight_mass=proposed_weight_mass,
            min_action_weight_mass=minimum_weight_mass,
            weight_mass_ready=weight_mass_ready,
            support_ready=support_ready,
            base_reward_mean=base_mean,
            proposed_reward_mean=proposed_mean,
            reward_advantage=advantage,
            min_reward_advantage=minimum_advantage,
            advantage_ready=advantage_ready,
            replacement_allowed=replacement_allowed,
        )


__all__ = [
    "EXPERIENTIAL_MEMORY_ADVANTAGE_GATE_CONFIG_SCHEMA",
    "EXPERIENTIAL_MEMORY_POLICY_SCHEMA",
    "ExperientialMemoryAdvantageGate",
    "ExperientialMemoryAdvantageGateConfig",
    "ExperientialMemoryAdvantageGateDiagnostics",
    "ExperientialMemoryAdvantageGateResourceDeclaration",
    "ExperientialMemoryPolicy",
    "ExperientialMemoryPolicyProposal",
    "ExperientialMemoryPolicyResourceDeclaration",
]
