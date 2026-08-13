"""Opt-in transactional WP3 objectives for :class:`PrototypeAgent`.

This module composes the isolated multi-timescale GVF/inverse-action kernel
with a Prototype agent that uses :class:`OnlineGatedStateBuilder`.  One outer
transaction authenticates the exact dispatched Prototype decision, consumes
only the resulting reward, continuation, and bootstrap observation, updates
the auxiliary heads, routes both representation-owner gradients through the
builder's public proposal/commit boundary, and caches the next dispatched
decision.  Any failed component rolls the complete composition back, including
Prototype RNG and learner state, so the same environment receipt can be
retried.

The current and bootstrap representation gradients are converted separately
through their matching recurrent sensitivities.  Their parameter gradients
are summed, globally clipped to the builder's declared limit, and committed as
one source-bound transformed proposal into the already-advanced destination
builder.  At an episode boundary the bootstrap owner is the final observation
reached by the action; the autoreset observation is used only for the next
decision cache.

This is an L0, nonpromoting, ``not_assessed`` integration mechanism.  It does
not establish calibrated objective balance, feature utility, retention,
control benefit, Forager performance, Alberta Plan completion, or SOTA status.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, UInt

from alberta_framework.core.balanced_state_objectives import (
    BALANCED_STATE_OBJECTIVES_EVIDENCE_LEVEL,
    BALANCED_STATE_OBJECTIVES_OUTCOME_STATUS,
    BalancedStateObjectives,
    BalancedStateObjectivesState,
    BalancedStateObjectiveUpdateResult,
    StateObjectiveActionCacheResult,
    StateObjectiveActionReceipt,
    measure_balanced_state_objectives_state_nbytes,
)
from alberta_framework.core.checkpoints import (
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)
from alberta_framework.core.prototype_agent import (
    PrototypeAgent,
    PrototypeAgentState,
    PrototypeTransition,
    PrototypeTransitionDiagnostics,
    measure_prototype_agent_state_resources,
)
from alberta_framework.core.state_builder import (
    OnlineGatedStateBuilder,
    OnlineGatedStateBuilderConfig,
    OnlineGatedStateBuilderState,
    OnlineGatedStateBuilderTransitionResult,
    StateBuilderLearningDiagnostics,
    replace_state_builder_learning_proposal_update,
)

PROTOTYPE_BALANCED_OBJECTIVES_CONFIG_SCHEMA = (
    "alberta.prototype-balanced-state-objectives-config.v1"
)
PROTOTYPE_BALANCED_OBJECTIVES_STATE_SCHEMA = (
    "alberta.prototype-balanced-state-objectives-state.v1"
)
PROTOTYPE_BALANCED_OBJECTIVES_CHECKPOINT_SCHEMA = (
    "alberta.prototype-balanced-state-objectives-checkpoint.v1"
)
PROTOTYPE_BALANCED_OBJECTIVES_RESOURCE_SCHEMA = (
    "alberta.prototype-balanced-state-objectives-resource.v1"
)
PROTOTYPE_BALANCED_OBJECTIVES_EVIDENCE_LEVEL = "L0"
PROTOTYPE_BALANCED_OBJECTIVES_OUTCOME_STATUS = "not_assessed"
PROTOTYPE_BALANCED_OBJECTIVES_LIFETIME_SEMANTICS = "exact-uint64-fail-stop"
PROTOTYPE_BALANCED_OBJECTIVES_OWNERSHIP = (
    "exact-prototype-decision-and-action; bit-exact-representation-receipt; "
    "observation-event-revision; decision-time-builder-parameter-revision; "
    "source-bound-online-builder-commit"
)
PROTOTYPE_BALANCED_OBJECTIVES_LIMITATIONS = (
    "opt-in-online-gated-builder-only",
    "one-reward-cumulant-and-prototype-continuation",
    "online-recurrent-sensitivity-approximation-after-parameter-updates",
    "fixed-not-empirically-calibrated-objective-balance",
    "no-feature-lifecycle-or-concurrent-builder-learning",
    "no-retention-control-forager-or-sota-evidence",
)
# One objective decision identity is consumed at ``start`` and every accepted
# transition must reserve the next dispatched identity atomically.  Therefore
# the portable armed-continuation bound is one less than uint64's maximum
# value; a coincident externally disarmed final Prototype event is not assumed.
PROTOTYPE_BALANCED_OBJECTIVES_MAX_TRANSITIONS = 2**64 - 2

_UINT32_MAX = 2**32 - 1
_FLOAT32_MAX = float(np.finfo(np.float32).max)


def _exact_manifest(
    payload: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> dict[str, Any]:
    if type(payload) is not dict:
        raise TypeError(f"{label} must be an exact dict")
    fields = dict(payload)
    supplied = set(fields)
    if supplied != expected:
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        raise ValueError(f"{label} field manifest is not exact; missing={missing}, extra={extra}")
    return fields


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_array(
    value: Any,
    *,
    label: str,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> Array:
    if getattr(value, "shape", None) != shape:
        raise ValueError(f"{label} must have shape {shape}")
    if getattr(value, "dtype", None) != dtype:
        raise TypeError(f"{label} must have dtype {dtype}")
    return jnp.asarray(value)


def _require_words(value: Any, *, label: str, width: int = 2) -> Array:
    return _require_array(
        value,
        label=label,
        shape=(width,),
        dtype=jnp.dtype(jnp.uint32),
    )


def _require_bool_scalar(value: Any, *, label: str) -> Array:
    return _require_array(
        value,
        label=label,
        shape=(),
        dtype=jnp.dtype(jnp.bool_),
    )


def _require_threefry_key(value: Any, *, label: str) -> None:
    try:
        key_data = jr.key_data(value)
        implementation = str(jr.key_impl(value))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be one typed Threefry JAX key") from exc
    if (
        getattr(value, "shape", None) != ()
        or key_data.shape != (2,)
        or key_data.dtype != jnp.dtype(jnp.uint32)
        or implementation != "threefry2x32"
    ):
        raise TypeError(f"{label} must be one typed Threefry JAX key")


def _increment_words(words: Array) -> tuple[UInt[Array, " 2"], Bool[Array, ""]]:
    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    carry = words[1] == maximum
    capacity = ~(carry & (words[0] == maximum))
    successor = jnp.stack(
        (
            words[0] + carry.astype(jnp.uint32),
            words[1] + jnp.asarray(1, dtype=jnp.uint32),
        )
    ).astype(jnp.uint32)
    return successor, capacity


def _float32_bits_equal(left: Array, right: Array) -> Bool[Array, ""]:
    return jnp.all(
        jax.lax.bitcast_convert_type(left, jnp.uint32)
        == jax.lax.bitcast_convert_type(right, jnp.uint32)
    )


def _builder_states_equal(
    left: OnlineGatedStateBuilderState,
    right: OnlineGatedStateBuilderState,
) -> Bool[Array, ""]:
    """Compare every fixed builder leaf, including float encodings."""

    return (
        _float32_bits_equal(left.parameters, right.parameters)
        & _float32_bits_equal(left.hidden, right.hidden)
        & _float32_bits_equal(left.parameter_sensitivity, right.parameter_sensitivity)
        & (left.step_count == right.step_count)
        & jnp.all(left.step_words == right.step_words)
        & (left.update_count == right.update_count)
        & jnp.all(left.update_words == right.update_words)
        & _float32_bits_equal(left.last_gradient_norm, right.last_gradient_norm)
    )


def _safe_clip_parameter_gradient(value: Array, limit: float) -> tuple[Array, Array, Array]:
    """Return finite verdict, globally clipped gradient, and safe norm."""

    finite = jnp.all(jnp.isfinite(value))
    safe = jnp.where(finite, value, jnp.zeros_like(value))
    scale = jnp.max(jnp.abs(safe))
    safe_scale = jnp.where(scale > 0.0, scale, jnp.float32(1.0))
    scaled_norm = jnp.sqrt(jnp.sum(jnp.square(safe / safe_scale)))
    raw_norm = scale * scaled_norm
    norm = jnp.where(
        finite & jnp.isfinite(raw_norm),
        raw_norm,
        jnp.asarray(_FLOAT32_MAX, dtype=jnp.float32),
    )
    denominator = jnp.where(norm > 0.0, norm, jnp.float32(1.0))
    factor = jnp.minimum(jnp.float32(1.0), jnp.float32(limit) / denominator)
    clipped = safe * factor
    valid = finite & jnp.all(jnp.isfinite(clipped))
    return valid, clipped, norm


def _state_array_nbytes(state: Any) -> int:
    total = 0
    for leaf in jax.tree.leaves(state):
        if hasattr(leaf, "dtype") and hasattr(leaf, "size"):
            total += int(leaf.size) * int(leaf.dtype.itemsize)
    return total


@chex.dataclass(frozen=True)
class PrototypeBalancedObjectivesState:
    """Atomic Prototype, auxiliary-head, and pending owner state.

    ``pending_builder_update_words`` is the exact parameter revision that
    emitted the cached decision representation.  After the first start it
    equals the current builder revision.  After every accepted adapter
    transition the current builder is exactly one auxiliary commit newer; the
    cached next decision still belongs to the pre-commit revision.
    """

    prototype_state: PrototypeAgentState
    objectives_state: BalancedStateObjectivesState
    pending_prototype_decision_id: UInt[Array, " 4"]
    pending_builder_step_words: UInt[Array, " 2"]
    pending_builder_update_words: UInt[Array, " 2"]
    pending_valid: Bool[Array, ""]
    transaction_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class PrototypeBalancedObjectivesStartResult:
    """Atomic result of priming and binding the first dispatched action."""

    state: PrototypeBalancedObjectivesState
    objective_cache: StateObjectiveActionCacheResult
    source_state_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    start_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeBalancedObjectivesUpdateResult:
    """Attempt diagnostics plus the all-or-nothing composed state."""

    state: PrototypeBalancedObjectivesState
    action: Array
    prototype_transition: PrototypeTransitionDiagnostics
    objective_update: BalancedStateObjectiveUpdateResult
    next_objective_cache: StateObjectiveActionCacheResult
    bootstrap_builder_transition: OnlineGatedStateBuilderTransitionResult
    builder_learning: StateBuilderLearningDiagnostics
    bootstrap_representation: Float[Array, " representation"]
    combined_raw_parameter_gradient_norm: Float[Array, ""]
    pre_transaction_words: UInt[Array, " 2"]
    post_transaction_words: UInt[Array, " 2"]
    source_state_valid: Bool[Array, ""]
    transition_identity_matches: Bool[Array, ""]
    bootstrap_event_capacity_available: Bool[Array, ""]
    bootstrap_transition_applied: Bool[Array, ""]
    prototype_transaction_applied: Bool[Array, ""]
    objective_transaction_applied: Bool[Array, ""]
    builder_sources_match: Bool[Array, ""]
    builder_destination_matches: Bool[Array, ""]
    builder_transaction_applied: Bool[Array, ""]
    next_cache_required: Bool[Array, ""]
    next_cache_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    update_applied: Bool[Array, ""]


@dataclasses.dataclass(frozen=True)
class PrototypeBalancedObjectivesResourceBudget:
    """Exact persistent bytes and fixed per-transition work bounds."""

    schema: str
    prototype_state_nbytes: int
    objectives_state_nbytes: int
    adapter_metadata_nbytes: int
    total_state_nbytes: int
    max_prototype_updates_per_transition: int
    max_objective_head_updates_per_transition: int
    max_builder_proposals_per_transition: int
    max_builder_commits_per_transition: int
    max_next_action_cache_writes_per_transition: int
    max_accepted_transitions: int
    persistent_bytes_scope: str
    diagnostic_bytes_scope: str
    temporary_bytes_scope: str

    def to_config(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def measure_prototype_balanced_objectives_state_nbytes(
    state: PrototypeBalancedObjectivesState,
) -> int:
    """Measure every persistent JAX-array leaf in the composed state."""

    if type(state) is not PrototypeBalancedObjectivesState:
        raise TypeError("state must be an exact PrototypeBalancedObjectivesState")
    return _state_array_nbytes(state)


class PrototypeBalancedStateObjectives:
    """Transactional adapter around Prototype and balanced state objectives."""

    def __init__(
        self,
        prototype: PrototypeAgent,
        objectives: BalancedStateObjectives,
    ) -> None:
        if type(prototype) is not PrototypeAgent:
            raise TypeError("prototype must be an exact PrototypeAgent")
        if type(objectives) is not BalancedStateObjectives:
            raise TypeError("objectives must be an exact BalancedStateObjectives")
        config = prototype.config
        if config.learning_value_router is not None:
            raise ValueError("adapter does not compose with learning_value_router")
        if type(config.state_builder) is not OnlineGatedStateBuilderConfig:
            raise ValueError("adapter requires an exact OnlineGatedStateBuilderConfig")
        if type(prototype.state_builder) is not OnlineGatedStateBuilder:
            raise ValueError("adapter requires an exact OnlineGatedStateBuilder instance")
        if config.prototype_feature_lifecycle is not None:
            raise ValueError("adapter does not compose with prototype_feature_lifecycle")
        if config.learn_state_builder_from_world_model:
            raise ValueError("adapter requires Prototype world-model builder learning disabled")
        if config.representation_gradient_mixer is not None:
            raise ValueError("adapter requires Prototype representation gradient mixing disabled")
        if config.auto_curate_every != 0:
            raise ValueError("adapter requires auto_curate_every == 0")
        builder = prototype.state_builder
        if objectives.config.representation_dim != builder.feature_dim():
            raise ValueError("objective representation_dim must match the builder feature_dim")
        if objectives.config.n_actions != config.oak.n_primitive_actions:
            raise ValueError("objective n_actions must match Prototype primitive actions")
        if BALANCED_STATE_OBJECTIVES_EVIDENCE_LEVEL != "L0":
            raise RuntimeError("balanced objectives must remain an L0 mechanism")
        if BALANCED_STATE_OBJECTIVES_OUTCOME_STATUS != "not_assessed":
            raise RuntimeError("balanced objectives must remain not_assessed")
        self._prototype = prototype
        self._objectives = objectives
        self._builder = builder

    @property
    def prototype(self) -> PrototypeAgent:
        return self._prototype

    @property
    def objectives(self) -> BalancedStateObjectives:
        return self._objectives

    @property
    def builder(self) -> OnlineGatedStateBuilder:
        return self._builder

    def to_config(self) -> dict[str, Any]:
        return {
            "type": "PrototypeBalancedStateObjectives",
            "schema": PROTOTYPE_BALANCED_OBJECTIVES_CONFIG_SCHEMA,
            "state_schema": PROTOTYPE_BALANCED_OBJECTIVES_STATE_SCHEMA,
            "checkpoint_schema": PROTOTYPE_BALANCED_OBJECTIVES_CHECKPOINT_SCHEMA,
            "resource_schema": PROTOTYPE_BALANCED_OBJECTIVES_RESOURCE_SCHEMA,
            "evidence_level": PROTOTYPE_BALANCED_OBJECTIVES_EVIDENCE_LEVEL,
            "outcome_status": PROTOTYPE_BALANCED_OBJECTIVES_OUTCOME_STATUS,
            "ownership": PROTOTYPE_BALANCED_OBJECTIVES_OWNERSHIP,
            "lifetime_semantics": PROTOTYPE_BALANCED_OBJECTIVES_LIFETIME_SEMANTICS,
            "max_transitions": PROTOTYPE_BALANCED_OBJECTIVES_MAX_TRANSITIONS,
            "limitations": list(PROTOTYPE_BALANCED_OBJECTIVES_LIMITATIONS),
            "prototype_config": self._prototype.to_config(),
            "objectives_config": self._objectives.to_config(),
        }

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> PrototypeBalancedStateObjectives:
        expected = {
            "type",
            "schema",
            "state_schema",
            "checkpoint_schema",
            "resource_schema",
            "evidence_level",
            "outcome_status",
            "ownership",
            "lifetime_semantics",
            "max_transitions",
            "limitations",
            "prototype_config",
            "objectives_config",
        }
        fields = _exact_manifest(payload, expected, label="prototype balanced objectives config")
        fixed = {
            "type": "PrototypeBalancedStateObjectives",
            "schema": PROTOTYPE_BALANCED_OBJECTIVES_CONFIG_SCHEMA,
            "state_schema": PROTOTYPE_BALANCED_OBJECTIVES_STATE_SCHEMA,
            "checkpoint_schema": PROTOTYPE_BALANCED_OBJECTIVES_CHECKPOINT_SCHEMA,
            "resource_schema": PROTOTYPE_BALANCED_OBJECTIVES_RESOURCE_SCHEMA,
            "evidence_level": PROTOTYPE_BALANCED_OBJECTIVES_EVIDENCE_LEVEL,
            "outcome_status": PROTOTYPE_BALANCED_OBJECTIVES_OUTCOME_STATUS,
            "ownership": PROTOTYPE_BALANCED_OBJECTIVES_OWNERSHIP,
            "lifetime_semantics": PROTOTYPE_BALANCED_OBJECTIVES_LIFETIME_SEMANTICS,
            "max_transitions": PROTOTYPE_BALANCED_OBJECTIVES_MAX_TRANSITIONS,
            "limitations": list(PROTOTYPE_BALANCED_OBJECTIVES_LIMITATIONS),
        }
        for name, expected_value in fixed.items():
            if fields.pop(name) != expected_value:
                raise ValueError(f"prototype balanced objectives {name} is unsupported")
        prototype_config = fields["prototype_config"]
        objectives_config = fields["objectives_config"]
        if type(prototype_config) is not dict or type(objectives_config) is not dict:
            raise TypeError("nested Prototype and objectives configs must be exact dicts")
        return cls(
            PrototypeAgent.from_config(prototype_config),
            BalancedStateObjectives.from_config(objectives_config),
        )

    def init(
        self,
        key: Array,
        *,
        lifecycle_id: Array | None = None,
    ) -> PrototypeBalancedObjectivesState:
        """Initialize both components from one typed Threefry key."""

        _require_threefry_key(key, label="key")
        prototype_key, objectives_key = jr.split(key)
        prototype_state = self._prototype.init(
            prototype_key,
            lifecycle_id=lifecycle_id,
        )
        return PrototypeBalancedObjectivesState(  # type: ignore[call-arg]
            prototype_state=prototype_state,
            objectives_state=self._objectives.init(objectives_key),
            pending_prototype_decision_id=jnp.zeros((4,), dtype=jnp.uint32),
            pending_builder_step_words=jnp.zeros((2,), dtype=jnp.uint32),
            pending_builder_update_words=jnp.zeros((2,), dtype=jnp.uint32),
            pending_valid=jnp.asarray(False, dtype=jnp.bool_),
            transaction_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def _require_state_contract(self, state: PrototypeBalancedObjectivesState) -> None:
        if type(state) is not PrototypeBalancedObjectivesState:
            raise TypeError("state must be an exact PrototypeBalancedObjectivesState")
        if type(state.prototype_state) is not PrototypeAgentState:
            raise TypeError("state.prototype_state must be an exact PrototypeAgentState")
        if type(state.objectives_state) is not BalancedStateObjectivesState:
            raise TypeError(
                "state.objectives_state must be an exact BalancedStateObjectivesState"
            )
        if type(state.prototype_state.state_builder_state) is not OnlineGatedStateBuilderState:
            raise TypeError("Prototype builder state must be an OnlineGatedStateBuilderState")
        _require_words(
            state.pending_prototype_decision_id,
            label="pending_prototype_decision_id",
            width=4,
        )
        _require_words(state.pending_builder_step_words, label="pending_builder_step_words")
        _require_words(
            state.pending_builder_update_words,
            label="pending_builder_update_words",
        )
        _require_bool_scalar(state.pending_valid, label="pending_valid")
        _require_words(state.transaction_words, label="transaction_words")
        self._objectives.state_valid(state.objectives_state)
        self._builder.state_valid(state.prototype_state.state_builder_state)

    def _dynamic_state_valid(
        self,
        state: PrototypeBalancedObjectivesState,
    ) -> Bool[Array, ""]:
        prototype_state = state.prototype_state
        objective_state = state.objectives_state
        builder_state = cast(
            OnlineGatedStateBuilderState,
            prototype_state.state_builder_state,
        )
        owner_successor, owner_has_successor = _increment_words(
            state.pending_builder_update_words
        )
        initial_decision_owner = (
            jnp.all(state.transaction_words == 0)
            & jnp.all(
                state.pending_builder_update_words == builder_state.update_words
            )
        )
        post_update_decision_owner = (
            jnp.any(state.transaction_words != 0)
            & owner_has_successor
            & jnp.all(owner_successor == builder_state.update_words)
        )
        pending_filled = (
            prototype_state.started
            & objective_state.pending_valid
            & jnp.array_equal(
                state.pending_prototype_decision_id,
                prototype_state.current_decision_id,
            )
            & _float32_bits_equal(
                objective_state.pending_representation,
                prototype_state.current_representation,
            )
            & (objective_state.pending_action == prototype_state.current_action)
            & jnp.all(
                objective_state.pending_representation_revision_words
                == prototype_state.observation_event_words
            )
            & jnp.all(state.pending_builder_step_words == builder_state.step_words)
            & (initial_decision_owner | post_update_decision_owner)
        )
        pending_empty = (
            (~prototype_state.started)
            & (~objective_state.pending_valid)
            & jnp.all(state.pending_prototype_decision_id == 0)
            & jnp.all(state.pending_builder_step_words == 0)
            & jnp.all(state.pending_builder_update_words == 0)
        )
        return (
            self._prototype.validate_state(prototype_state)
            & self._objectives.state_valid(objective_state)
            & self._builder.state_valid(builder_state)
            & (state.pending_valid == objective_state.pending_valid)
            & jnp.where(state.pending_valid, pending_filled, pending_empty)
            & jnp.all(state.transaction_words == prototype_state.step_words)
            & jnp.all(state.transaction_words == objective_state.update_words)
            & jnp.all(state.transaction_words == builder_state.update_words)
        )

    def state_valid(
        self,
        state: PrototypeBalancedObjectivesState,
    ) -> Bool[Array, ""]:
        self._require_state_contract(state)
        return self._dynamic_state_valid(state)

    def _receipt(self, state: BalancedStateObjectivesState) -> StateObjectiveActionReceipt:
        return StateObjectiveActionReceipt(  # type: ignore[call-arg]
            representation=state.pending_representation,
            action=state.pending_action,
            representation_revision_words=state.pending_representation_revision_words,
            action_identity_words=state.pending_action_identity_words,
        )

    def start(
        self,
        state: PrototypeBalancedObjectivesState,
        initial_observation: Array,
    ) -> PrototypeBalancedObjectivesStartResult:
        """Prime Prototype and cache its exact first primitive dispatch."""

        self._require_state_contract(state)
        return cast(
            PrototypeBalancedObjectivesStartResult,
            self._start_jit(state, initial_observation),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _start_jit(
        self,
        state: PrototypeBalancedObjectivesState,
        initial_observation: Array,
    ) -> PrototypeBalancedObjectivesStartResult:
        source_valid = self._dynamic_state_valid(state) & (~state.pending_valid)
        candidate_prototype = self._prototype.start(
            state.prototype_state,
            initial_observation,
        )
        builder_state = cast(
            OnlineGatedStateBuilderState,
            candidate_prototype.state_builder_state,
        )
        cached = self._objectives.cache_action(
            state.objectives_state,
            candidate_prototype.current_representation,
            candidate_prototype.current_action,
            candidate_prototype.observation_event_words,
        )
        candidate = PrototypeBalancedObjectivesState(  # type: ignore[call-arg]
            prototype_state=candidate_prototype,
            objectives_state=cached.state,
            pending_prototype_decision_id=candidate_prototype.current_decision_id,
            pending_builder_step_words=builder_state.step_words,
            pending_builder_update_words=builder_state.update_words,
            pending_valid=jnp.asarray(True, dtype=jnp.bool_),
            transaction_words=state.transaction_words,
        )
        candidate_valid = self._dynamic_state_valid(candidate)
        applied = (
            source_valid
            & candidate_prototype.started
            & cached.cache_applied
            & candidate_valid
        )
        final_state = cast(
            PrototypeBalancedObjectivesState,
            jax.lax.cond(applied, lambda: candidate, lambda: state),
        )
        return PrototypeBalancedObjectivesStartResult(  # type: ignore[call-arg]
            state=final_state,
            objective_cache=cached,
            source_state_valid=source_valid,
            candidate_state_valid=candidate_valid,
            start_applied=applied,
        )

    def update_transition(
        self,
        state: PrototypeBalancedObjectivesState,
        transition: PrototypeTransition,
    ) -> PrototypeBalancedObjectivesUpdateResult:
        """Apply one exact environment receipt as an atomic composition."""

        self._require_state_contract(state)
        if type(transition) is not PrototypeTransition:
            raise TypeError("transition must be an exact PrototypeTransition")
        return cast(
            PrototypeBalancedObjectivesUpdateResult,
            self._update_transition_jit(state, transition),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _update_transition_jit(
        self,
        state: PrototypeBalancedObjectivesState,
        transition: PrototypeTransition,
    ) -> PrototypeBalancedObjectivesUpdateResult:
        source_valid = self._dynamic_state_valid(state) & state.pending_valid
        prototype_state = state.prototype_state
        source_builder = cast(
            OnlineGatedStateBuilderState,
            prototype_state.state_builder_state,
        )
        objective_receipt = self._receipt(state.objectives_state)
        transition_identity_matches = (
            jnp.array_equal(
                transition.decision_id,
                state.pending_prototype_decision_id,
            )
            & (transition.action == objective_receipt.action)
            & _float32_bits_equal(
                transition.observation,
                prototype_state.current_raw_observation,
            )
            & _float32_bits_equal(
                objective_receipt.representation,
                prototype_state.current_representation,
            )
            & jnp.all(
                objective_receipt.representation_revision_words
                == prototype_state.observation_event_words
            )
        )

        prototype_result = self._prototype.update_transition(prototype_state, transition)
        prototype_applied = prototype_result.transition_diagnostics.valid
        bootstrap_transition = self._builder.update_with_status(
            source_builder,
            transition.next_observation,
            transition.action,
            transition.reward,
            transition.discount,
        )
        boundary = transition.terminated | transition.truncated
        reset_builder = self._builder.reset_episode(bootstrap_transition.state)
        restart_transition = self._builder.update_with_status(
            reset_builder,
            transition.next_decision_observation,
            jnp.asarray(-1, dtype=jnp.int32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(1.0, dtype=jnp.float32),
        )
        expected_destination = cast(
            OnlineGatedStateBuilderState,
            jax.lax.cond(
                boundary,
                lambda: restart_transition.state,
                lambda: bootstrap_transition.state,
            ),
        )
        expected_destination_valid = bootstrap_transition.transition_applied & jnp.where(
            boundary,
            restart_transition.transition_applied,
            jnp.asarray(True, dtype=jnp.bool_),
        )
        prototype_destination = cast(
            OnlineGatedStateBuilderState,
            prototype_result.state.state_builder_state,
        )
        destination_matches = _builder_states_equal(
            expected_destination,
            prototype_destination,
        )

        bootstrap_event_words, bootstrap_event_capacity = _increment_words(
            prototype_state.observation_event_words
        )
        objective_update = self._objectives.update(
            state.objectives_state,
            objective_receipt,
            bootstrap_transition.representation,
            bootstrap_event_words,
            transition.reward,
            transition.discount,
        )
        current_proposal = self._builder.propose_learning_update(
            source_builder,
            objective_update.current_representation_gradient,
        )
        next_proposal = self._builder.propose_learning_update(
            bootstrap_transition.state,
            objective_update.next_representation_gradient,
        )
        builder_sources_match = (
            _float32_bits_equal(
                current_proposal.source_parameters,
                next_proposal.source_parameters,
            )
            & (current_proposal.source_update_count == next_proposal.source_update_count)
            & jnp.all(
                current_proposal.source_update_words == next_proposal.source_update_words
            )
            & jnp.all(
                current_proposal.builder_fingerprint == next_proposal.builder_fingerprint
            )
        )
        combined_raw_gradient = (
            current_proposal.raw_parameter_gradient + next_proposal.raw_parameter_gradient
        )
        combined_valid, combined_clipped, combined_norm = _safe_clip_parameter_gradient(
            combined_raw_gradient,
            self._builder.config.gradient_clip,
        )
        combined_parameter_update = (
            -jnp.asarray(self._builder.config.step_size, dtype=jnp.float32)
            * combined_clipped
        )
        proposal_approved = (
            source_valid
            & transition_identity_matches
            & expected_destination_valid
            & prototype_applied
            & objective_update.update_applied
            & current_proposal.valid
            & next_proposal.valid
            & builder_sources_match
            & destination_matches
            & bootstrap_event_capacity
            & combined_valid
        )
        combined_proposal = replace_state_builder_learning_proposal_update(
            current_proposal,
            combined_parameter_update,
            proposal_approved,
        )
        learned_builder, builder_diagnostics = self._builder.commit_learning_update(
            prototype_destination,
            combined_proposal,
        )
        learned_prototype = cast(
            PrototypeAgentState,
            dataclasses.replace(
                cast(Any, prototype_result.state),
                state_builder_state=learned_builder,
            ),
        )

        next_cache = self._objectives.cache_action(
            objective_update.state,
            learned_prototype.current_representation,
            learned_prototype.current_action,
            learned_prototype.observation_event_words,
        )
        next_cache_required = learned_prototype.started
        next_cache_valid = jnp.where(
            next_cache_required,
            next_cache.cache_applied,
            ~objective_update.state.pending_valid,
        )
        candidate_objectives = cast(
            BalancedStateObjectivesState,
            jax.lax.cond(
                next_cache_required,
                lambda: next_cache.state,
                lambda: objective_update.state,
            ),
        )
        proposed_transaction_words, transaction_capacity = _increment_words(
            state.transaction_words
        )
        candidate_pending_decision = jnp.where(
            next_cache_required,
            learned_prototype.current_decision_id,
            jnp.zeros((4,), dtype=jnp.uint32),
        )
        candidate_pending_step = jnp.where(
            next_cache_required,
            learned_builder.step_words,
            jnp.zeros((2,), dtype=jnp.uint32),
        )
        candidate_pending_update = jnp.where(
            next_cache_required,
            prototype_destination.update_words,
            jnp.zeros((2,), dtype=jnp.uint32),
        )
        candidate = PrototypeBalancedObjectivesState(  # type: ignore[call-arg]
            prototype_state=learned_prototype,
            objectives_state=candidate_objectives,
            pending_prototype_decision_id=candidate_pending_decision,
            pending_builder_step_words=candidate_pending_step,
            pending_builder_update_words=candidate_pending_update,
            pending_valid=next_cache_required,
            transaction_words=proposed_transaction_words,
        )
        candidate_valid = self._dynamic_state_valid(candidate)
        applied = (
            proposal_approved
            & builder_diagnostics.applied
            & next_cache_valid
            & transaction_capacity
            & candidate_valid
        )
        final_state = cast(
            PrototypeBalancedObjectivesState,
            jax.lax.cond(applied, lambda: candidate, lambda: state),
        )
        return PrototypeBalancedObjectivesUpdateResult(  # type: ignore[call-arg]
            state=final_state,
            action=final_state.prototype_state.current_action,
            prototype_transition=prototype_result.transition_diagnostics,
            objective_update=objective_update,
            next_objective_cache=next_cache,
            bootstrap_builder_transition=bootstrap_transition,
            builder_learning=builder_diagnostics,
            bootstrap_representation=bootstrap_transition.representation,
            combined_raw_parameter_gradient_norm=combined_norm,
            pre_transaction_words=state.transaction_words,
            post_transaction_words=final_state.transaction_words,
            source_state_valid=source_valid,
            transition_identity_matches=transition_identity_matches,
            bootstrap_event_capacity_available=bootstrap_event_capacity,
            bootstrap_transition_applied=expected_destination_valid,
            prototype_transaction_applied=prototype_applied,
            objective_transaction_applied=objective_update.update_applied,
            builder_sources_match=builder_sources_match,
            builder_destination_matches=destination_matches,
            builder_transaction_applied=builder_diagnostics.applied,
            next_cache_required=next_cache_required,
            next_cache_valid=next_cache_valid,
            lifetime_capacity_available=transaction_capacity,
            candidate_state_valid=candidate_valid,
            update_applied=applied,
        )

    def resource_budget(
        self,
        state: PrototypeBalancedObjectivesState | None = None,
    ) -> PrototypeBalancedObjectivesResourceBudget:
        """Return exact persistent bytes and declared fixed work bounds."""

        reference = self.init(jr.key(0)) if state is None else state
        self._require_state_contract(reference)
        prototype_nbytes = measure_prototype_agent_state_resources(
            reference.prototype_state
        ).total_nbytes
        objectives_nbytes = measure_balanced_state_objectives_state_nbytes(
            reference.objectives_state
        )
        metadata_nbytes = 16 + 8 + 8 + 1 + 8
        budget = PrototypeBalancedObjectivesResourceBudget(
            schema=PROTOTYPE_BALANCED_OBJECTIVES_RESOURCE_SCHEMA,
            prototype_state_nbytes=prototype_nbytes,
            objectives_state_nbytes=objectives_nbytes,
            adapter_metadata_nbytes=metadata_nbytes,
            total_state_nbytes=prototype_nbytes + objectives_nbytes + metadata_nbytes,
            max_prototype_updates_per_transition=1,
            max_objective_head_updates_per_transition=1,
            max_builder_proposals_per_transition=2,
            max_builder_commits_per_transition=1,
            max_next_action_cache_writes_per_transition=1,
            max_accepted_transitions=PROTOTYPE_BALANCED_OBJECTIVES_MAX_TRANSITIONS,
            persistent_bytes_scope=(
                "all-JAX-array-leaves-in-composed-state; excludes-Python-composition-objects"
            ),
            diagnostic_bytes_scope=(
                "result-and-component-diagnostics-excluded-from-persistent-total"
            ),
            temporary_bytes_scope=(
                "source-level-named-arrays; excludes-compiler-and-XLA-workspaces; "
                "not-a-measured-device-peak"
            ),
        )
        if measure_prototype_balanced_objectives_state_nbytes(reference) != (
            budget.total_state_nbytes
        ):
            raise ValueError("composed state allocation differs from its resource declaration")
        return budget


def save_prototype_balanced_objectives_checkpoint(
    adapter: PrototypeBalancedStateObjectives,
    state: PrototypeBalancedObjectivesState,
    path: str | Path,
) -> None:
    """Persist the complete composition with strict L0 metadata."""

    if type(adapter) is not PrototypeBalancedStateObjectives:
        raise TypeError("adapter must be an exact PrototypeBalancedStateObjectives")
    adapter._require_state_contract(state)
    if not bool(adapter.state_valid(state)):
        raise ValueError("cannot checkpoint an invalid composed state")
    config = adapter.to_config()
    save_checkpoint(
        state,
        path,
        metadata={
            "schema": PROTOTYPE_BALANCED_OBJECTIVES_CHECKPOINT_SCHEMA,
            "evidence_level": PROTOTYPE_BALANCED_OBJECTIVES_EVIDENCE_LEVEL,
            "outcome_status": PROTOTYPE_BALANCED_OBJECTIVES_OUTCOME_STATUS,
            "ownership": PROTOTYPE_BALANCED_OBJECTIVES_OWNERSHIP,
            "adapter_config": config,
            "config_sha256": _canonical_digest(config),
            "resource_budget": adapter.resource_budget(state).to_config(),
        },
    )


def load_prototype_balanced_objectives_checkpoint(
    path: str | Path,
) -> tuple[PrototypeBalancedStateObjectives, PrototypeBalancedObjectivesState]:
    """Restore only a canonical, resource-consistent composed checkpoint."""

    metadata = load_checkpoint_metadata(path)
    expected = {
        "schema",
        "evidence_level",
        "outcome_status",
        "ownership",
        "adapter_config",
        "config_sha256",
        "resource_budget",
    }
    fields = _exact_manifest(metadata, expected, label="prototype balanced checkpoint")
    fixed = {
        "schema": PROTOTYPE_BALANCED_OBJECTIVES_CHECKPOINT_SCHEMA,
        "evidence_level": PROTOTYPE_BALANCED_OBJECTIVES_EVIDENCE_LEVEL,
        "outcome_status": PROTOTYPE_BALANCED_OBJECTIVES_OUTCOME_STATUS,
        "ownership": PROTOTYPE_BALANCED_OBJECTIVES_OWNERSHIP,
    }
    for name, expected_value in fixed.items():
        if fields[name] != expected_value:
            raise ValueError(f"prototype balanced checkpoint {name} is unsupported")
    config = fields["adapter_config"]
    if type(config) is not dict:
        raise TypeError("prototype balanced checkpoint config must be an exact dict")
    if fields["config_sha256"] != _canonical_digest(config):
        raise ValueError("prototype balanced checkpoint config digest differs")
    adapter = PrototypeBalancedStateObjectives.from_config(config)
    if adapter.to_config() != config:
        raise ValueError("prototype balanced checkpoint config is noncanonical")
    template = adapter.init(jr.key(0))
    expected_budget = adapter.resource_budget(template).to_config()
    if fields["resource_budget"] != expected_budget:
        raise ValueError("prototype balanced checkpoint resource budget differs")
    restored, restored_metadata = load_checkpoint(template, path)
    if restored_metadata != metadata:
        raise ValueError("prototype balanced checkpoint metadata changed between reads")
    state = cast(PrototypeBalancedObjectivesState, restored)
    adapter._require_state_contract(state)
    if not bool(adapter.state_valid(state)):
        raise ValueError("restored prototype balanced state is invalid")
    adapter.resource_budget(state)
    return adapter, state


__all__ = [
    "PROTOTYPE_BALANCED_OBJECTIVES_CHECKPOINT_SCHEMA",
    "PROTOTYPE_BALANCED_OBJECTIVES_CONFIG_SCHEMA",
    "PROTOTYPE_BALANCED_OBJECTIVES_EVIDENCE_LEVEL",
    "PROTOTYPE_BALANCED_OBJECTIVES_LIFETIME_SEMANTICS",
    "PROTOTYPE_BALANCED_OBJECTIVES_LIMITATIONS",
    "PROTOTYPE_BALANCED_OBJECTIVES_MAX_TRANSITIONS",
    "PROTOTYPE_BALANCED_OBJECTIVES_OUTCOME_STATUS",
    "PROTOTYPE_BALANCED_OBJECTIVES_OWNERSHIP",
    "PROTOTYPE_BALANCED_OBJECTIVES_RESOURCE_SCHEMA",
    "PROTOTYPE_BALANCED_OBJECTIVES_STATE_SCHEMA",
    "PrototypeBalancedObjectivesResourceBudget",
    "PrototypeBalancedObjectivesStartResult",
    "PrototypeBalancedObjectivesState",
    "PrototypeBalancedObjectivesUpdateResult",
    "PrototypeBalancedStateObjectives",
    "load_prototype_balanced_objectives_checkpoint",
    "measure_prototype_balanced_objectives_state_nbytes",
    "save_prototype_balanced_objectives_checkpoint",
]
