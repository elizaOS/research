# mypy: disable-error-code="attr-defined,call-arg,no-any-return"
"""Authoritative R35 feature-ledger binding for learned experiential memory.

This additive host/eager wrapper owns one existing
``LearnedExperientialMemoryControllerState`` and binds every valid one of its
64 rows, plus any pending retrieval, to one exact HCCL v2 feature-birth
ledger.  A successful route rebind reconstructs observation and key rows from
one shared ``[physical16, context3, fast4, pair12]`` encoding, reconstructs
outcome rows independently, scrubs inactive and newborn context coordinates,
and stamps the destination ledger and generation.

Two separate transactional operation bindings are also provided.  ``step``
performs exactly one controller query/gate/write call under the current bank;
``settle`` performs exactly one controller feedback call against the currently
stamped pending receipt.  Both re-seal row and pending ledger identities and
fail closed to the complete source wrapper.  The representation version is
always derived from the ledger generation, and transient R35 inputs are
checked against the ledger rather than trusted as opaque float vectors.

Controller clocks, memory clocks, insertion identities, learned parameters,
pending controller contents, and all non-representation payload remain
unchanged during a *pure rebind*.  Operation results are still bounded L0
mechanisms.  Counterfactual feedback remains caller asserted and unauthenticated;
this wrapper owns no action dispatch, outer transaction, planner, environment,
benchmark, artifact, evidence, or scientific-promotion authority.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from typing import Any, ClassVar, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core.experiential_memory import (
    ExperientialMemoryEntries,
    ExperientialMemoryEntry,
    ExperientialMemoryRetrieval,
    ExperientialMemoryState,
)
from alberta_framework.core.hccl_feature_consumer_route import (
    HCCL_FEATURE_CONTEXT_START,
    HCCL_FEATURE_FAST_START,
    HCCL_FEATURE_PAIR_SLOTS,
    HCCL_FEATURE_PAIR_START,
    HCCL_FEATURE_PHYSICAL_DIM,
    HCCL_FEATURE_TOTAL_DIM,
    HCCLFeatureBirthLedger,
    HCCLFeatureConsumerRoute,
    HCCLFeatureConsumerRouteResult,
)
from alberta_framework.core.learned_experiential_memory_controller import (
    LearnedExperientialMemoryController,
    LearnedExperientialMemoryControllerConfig,
    LearnedExperientialMemoryControllerState,
    LearnedExperientialMemoryFeedback,
    LearnedExperientialMemoryFeedbackDiagnostics,
    LearnedExperientialMemoryFeedbackResult,
    LearnedExperientialMemoryStepDiagnostics,
    LearnedExperientialMemoryStepResult,
)

HCCL_FEATURE_BOUND_MEMORY_CONFIG_SCHEMA = (
    "alberta.hccl-feature-bound-memory.config.v1"
)
HCCL_FEATURE_BOUND_MEMORY_STATE_SCHEMA = (
    "alberta.hccl-feature-bound-memory.state.v1"
)
HCCL_FEATURE_BOUND_MEMORY_STATUS = "l0-development-feature-bound-memory-only"
HCCL_FEATURE_BOUND_MEMORY_FULL_INTEGRATION_CLAIMED = False
HCCL_FEATURE_BOUND_MEMORY_SCIENTIFIC_PROMOTION_ALLOWED = False
HCCL_FEATURE_BOUND_MEMORY_COUNTERFACTUAL_FEEDBACK_AUTHENTICATED = False
HCCL_FEATURE_BOUND_MEMORY_ACTION_DISPATCH_AUTHORITY = False
HCCL_FEATURE_BOUND_MEMORY_OUTER_TRANSACTION_AUTHORITY = False
HCCL_FEATURE_BOUND_MEMORY_EVIDENCE_AUTHORITY = False

_CAPACITY = 64
_ACTION_DIM = 2
_BASE_DIM = HCCL_FEATURE_PAIR_START
_TOKEN_NBYTES = 32
_COUNTER_WORDS = 2
_INT32_MAX = 2**31 - 1
_OBSERVATION_PAIR_PRODUCTS = HCCL_FEATURE_PAIR_SLOTS * _CAPACITY
_OUTCOME_PAIR_PRODUCTS = HCCL_FEATURE_PAIR_SLOTS * _CAPACITY
_TOTAL_PAIR_PRODUCTS = _OBSERVATION_PAIR_PRODUCTS + _OUTCOME_PAIR_PRODUCTS
_STEP_REPRESENTATIONS_VALIDATED = 4
_STEP_PAIR_PRODUCTS = _STEP_REPRESENTATIONS_VALIDATED * HCCL_FEATURE_PAIR_SLOTS


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _contains_tracer(value: object) -> bool:
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(value))


def _require_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must expose exact array metadata")
    array = cast(Array, value)
    if tuple(array.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}; got {tuple(array.shape)}")
    expected = jnp.dtype(dtype)
    if jnp.dtype(array.dtype) != expected:
        raise TypeError(f"{name} must have dtype {expected}; got {array.dtype}")
    return array


def _host(value: object) -> np.ndarray[Any, Any]:
    return np.asarray(jax.device_get(value))


def _host_bool(value: object) -> bool:
    return bool(_host(value))


def _array_exact_equal(left: object, right: object) -> bool:
    left_host = np.ascontiguousarray(_host(left))
    right_host = np.ascontiguousarray(_host(right))
    return (
        left_host.dtype == right_host.dtype
        and left_host.shape == right_host.shape
        and left_host.tobytes(order="C") == right_host.tobytes(order="C")
    )


def _tree_exact_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    if cast(object, left_tree) != cast(object, right_tree):
        return False
    if len(left_leaves) != len(right_leaves):
        return False
    return all(
        _array_exact_equal(left_leaf, right_leaf)
        for left_leaf, right_leaf in zip(
            left_leaves,
            right_leaves,
            strict=True,
        )
    )


def _digest_tree(schema: str, *values: object) -> UInt[Array, " 32"]:
    if _contains_tracer(values):
        raise TypeError("HCCL feature-bound memory integrity is host/eager-only")
    digest = hashlib.sha256()
    digest.update(schema.encode("ascii"))
    for value in values:
        digest.update(type(value).__module__.encode("utf-8"))
        digest.update(type(value).__qualname__.encode("utf-8"))
        leaves, structure = jax.tree.flatten(value)
        digest.update(repr(structure).encode("utf-8"))
        digest.update(len(leaves).to_bytes(8, "big"))
        for leaf in leaves:
            host = np.ascontiguousarray(_host(leaf))
            digest.update(str(host.dtype).encode("ascii"))
            digest.update(np.asarray(host.shape, dtype=np.int64).tobytes())
            digest.update(host.tobytes(order="C"))
    return jnp.asarray(tuple(digest.digest()), dtype=jnp.uint8)


def _float32_bits(value: Array) -> Array:
    return jax.lax.bitcast_convert_type(value, jnp.uint32)


def _telemetry_from_words(words: Array) -> Array:
    below_saturation = (words[0] == jnp.uint32(0)) & (
        words[1] < jnp.uint32(_INT32_MAX)
    )
    return jnp.where(
        below_saturation,
        words[1].astype(jnp.int32),
        jnp.int32(_INT32_MAX),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLFeatureBoundMemoryConfig:
    """Exact learned-controller and per-agent R35 composition."""

    agent_index: int
    controller: LearnedExperientialMemoryControllerConfig

    SCHEMA_VERSION: ClassVar[str] = HCCL_FEATURE_BOUND_MEMORY_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if type(self.agent_index) is not int or self.agent_index not in (0, 1):
            raise ValueError("agent_index must be the exact dyad index 0 or 1")
        if type(self.controller) is not LearnedExperientialMemoryControllerConfig:
            raise TypeError(
                "controller must be an exact LearnedExperientialMemoryControllerConfig"
            )
        LearnedExperientialMemoryController(self.controller)
        memory = self.controller.memory
        if memory.capacity != _CAPACITY:
            raise ValueError("learned-memory capacity must equal 64")
        if memory.observation_dim != HCCL_FEATURE_TOTAL_DIM:
            raise ValueError("learned-memory observation_dim must equal 35")
        if memory.key_dim != HCCL_FEATURE_TOTAL_DIM:
            raise ValueError("learned-memory key_dim must equal 35")
        if memory.outcome_dim != HCCL_FEATURE_TOTAL_DIM:
            raise ValueError("learned-memory outcome_dim must equal 35")
        if memory.action_dim != _ACTION_DIM:
            raise ValueError("learned-memory action_dim must equal 2")

    def to_config(self) -> dict[str, object]:
        route = HCCLFeatureConsumerRoute(agent_index=self.agent_index)
        return {
            "type": type(self).__name__,
            "schema": self.SCHEMA_VERSION,
            "state_schema": HCCL_FEATURE_BOUND_MEMORY_STATE_SCHEMA,
            "mechanism_status": HCCL_FEATURE_BOUND_MEMORY_STATUS,
            "agent_index": self.agent_index,
            "controller": self.controller.to_config(),
            "feature_route": route.to_config(),
            "memory_capacity": _CAPACITY,
            "representation_width": HCCL_FEATURE_TOTAL_DIM,
            "representation_order": [
                "physical16",
                "context3",
                "fast4",
                "pair12",
            ],
            "pair_source": "raw-physical16-only",
            "observation_key_encoding_shared": True,
            "outcome_encoding_separate": True,
            "newborn_context_coordinates_zeroed": True,
            "row_ledger_generation_stamps_authoritative": True,
            "pending_ledger_generation_stamp_authoritative": True,
            "pair_products_per_rebind": _TOTAL_PAIR_PRODUCTS,
            "candidate_reencoding_executes_per_static_valid_rebind": True,
            "controller_clock_advances_per_rebind": 0,
            "memory_step_advances_per_rebind": 0,
            "insertion_clock_mutations_per_rebind": 0,
            "rng_draws_per_rebind": 0,
            "host_eager_only": True,
            "full_integration_claimed": False,
            "scientific_promotion_allowed": False,
        }

    @classmethod
    def from_config(cls, payload: Mapping[str, object]) -> HCCLFeatureBoundMemoryConfig:
        if type(payload) is not dict:
            raise TypeError("feature-bound memory config must be an exact dict")
        controller_raw = payload.get("controller")
        if type(controller_raw) is not dict:
            raise ValueError("controller must serialize as an exact dict")
        if type(payload.get("agent_index")) is not int:
            raise ValueError("serialized agent_index must be an exact int")
        result = cls(
            agent_index=cast(int, payload["agent_index"]),
            controller=LearnedExperientialMemoryControllerConfig.from_config(
                controller_raw
            ),
        )
        if _canonical_json_bytes(result.to_config()) != _canonical_json_bytes(payload):
            raise ValueError("feature-bound memory config is noncanonical")
        return result


@chex.dataclass(frozen=True)
class HCCLFeatureBoundMemoryState:
    """Controller state plus authoritative current, row, and pending identities."""

    controller_state: LearnedExperientialMemoryControllerState
    feature_ledger: HCCLFeatureBirthLedger
    row_ledger_content_tokens: UInt[Array, "64 32"]
    row_semantic_generation_words: UInt[Array, "64 2"]
    pending_ledger_content_token: UInt[Array, " 32"]
    pending_semantic_generation_words: UInt[Array, " 2"]
    config_token: UInt[Array, " 32"]
    content_token: UInt[Array, " 32"]


@chex.dataclass(frozen=True)
class HCCLFeatureBoundMemoryRebindWork:
    """Exact history-independent work for one full-capacity rebind attempt."""

    observation_key_pair_products_evaluated: Int[Array, ""]
    outcome_pair_products_evaluated: Int[Array, ""]
    pair_products_evaluated: Int[Array, ""]
    controller_queries: Int[Array, ""]
    controller_writes: Int[Array, ""]
    controller_settlements: Int[Array, ""]
    controller_clock_advances: Int[Array, ""]
    memory_step_advances: Int[Array, ""]
    insertion_clock_mutations: Int[Array, ""]
    rng_draws: Int[Array, ""]


@chex.dataclass(frozen=True)
class HCCLFeatureBoundMemoryRebindDiagnostics:
    """Fixed-shape audit for an applied or fail-closed route rebind."""

    source_state_valid: Bool[Array, ""]
    source_ledger_matches: Bool[Array, ""]
    route_result_valid: Bool[Array, ""]
    route_transaction_applied: Bool[Array, ""]
    reencode_attempted: Bool[Array, ""]
    candidate_values_finite: Bool[Array, ""]
    candidate_controller_state_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]
    complete_source_returned: Bool[Array, ""]
    valid_rows_before: Int[Array, ""]
    valid_rows_reencoded: Int[Array, ""]
    valid_rows_after: Int[Array, ""]
    context_newborn_slots: Int[Array, ""]
    controller_transaction_words_before: UInt[Array, " 2"]
    controller_transaction_words_after: UInt[Array, " 2"]
    memory_step_words_before: UInt[Array, " 2"]
    memory_step_words_after: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class HCCLFeatureBoundMemoryRebindResult:
    """Selected wrapper, unselected candidate, diagnostics, and exact work."""

    state: HCCLFeatureBoundMemoryState
    candidate_state: HCCLFeatureBoundMemoryState
    diagnostics: HCCLFeatureBoundMemoryRebindDiagnostics
    work: HCCLFeatureBoundMemoryRebindWork


@chex.dataclass(frozen=True)
class HCCLFeatureBoundMemorySettleWork:
    """Exact eager work for one bound feedback-settlement attempt."""

    controller_settle_calls: Int[Array, ""]
    settlement_candidate_reconstructions: Int[Array, ""]
    retention_identity_checks: Int[Array, ""]
    row_stamp_evaluations: Int[Array, ""]
    pending_stamp_evaluations: Int[Array, ""]
    controller_step_calls: Int[Array, ""]
    representation_pair_products_evaluated: Int[Array, ""]
    rng_draws: Int[Array, ""]


@chex.dataclass(frozen=True)
class HCCLFeatureBoundMemorySettleDiagnostics:
    """Fail-closed audit of one exact pending-feedback settlement."""

    source_state_valid: Bool[Array, ""]
    pending_available: Bool[Array, ""]
    pending_stamp_matches_current_ledger: Bool[Array, ""]
    donor_result_valid: Bool[Array, ""]
    donor_transaction_applied: Bool[Array, ""]
    candidate_controller_state_valid: Bool[Array, ""]
    candidate_stamps_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]
    complete_source_returned: Bool[Array, ""]
    counterfactual_feedback_authenticated: Bool[Array, ""]
    action_dispatch_authority: Bool[Array, ""]
    outer_transaction_authority: Bool[Array, ""]
    evidence_authority: Bool[Array, ""]


@chex.dataclass(frozen=True)
class HCCLFeatureBoundMemorySettleResult:
    """Selected wrapper, donor feedback result, audit, and exact work."""

    state: HCCLFeatureBoundMemoryState
    candidate_state: HCCLFeatureBoundMemoryState
    controller_result: LearnedExperientialMemoryFeedbackResult
    diagnostics: HCCLFeatureBoundMemorySettleDiagnostics
    work: HCCLFeatureBoundMemorySettleWork


@chex.dataclass(frozen=True)
class HCCLFeatureBoundMemoryStepWork:
    """Exact eager work for one current-bank controller-step attempt."""

    controller_step_calls: Int[Array, ""]
    controller_query_kernels: Int[Array, ""]
    controller_write_attempts: Int[Array, ""]
    controller_settle_calls: Int[Array, ""]
    step_candidate_reconstructions: Int[Array, ""]
    memory_write_validation_replays: Int[Array, ""]
    representations_validated: Int[Array, ""]
    representation_pair_products_evaluated: Int[Array, ""]
    row_stamp_evaluations: Int[Array, ""]
    pending_stamp_evaluations: Int[Array, ""]
    donor_query_replays: Int[Array, ""]
    rng_draws: Int[Array, ""]


@chex.dataclass(frozen=True)
class HCCLFeatureBoundMemoryStepDiagnostics:
    """Fail-closed audit of one exact current-bank query/gate/write step."""

    source_state_valid: Bool[Array, ""]
    ledger_valid: Bool[Array, ""]
    query_representation_valid: Bool[Array, ""]
    entry_observation_valid: Bool[Array, ""]
    entry_key_valid: Bool[Array, ""]
    entry_outcome_valid: Bool[Array, ""]
    entry_key_matches_observation: Bool[Array, ""]
    entry_action_is_categorical_one_hot: Bool[Array, ""]
    entry_source_is_local: Bool[Array, ""]
    entry_provenance_nonnegative: Bool[Array, ""]
    input_semantics_valid: Bool[Array, ""]
    derived_representation_version: Int[Array, ""]
    donor_result_valid: Bool[Array, ""]
    donor_transaction_applied: Bool[Array, ""]
    candidate_controller_state_valid: Bool[Array, ""]
    candidate_matches_donor_controller: Bool[Array, ""]
    candidate_stamps_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]
    complete_source_returned: Bool[Array, ""]
    action_dispatch_authority: Bool[Array, ""]
    outer_transaction_authority: Bool[Array, ""]
    evidence_authority: Bool[Array, ""]


@chex.dataclass(frozen=True)
class HCCLFeatureBoundMemoryStepResult:
    """Selected wrapper, donor step result, audit, and exact work."""

    state: HCCLFeatureBoundMemoryState
    candidate_state: HCCLFeatureBoundMemoryState
    controller_result: LearnedExperientialMemoryStepResult
    diagnostics: HCCLFeatureBoundMemoryStepDiagnostics
    work: HCCLFeatureBoundMemoryStepWork


def _fixed_work() -> HCCLFeatureBoundMemoryRebindWork:
    zero = jnp.asarray(0, dtype=jnp.int32)
    return HCCLFeatureBoundMemoryRebindWork(
        observation_key_pair_products_evaluated=jnp.asarray(
            _OBSERVATION_PAIR_PRODUCTS,
            dtype=jnp.int32,
        ),
        outcome_pair_products_evaluated=jnp.asarray(
            _OUTCOME_PAIR_PRODUCTS,
            dtype=jnp.int32,
        ),
        pair_products_evaluated=jnp.asarray(
            _TOTAL_PAIR_PRODUCTS,
            dtype=jnp.int32,
        ),
        controller_queries=zero,
        controller_writes=zero,
        controller_settlements=zero,
        controller_clock_advances=zero,
        memory_step_advances=zero,
        insertion_clock_mutations=zero,
        rng_draws=zero,
    )


def _checked_words_increment_host(words: Array) -> tuple[Array, bool]:
    host = _host(words)
    high = int(host[0])
    low = int(host[1])
    maximum = 2**32 - 1
    if high == maximum and low == maximum:
        return jnp.asarray((high, low), dtype=jnp.uint32), False
    if low == maximum:
        return jnp.asarray((high + 1, 0), dtype=jnp.uint32), True
    return jnp.asarray((high, low + 1), dtype=jnp.uint32), True


def _saturating_increment(value: Array, condition: bool) -> Array:
    if not condition:
        return value
    return jnp.minimum(
        value,
        jnp.asarray(_INT32_MAX - 1, dtype=jnp.int32),
    ) + jnp.asarray(1, dtype=jnp.int32)


def _exact_bool(value: object, expected: bool) -> bool:
    return _array_exact_equal(
        value,
        jnp.asarray(expected, dtype=jnp.bool_),
    )


def _exact_int32(value: object, expected: int) -> bool:
    return _array_exact_equal(
        value,
        jnp.asarray(expected, dtype=jnp.int32),
    )


class HCCLFeatureBoundMemory:
    """Atomic feature-ledger wrapper around one learned memory controller."""

    def __init__(self, config: HCCLFeatureBoundMemoryConfig) -> None:
        if type(config) is not HCCLFeatureBoundMemoryConfig:
            raise TypeError("config must be an exact HCCLFeatureBoundMemoryConfig")
        self._config = config
        self._controller = LearnedExperientialMemoryController(config.controller)
        self._route = HCCLFeatureConsumerRoute(agent_index=config.agent_index)
        self._config_token = jnp.asarray(
            tuple(hashlib.sha256(_canonical_json_bytes(config.to_config())).digest()),
            dtype=jnp.uint8,
        )

    @property
    def config(self) -> HCCLFeatureBoundMemoryConfig:
        return self._config

    @property
    def controller(self) -> LearnedExperientialMemoryController:
        return self._controller

    @property
    def route(self) -> HCCLFeatureConsumerRoute:
        return self._route

    def to_config(self) -> dict[str, object]:
        return self._config.to_config()

    @classmethod
    def from_config(cls, payload: Mapping[str, object]) -> HCCLFeatureBoundMemory:
        return cls(HCCLFeatureBoundMemoryConfig.from_config(payload))

    def _state_content_token(self, state: HCCLFeatureBoundMemoryState) -> Array:
        return _digest_tree(
            HCCL_FEATURE_BOUND_MEMORY_STATE_SCHEMA,
            state.controller_state,
            state.feature_ledger,
            state.row_ledger_content_tokens,
            state.row_semantic_generation_words,
            state.pending_ledger_content_token,
            state.pending_semantic_generation_words,
            state.config_token,
        )

    def _seal_state(
        self,
        state: HCCLFeatureBoundMemoryState,
    ) -> HCCLFeatureBoundMemoryState:
        return cast(
            HCCLFeatureBoundMemoryState,
            cast(Any, state).replace(content_token=self._state_content_token(state)),
        )

    def _state_static_contract(self, state: HCCLFeatureBoundMemoryState) -> None:
        if type(state) is not HCCLFeatureBoundMemoryState:
            raise TypeError("state must be an exact HCCLFeatureBoundMemoryState")
        self._controller._validate_state_static(state.controller_state)
        for name in ("config_token", "content_token"):
            _require_array(
                getattr(state, name),
                name=f"state.{name}",
                shape=(_TOKEN_NBYTES,),
                dtype=jnp.uint8,
            )
        _require_array(
            state.row_ledger_content_tokens,
            name="state.row_ledger_content_tokens",
            shape=(_CAPACITY, _TOKEN_NBYTES),
            dtype=jnp.uint8,
        )
        _require_array(
            state.row_semantic_generation_words,
            name="state.row_semantic_generation_words",
            shape=(_CAPACITY, _COUNTER_WORDS),
            dtype=jnp.uint32,
        )
        _require_array(
            state.pending_ledger_content_token,
            name="state.pending_ledger_content_token",
            shape=(_TOKEN_NBYTES,),
            dtype=jnp.uint8,
        )
        _require_array(
            state.pending_semantic_generation_words,
            name="state.pending_semantic_generation_words",
            shape=(_COUNTER_WORDS,),
            dtype=jnp.uint32,
        )

    def _result_static_contract(
        self,
        result: HCCLFeatureConsumerRouteResult,
    ) -> None:
        if type(result) is not HCCLFeatureConsumerRouteResult:
            raise TypeError("route_result must be an exact HCCLFeatureConsumerRouteResult")

    def _retrieval_static_contract(
        self,
        retrieval: ExperientialMemoryRetrieval,
        *,
        name: str,
    ) -> None:
        if type(retrieval) is not ExperientialMemoryRetrieval:
            raise TypeError(f"{name} must be an exact ExperientialMemoryRetrieval")
        top_k = self._config.controller.memory.top_k
        for field_name, shape, dtype in (
            ("accepted", (), jnp.bool_),
            ("observation", (HCCL_FEATURE_TOTAL_DIM,), jnp.float32),
            ("action", (_ACTION_DIM,), jnp.float32),
            ("outcome", (HCCL_FEATURE_TOTAL_DIM,), jnp.float32),
            ("reward", (), jnp.float32),
            ("uncertainty", (), jnp.float32),
            ("safety_cost", (), jnp.float32),
            ("effective_reliability", (), jnp.float32),
            ("neighbor_indices", (top_k,), jnp.int32),
            ("neighbor_mask", (top_k,), jnp.bool_),
            ("neighbor_weights", (top_k,), jnp.float32),
            ("neighbor_similarities", (top_k,), jnp.float32),
            ("neighbor_reliabilities", (top_k,), jnp.float32),
            ("neighbor_ages", (top_k,), jnp.int32),
            ("neighbor_provenance_ids", (top_k,), jnp.int32),
            ("state_valid", (), jnp.bool_),
            ("query_valid", (), jnp.bool_),
            ("version_compatible", (), jnp.bool_),
            ("freshness_ok", (), jnp.bool_),
            ("uncertainty_available", (), jnp.bool_),
            ("safety_cost_available", (), jnp.bool_),
            ("uncertainty_ok", (), jnp.bool_),
            ("safety_ok", (), jnp.bool_),
            ("has_neighbors", (), jnp.bool_),
        ):
            _require_array(
                getattr(retrieval, field_name),
                name=f"{name}.{field_name}",
                shape=shape,
                dtype=dtype,
            )

    def _step_result_static_contract(
        self,
        result: LearnedExperientialMemoryStepResult,
    ) -> None:
        if type(result) is not LearnedExperientialMemoryStepResult:
            raise TypeError(
                "controller step result must be an exact "
                "LearnedExperientialMemoryStepResult"
            )
        self._controller._validate_state_static(result.state)
        self._retrieval_static_contract(result.retrieval, name="controller_result.retrieval")
        self._retrieval_static_contract(
            result.fixed_store_retrieval,
            name="controller_result.fixed_store_retrieval",
        )
        for field_name, dtype in (
            ("wrote", jnp.bool_),
            ("slot", jnp.int32),
            ("evicted", jnp.bool_),
            ("evicted_provenance_id", jnp.int32),
        ):
            _require_array(
                getattr(result, field_name),
                name=f"controller_result.{field_name}",
                shape=(),
                dtype=dtype,
            )
        diagnostics = result.diagnostics
        if type(diagnostics) is not LearnedExperientialMemoryStepDiagnostics:
            raise TypeError(
                "controller step diagnostics must be exact "
                "LearnedExperientialMemoryStepDiagnostics"
            )
        for field_name, dtype in (
            ("source_state_valid", jnp.bool_),
            ("input_valid", jnp.bool_),
            ("pending_blocked", jnp.bool_),
            ("fixed_store_retrieval_accepted", jnp.bool_),
            ("learned_admission_score", jnp.float32),
            ("learned_retrieval_admitted", jnp.bool_),
            ("write_succeeded", jnp.bool_),
            ("transaction_applied", jnp.bool_),
            ("pending_created", jnp.bool_),
        ):
            _require_array(
                getattr(diagnostics, field_name),
                name=f"controller_result.diagnostics.{field_name}",
                shape=(),
                dtype=dtype,
            )

    def _settle_result_static_contract(
        self,
        result: LearnedExperientialMemoryFeedbackResult,
    ) -> None:
        if type(result) is not LearnedExperientialMemoryFeedbackResult:
            raise TypeError(
                "controller settle result must be an exact "
                "LearnedExperientialMemoryFeedbackResult"
            )
        self._controller._validate_state_static(result.state)
        diagnostics = result.diagnostics
        if type(diagnostics) is not LearnedExperientialMemoryFeedbackDiagnostics:
            raise TypeError(
                "controller settle diagnostics must be exact "
                "LearnedExperientialMemoryFeedbackDiagnostics"
            )
        for field_name, dtype in (
            ("source_state_valid", jnp.bool_),
            ("pending_available", jnp.bool_),
            ("receipt_matches", jnp.bool_),
            ("feedback_valid", jnp.bool_),
            ("learning_eligible", jnp.bool_),
            ("admission_updated", jnp.bool_),
            ("retention_rows_updated", jnp.int32),
            ("transaction_applied", jnp.bool_),
            ("counterfactual_feedback_authenticated", jnp.bool_),
        ):
            _require_array(
                getattr(diagnostics, field_name),
                name=f"controller_result.diagnostics.{field_name}",
                shape=(),
                dtype=dtype,
            )

    @staticmethod
    def _representation_semantics_valid(
        representation: Array,
        ledger: HCCLFeatureBirthLedger,
    ) -> bool:
        parents = ledger.parents[HCCL_FEATURE_PAIR_START:]
        safe_left = jnp.clip(parents[:, 0], 0, HCCL_FEATURE_PHYSICAL_DIM - 1)
        safe_right = jnp.clip(parents[:, 1], 0, HCCL_FEATURE_PHYSICAL_DIM - 1)
        physical = representation[:HCCL_FEATURE_PHYSICAL_DIM]
        expected_pairs = physical[safe_left] * physical[safe_right]
        pair_active = ledger.active[HCCL_FEATURE_PAIR_START:]
        expected_pairs = jnp.where(pair_active, expected_pairs, jnp.float32(0.0))
        inactive = ~ledger.active
        positive_zero = jnp.asarray(0, dtype=jnp.uint32)
        return _host_bool(
            jnp.all(jnp.isfinite(representation))
            & jnp.all(
                _float32_bits(representation[HCCL_FEATURE_PAIR_START:])
                == _float32_bits(expected_pairs)
            )
            & jnp.all(
                (~inactive)
                | (_float32_bits(representation) == positive_zero)
            )
        )

    @staticmethod
    def _categorical_one_hot_valid(action: Array) -> bool:
        bits = _host(_float32_bits(action))
        zero = np.uint32(0)
        one = np.asarray(np.float32(1.0)).view(np.uint32)
        return bool(
            np.array_equal(bits, np.asarray((one, zero), dtype=np.uint32))
            or np.array_equal(bits, np.asarray((zero, one), dtype=np.uint32))
        )

    @staticmethod
    def _encoded_row_invariants(
        controller_state: LearnedExperientialMemoryControllerState,
        ledger: HCCLFeatureBirthLedger,
    ) -> bool:
        entries = controller_state.memory.entries
        valid = entries.valid
        physical_observations = entries.observations[:, :HCCL_FEATURE_PHYSICAL_DIM]
        physical_outcomes = entries.outcomes[:, :HCCL_FEATURE_PHYSICAL_DIM]
        parents = ledger.parents[HCCL_FEATURE_PAIR_START:]
        pair_active = ledger.active[HCCL_FEATURE_PAIR_START:]
        safe_left = jnp.clip(parents[:, 0], 0, HCCL_FEATURE_PHYSICAL_DIM - 1)
        safe_right = jnp.clip(parents[:, 1], 0, HCCL_FEATURE_PHYSICAL_DIM - 1)
        expected_observation_pairs = (
            physical_observations[:, safe_left]
            * physical_observations[:, safe_right]
        )
        expected_outcome_pairs = (
            physical_outcomes[:, safe_left] * physical_outcomes[:, safe_right]
        )
        expected_observation_pairs = jnp.where(
            pair_active[None, :],
            expected_observation_pairs,
            jnp.float32(0.0),
        )
        expected_outcome_pairs = jnp.where(
            pair_active[None, :],
            expected_outcome_pairs,
            jnp.float32(0.0),
        )
        row_mask = valid[:, None]
        observations_match_keys = jnp.all(
            (~row_mask)
            | (
                _float32_bits(entries.observations)
                == _float32_bits(entries.keys)
            )
        )
        observation_pairs_match = jnp.all(
            (~row_mask)
            | (
                _float32_bits(
                    entries.observations[:, HCCL_FEATURE_PAIR_START:]
                )
                == _float32_bits(expected_observation_pairs)
            )
        )
        outcome_pairs_match = jnp.all(
            (~row_mask)
            | (
                _float32_bits(entries.outcomes[:, HCCL_FEATURE_PAIR_START:])
                == _float32_bits(expected_outcome_pairs)
            )
        )
        inactive_context = ~ledger.active[
            HCCL_FEATURE_CONTEXT_START:HCCL_FEATURE_FAST_START
        ]
        observation_context = entries.observations[
            :, HCCL_FEATURE_CONTEXT_START:HCCL_FEATURE_FAST_START
        ]
        key_context = entries.keys[
            :, HCCL_FEATURE_CONTEXT_START:HCCL_FEATURE_FAST_START
        ]
        outcome_context = entries.outcomes[
            :, HCCL_FEATURE_CONTEXT_START:HCCL_FEATURE_FAST_START
        ]
        positive_zero = jnp.asarray(0, dtype=jnp.uint32)
        inactive_context_zero = jnp.all(
            (~row_mask)
            | (~inactive_context[None, :])
            | (
                (_float32_bits(observation_context) == positive_zero)
                & (_float32_bits(key_context) == positive_zero)
                & (_float32_bits(outcome_context) == positive_zero)
            )
        )
        expected_version = _telemetry_from_words(ledger.semantic_generation_words)
        versions_match = jnp.all(
            (~valid) | (entries.representation_versions == expected_version)
        )
        products_finite = jnp.all(
            (~row_mask)
            | (
                jnp.isfinite(expected_observation_pairs)
                & jnp.isfinite(expected_outcome_pairs)
            )
        )
        return _host_bool(
            observations_match_keys
            & observation_pairs_match
            & outcome_pairs_match
            & inactive_context_zero
            & versions_match
            & products_finite
        )

    @staticmethod
    def _stamp_invariants(
        state: HCCLFeatureBoundMemoryState,
    ) -> bool:
        valid = state.controller_state.memory.entries.valid
        expected_row_tokens = jnp.where(
            valid[:, None],
            state.feature_ledger.content_token[None, :],
            jnp.zeros((_CAPACITY, _TOKEN_NBYTES), dtype=jnp.uint8),
        )
        expected_row_generations = jnp.where(
            valid[:, None],
            state.feature_ledger.semantic_generation_words[None, :],
            jnp.zeros((_CAPACITY, _COUNTER_WORDS), dtype=jnp.uint32),
        )
        pending = state.controller_state.pending.available
        expected_pending_token = jnp.where(
            pending,
            state.feature_ledger.content_token,
            jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8),
        )
        expected_pending_generation = jnp.where(
            pending,
            state.feature_ledger.semantic_generation_words,
            jnp.zeros((_COUNTER_WORDS,), dtype=jnp.uint32),
        )
        return bool(
            _array_exact_equal(
                state.row_ledger_content_tokens,
                expected_row_tokens,
            )
            and _array_exact_equal(
                state.row_semantic_generation_words,
                expected_row_generations,
            )
            and _array_exact_equal(
                state.pending_ledger_content_token,
                expected_pending_token,
            )
            and _array_exact_equal(
                state.pending_semantic_generation_words,
                expected_pending_generation,
            )
        )

    @staticmethod
    def _stored_action_invariants(
        controller_state: LearnedExperientialMemoryControllerState,
    ) -> bool:
        entries = controller_state.memory.entries
        bits = _float32_bits(entries.actions)
        zero = jnp.asarray(0, dtype=jnp.uint32)
        one = _float32_bits(jnp.asarray(1.0, dtype=jnp.float32))
        categorical = (
            ((bits[:, 0] == one) & (bits[:, 1] == zero))
            | ((bits[:, 0] == zero) & (bits[:, 1] == one))
        )
        return _host_bool(jnp.all((~entries.valid) | categorical))

    def _state_is_valid(
        self,
        state: HCCLFeatureBoundMemoryState,
        expected_ledger: HCCLFeatureBirthLedger | None,
    ) -> bool:
        controller_valid = _host_bool(
            self._controller.state_valid(state.controller_state)
        )
        ledger_valid = _host_bool(self._route.ledger_valid(state.feature_ledger))
        expected_matches = (
            True
            if expected_ledger is None
            else _tree_exact_equal(state.feature_ledger, expected_ledger)
        )
        entries = state.controller_state.memory.entries
        local_source = jnp.asarray(self._config.agent_index, dtype=jnp.int32)
        row_sources_local = _host_bool(
            jnp.all((~entries.valid) | (entries.source_ids == local_source))
        )
        pending = state.controller_state.pending
        pending_sources_local = _host_bool(
            jnp.all(
                (~pending.neighbor_mask)
                | (pending.neighbor_source_ids == local_source)
            )
        )
        return bool(
            controller_valid
            and ledger_valid
            and expected_matches
            and row_sources_local
            and pending_sources_local
            and self._stored_action_invariants(state.controller_state)
            and _array_exact_equal(state.config_token, self._config_token)
            and self._encoded_row_invariants(
                state.controller_state,
                state.feature_ledger,
            )
            and self._stamp_invariants(state)
            and _array_exact_equal(
                state.content_token,
                self._state_content_token(state),
            )
        )

    def state_valid(
        self,
        state: HCCLFeatureBoundMemoryState,
        expected_ledger: HCCLFeatureBirthLedger | None = None,
    ) -> Bool[Array, ""]:
        """Validate donor state, exact encoding, ledger stamps, and content."""

        self._state_static_contract(state)
        if _contains_tracer((state, expected_ledger)):
            raise TypeError("HCCL feature-bound memory validity is host/eager-only")
        if expected_ledger is not None:
            self._route.ledger_valid(expected_ledger)
        return jnp.asarray(
            self._state_is_valid(state, expected_ledger),
            dtype=jnp.bool_,
        )

    @staticmethod
    def _row_stamps(
        controller_state: LearnedExperientialMemoryControllerState,
        ledger: HCCLFeatureBirthLedger,
    ) -> tuple[Array, Array, Array, Array]:
        valid = controller_state.memory.entries.valid
        row_tokens = jnp.where(
            valid[:, None],
            ledger.content_token[None, :],
            jnp.zeros((_CAPACITY, _TOKEN_NBYTES), dtype=jnp.uint8),
        )
        row_generations = jnp.where(
            valid[:, None],
            ledger.semantic_generation_words[None, :],
            jnp.zeros((_CAPACITY, _COUNTER_WORDS), dtype=jnp.uint32),
        )
        pending_token = jnp.where(
            controller_state.pending.available,
            ledger.content_token,
            jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8),
        )
        pending_generation = jnp.where(
            controller_state.pending.available,
            ledger.semantic_generation_words,
            jnp.zeros((_COUNTER_WORDS,), dtype=jnp.uint32),
        )
        return row_tokens, row_generations, pending_token, pending_generation

    def _wrap_controller_state(
        self,
        controller_state: LearnedExperientialMemoryControllerState,
        ledger: HCCLFeatureBirthLedger,
    ) -> HCCLFeatureBoundMemoryState:
        stamps = self._row_stamps(controller_state, ledger)
        unsigned = HCCLFeatureBoundMemoryState(
            controller_state=controller_state,
            feature_ledger=ledger,
            row_ledger_content_tokens=stamps[0],
            row_semantic_generation_words=stamps[1],
            pending_ledger_content_token=stamps[2],
            pending_semantic_generation_words=stamps[3],
            config_token=self._config_token,
            content_token=jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8),
        )
        return self._seal_state(unsigned)

    def init(
        self,
        ledger: HCCLFeatureBirthLedger,
        controller_state: LearnedExperientialMemoryControllerState | None = None,
    ) -> HCCLFeatureBoundMemoryState:
        """Bind an empty or already correctly R35-encoded controller state."""

        if _contains_tracer((ledger, controller_state)):
            raise TypeError("HCCL feature-bound memory initialization is host/eager-only")
        if not _host_bool(self._route.ledger_valid(ledger)):
            raise ValueError("initial feature ledger is invalid for this agent")
        selected_controller = (
            self._controller.init() if controller_state is None else controller_state
        )
        self._controller._validate_state_static(selected_controller)
        populated = bool(
            _host_bool(jnp.any(selected_controller.memory.entries.valid))
            or _host_bool(selected_controller.pending.available)
        )
        genesis_ledger = bool(
            _array_exact_equal(
                ledger.source_clock_words,
                jnp.zeros((_COUNTER_WORDS,), dtype=jnp.uint32),
            )
            and _array_exact_equal(
                ledger.semantic_generation_words,
                jnp.zeros((_COUNTER_WORDS,), dtype=jnp.uint32),
            )
        )
        if populated and not genesis_ledger:
            raise ValueError(
                "a populated bare controller can only be bound at genesis; "
                "restore its existing HCCLFeatureBoundMemoryState or use rebind"
            )
        state = self._wrap_controller_state(selected_controller, ledger)
        if not _host_bool(self.state_valid(state, ledger)):
            raise ValueError("initial controller state is not encoded for the ledger")
        return state

    @staticmethod
    def _reset_context(
        base: Array,
        reset_mask: Array,
    ) -> Array:
        context = jnp.where(
            reset_mask[None, :],
            jnp.float32(0.0),
            base[:, HCCL_FEATURE_CONTEXT_START:HCCL_FEATURE_FAST_START],
        )
        return jnp.concatenate(
            (
                base[:, :HCCL_FEATURE_CONTEXT_START],
                context,
                base[:, HCCL_FEATURE_FAST_START:_BASE_DIM],
            ),
            axis=1,
        ).astype(jnp.float32)

    @staticmethod
    def _encode_rows(
        base: Array,
        ledger: HCCLFeatureBirthLedger,
    ) -> tuple[Array, Array]:
        parents = ledger.parents[HCCL_FEATURE_PAIR_START:]
        active = ledger.active[HCCL_FEATURE_PAIR_START:]
        safe_left = jnp.clip(parents[:, 0], 0, HCCL_FEATURE_PHYSICAL_DIM - 1)
        safe_right = jnp.clip(parents[:, 1], 0, HCCL_FEATURE_PHYSICAL_DIM - 1)
        physical = base[:, :HCCL_FEATURE_PHYSICAL_DIM]
        products = physical[:, safe_left] * physical[:, safe_right]
        products = jnp.where(active[None, :], products, jnp.float32(0.0))
        return jnp.concatenate((base, products), axis=1).astype(jnp.float32), products

    def _candidate_state(
        self,
        state: HCCLFeatureBoundMemoryState,
        route_result: HCCLFeatureConsumerRouteResult,
    ) -> tuple[HCCLFeatureBoundMemoryState, bool]:
        destination = route_result.ledger
        entries = state.controller_state.memory.entries
        valid = entries.valid
        context_newborn = route_result.witness.route_map.newborn_mask[
            HCCL_FEATURE_CONTEXT_START:HCCL_FEATURE_FAST_START
        ]
        context_inactive = route_result.witness.route_map.inactive_mask[
            HCCL_FEATURE_CONTEXT_START:HCCL_FEATURE_FAST_START
        ]
        context_reset = context_newborn | context_inactive

        observation_base = self._reset_context(
            entries.observations[:, :_BASE_DIM],
            context_reset,
        )
        encoded_observations, observation_products = self._encode_rows(
            observation_base,
            destination,
        )
        candidate_observations = jnp.where(
            valid[:, None],
            encoded_observations,
            entries.observations,
        )
        candidate_keys = jnp.where(
            valid[:, None],
            candidate_observations,
            entries.keys,
        )

        outcome_base = self._reset_context(
            entries.outcomes[:, :_BASE_DIM],
            context_reset,
        )
        encoded_outcomes, outcome_products = self._encode_rows(
            outcome_base,
            destination,
        )
        candidate_outcomes = jnp.where(
            valid[:, None],
            encoded_outcomes,
            entries.outcomes,
        )
        candidate_versions = jnp.where(
            valid,
            _telemetry_from_words(destination.semantic_generation_words),
            entries.representation_versions,
        ).astype(jnp.int32)
        candidate_entries = cast(
            ExperientialMemoryEntries,
            entries.replace(
                observations=candidate_observations,
                keys=candidate_keys,
                outcomes=candidate_outcomes,
                representation_versions=candidate_versions,
            ),
        )
        candidate_memory = cast(
            ExperientialMemoryState,
            state.controller_state.memory.replace(entries=candidate_entries),
        )
        candidate_controller = cast(
            LearnedExperientialMemoryControllerState,
            state.controller_state.replace(memory=candidate_memory),
        )
        stamps = self._row_stamps(candidate_controller, destination)
        unsigned = HCCLFeatureBoundMemoryState(
            controller_state=candidate_controller,
            feature_ledger=destination,
            row_ledger_content_tokens=stamps[0],
            row_semantic_generation_words=stamps[1],
            pending_ledger_content_token=stamps[2],
            pending_semantic_generation_words=stamps[3],
            config_token=state.config_token,
            content_token=jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8),
        )
        candidate = self._seal_state(unsigned)
        row_mask = valid[:, None]
        values_finite = _host_bool(
            jnp.all((~row_mask) | jnp.isfinite(observation_products))
            & jnp.all((~row_mask) | jnp.isfinite(outcome_products))
            & jnp.all((~row_mask) | jnp.isfinite(candidate_observations))
            & jnp.all((~row_mask) | jnp.isfinite(candidate_outcomes))
        )
        return candidate, values_finite

    @staticmethod
    def _nonrepresentation_preserved(
        source: LearnedExperientialMemoryControllerState,
        candidate: LearnedExperientialMemoryControllerState,
    ) -> bool:
        for field in dataclasses.fields(cast(Any, source)):
            if field.name == "memory":
                continue
            if not _tree_exact_equal(
                getattr(source, field.name),
                getattr(candidate, field.name),
            ):
                return False
        for field in dataclasses.fields(cast(Any, source.memory)):
            if field.name == "entries":
                continue
            if not _tree_exact_equal(
                getattr(source.memory, field.name),
                getattr(candidate.memory, field.name),
            ):
                return False
        representation_fields = {
            "observations",
            "keys",
            "outcomes",
            "representation_versions",
        }
        for field in dataclasses.fields(cast(Any, source.memory.entries)):
            if field.name in representation_fields:
                continue
            if not _tree_exact_equal(
                getattr(source.memory.entries, field.name),
                getattr(candidate.memory.entries, field.name),
            ):
                return False
        return True

    @staticmethod
    def _retrieval_values_finite(retrieval: ExperientialMemoryRetrieval) -> bool:
        return _host_bool(
            jnp.all(jnp.isfinite(retrieval.observation))
            & jnp.all(jnp.isfinite(retrieval.action))
            & jnp.all(jnp.isfinite(retrieval.outcome))
            & jnp.isfinite(retrieval.reward)
            & jnp.isfinite(retrieval.uncertainty)
            & jnp.isfinite(retrieval.safety_cost)
            & jnp.isfinite(retrieval.effective_reliability)
            & jnp.all(jnp.isfinite(retrieval.neighbor_weights))
            & jnp.all(jnp.isfinite(retrieval.neighbor_similarities))
            & jnp.all(jnp.isfinite(retrieval.neighbor_reliabilities))
        )

    def _written_entry_matches(
        self,
        controller_state: LearnedExperientialMemoryControllerState,
        slot: int,
        controlled_entry: ExperientialMemoryEntry,
        insertion_words: Array,
    ) -> bool:
        entries = controller_state.memory.entries
        retention_prior = jnp.asarray(
            self._config.controller.retention_prior,
            dtype=jnp.float32,
        )
        comparisons = (
            _array_exact_equal(entries.observations[slot], controlled_entry.observation),
            _array_exact_equal(entries.keys[slot], controlled_entry.key),
            _array_exact_equal(entries.actions[slot], controlled_entry.action),
            _array_exact_equal(entries.outcomes[slot], controlled_entry.outcome),
            _array_exact_equal(entries.rewards[slot], controlled_entry.reward),
            _array_exact_equal(entries.uncertainties[slot], controlled_entry.uncertainty),
            _array_exact_equal(
                entries.uncertainty_available[slot],
                controlled_entry.uncertainty_available,
            ),
            _array_exact_equal(entries.safety_costs[slot], controlled_entry.safety_cost),
            _array_exact_equal(
                entries.safety_cost_available[slot],
                controlled_entry.safety_cost_available,
            ),
            _array_exact_equal(entries.reliabilities[slot], controlled_entry.reliability),
            _array_exact_equal(entries.utilities[slot], retention_prior),
            _exact_bool(entries.utility_available[slot], True),
            _array_exact_equal(
                entries.representation_versions[slot],
                controlled_entry.representation_version,
            ),
            _exact_bool(entries.valid[slot], True),
            _array_exact_equal(entries.ages[slot], controlled_entry.age),
            _array_exact_equal(entries.recency_ages[slot], controlled_entry.age),
            _array_exact_equal(entries.insertion_step_words[slot], insertion_words),
            _array_exact_equal(entries.last_access_step_words[slot], insertion_words),
            _array_exact_equal(entries.insertion_age_offsets[slot], controlled_entry.age),
            _array_exact_equal(entries.last_access_age_offsets[slot], controlled_entry.age),
            _array_exact_equal(
                entries.provenance_ids[slot],
                controlled_entry.provenance_id,
            ),
            _array_exact_equal(entries.source_ids[slot], controlled_entry.source_id),
            _exact_int32(entries.retrieval_counts[slot], 0),
        )
        return all(comparisons)

    def _step_donor_result_valid(
        self,
        source: LearnedExperientialMemoryControllerState,
        query_key: Array,
        representation_version: Array,
        query_uncertainty: Array,
        query_uncertainty_available: Array,
        controlled_entry: ExperientialMemoryEntry,
        result: LearnedExperientialMemoryStepResult,
    ) -> tuple[bool, bool, bool]:
        source_valid = _host_bool(self._controller.state_valid(source))
        input_valid = _host_bool(
            self._controller.memory._entry_is_valid(
                self._controller._controlled_entry(controlled_entry)
            )
        )
        blocked = _host_bool(source.pending.available)
        next_words, clock_available = _checked_words_increment_host(
            source.transaction_words
        )
        expected_applied = bool(
            source_valid and input_valid and not blocked and clock_available
        )
        diagnostics = result.diagnostics
        common_diagnostics_valid = bool(
            _exact_bool(diagnostics.source_state_valid, source_valid)
            and _exact_bool(diagnostics.input_valid, input_valid)
            and _exact_bool(diagnostics.pending_blocked, blocked)
            and _exact_bool(diagnostics.transaction_applied, expected_applied)
            and _exact_bool(result.wrote, expected_applied)
            and _exact_bool(diagnostics.write_succeeded, expected_applied)
            and _exact_bool(
                diagnostics.fixed_store_retrieval_accepted,
                expected_applied
                and _host_bool(result.fixed_store_retrieval.accepted),
            )
            and self._retrieval_values_finite(result.retrieval)
            and self._retrieval_values_finite(result.fixed_store_retrieval)
        )
        if not common_diagnostics_valid:
            return False, False, False

        if not expected_applied:
            blank = self._controller._blank_retrieval()
            return (
                bool(
                    _tree_exact_equal(result.state, source)
                    and _tree_exact_equal(result.retrieval, blank)
                    and _tree_exact_equal(result.fixed_store_retrieval, blank)
                    and _exact_int32(result.slot, -1)
                    and _exact_bool(result.evicted, False)
                    and _exact_int32(result.evicted_provenance_id, -1)
                    and _exact_bool(
                        diagnostics.learned_retrieval_admitted,
                        False,
                    )
                    and _exact_bool(diagnostics.pending_created, False)
                    and _array_exact_equal(
                        diagnostics.learned_admission_score,
                        jnp.asarray(0.0, dtype=jnp.float32),
                    )
                ),
                False,
                False,
            )

        expected_fixed_retrieval = self._controller.memory._query_jit(
            source.memory,
            query_key,
            representation_version,
            query_uncertainty,
            query_uncertainty_available,
        )
        if not _tree_exact_equal(
            result.fixed_store_retrieval,
            expected_fixed_retrieval,
        ):
            return False, False, True
        if not _host_bool(self._controller.state_valid(result.state)):
            return False, False, True
        features = self._controller._admission_features(
            expected_fixed_retrieval
        )
        score = jnp.dot(source.admission_weights, features)
        admitted = _host_bool(
            expected_fixed_retrieval.accepted
            & jnp.isfinite(score)
            & (
                score
                >= jnp.asarray(
                    self._config.controller.admission_threshold,
                    dtype=jnp.float32,
                )
            )
        )
        expected_retrieval = self._controller._gate_retrieval(
            expected_fixed_retrieval,
            jnp.asarray(admitted, dtype=jnp.bool_),
        )
        expected_pending = (
            self._controller._make_pending(
                source.memory,
                expected_retrieval,
                next_words,
                features,
                score,
            )
            if admitted
            else self._controller._empty_pending()
        )
        advanced_memory = self._controller.memory._advance(source.memory)
        accessed_memory = self._controller.memory._record_query(
            advanced_memory,
            expected_retrieval,
        )
        expected_write = self._controller.memory._write_advanced(
            accessed_memory,
            self._controller._controlled_entry(controlled_entry),
        )
        expected_controller_state = LearnedExperientialMemoryControllerState(
            memory=expected_write.state,
            admission_weights=source.admission_weights,
            transaction_words=next_words,
            feedback_count=source.feedback_count,
            learned_feedback_count=source.learned_feedback_count,
            positive_feedback_count=source.positive_feedback_count,
            nonpositive_feedback_count=source.nonpositive_feedback_count,
            pending=expected_pending,
            config_digest_words=source.config_digest_words,
        )
        slot = int(_host(result.slot))
        slot_valid = 0 <= slot < _CAPACITY
        source_full = int(_host(source.memory.active_count)) == _CAPACITY
        evicted_provenance_valid = (
            slot_valid
            and (
                _exact_int32(
                    result.evicted_provenance_id,
                    int(_host(source.memory.entries.provenance_ids[slot])),
                )
                if source_full
                else _exact_int32(result.evicted_provenance_id, -1)
            )
        )
        controller_fields_preserved = all(
            _tree_exact_equal(getattr(source, name), getattr(result.state, name))
            for name in (
                "admission_weights",
                "feedback_count",
                "learned_feedback_count",
                "positive_feedback_count",
                "nonpositive_feedback_count",
                "config_digest_words",
            )
        )
        return (
            bool(
                _tree_exact_equal(result.state, expected_controller_state)
                and _array_exact_equal(result.wrote, expected_write.wrote)
                and _array_exact_equal(result.slot, expected_write.slot)
                and _array_exact_equal(result.evicted, expected_write.evicted)
                and _array_exact_equal(
                    result.evicted_provenance_id,
                    expected_write.evicted_provenance_id,
                )
                and _array_exact_equal(result.state.transaction_words, next_words)
                and _array_exact_equal(result.state.memory.step_words, next_words)
                and controller_fields_preserved
                and _tree_exact_equal(result.state.pending, expected_pending)
                and _tree_exact_equal(result.retrieval, expected_retrieval)
                and _array_exact_equal(diagnostics.learned_admission_score, score)
                and _exact_bool(diagnostics.learned_retrieval_admitted, admitted)
                and _exact_bool(diagnostics.pending_created, admitted)
                and slot_valid
                and _exact_bool(result.evicted, source_full)
                and evicted_provenance_valid
                and self._written_entry_matches(
                    result.state,
                    slot,
                    controlled_entry,
                    next_words,
                )
            ),
            True,
            True,
        )

    def _settle_donor_result_valid(
        self,
        source: LearnedExperientialMemoryControllerState,
        feedback: LearnedExperientialMemoryFeedback,
        result: LearnedExperientialMemoryFeedbackResult,
    ) -> bool:
        source_valid = _host_bool(self._controller.state_valid(source))
        pending_available = _host_bool(source.pending.available)
        receipt_matches = bool(
            pending_available
            and _array_exact_equal(
                feedback.transaction_words,
                source.pending.transaction_words,
            )
        )
        delta = float(_host(feedback.counterfactual_delta))
        represented_delta_bound = float(
            np.float32(self._config.controller.max_abs_counterfactual_delta)
        )
        feedback_valid = bool(
            np.isfinite(delta)
            and abs(delta) <= represented_delta_bound
        )
        applies = bool(source_valid and receipt_matches and feedback_valid)
        learning_eligible = bool(
            applies
            and _host_bool(feedback.retrieval_used)
            and _host_bool(feedback.counterfactual_available)
        )
        normalized_target = jnp.clip(
            feedback.counterfactual_delta
            / jnp.asarray(
                self._config.controller.max_abs_counterfactual_delta,
                dtype=jnp.float32,
            ),
            -1.0,
            1.0,
        )
        prediction = jnp.tanh(source.pending.admission_score)
        derivative = 1.0 - prediction * prediction
        gradient = (
            (normalized_target - prediction)
            * derivative
            * source.pending.admission_features
        )
        proposed_weights = jnp.clip(
            source.admission_weights
            + jnp.asarray(
                self._config.controller.admission_step_size,
                dtype=jnp.float32,
            )
            * gradient,
            -jnp.asarray(
                self._config.controller.max_abs_admission_weight,
                dtype=jnp.float32,
            ),
            jnp.asarray(
                self._config.controller.max_abs_admission_weight,
                dtype=jnp.float32,
            ),
        )
        expected_weights = jnp.where(
            learning_eligible,
            proposed_weights,
            source.admission_weights,
        )
        retention_target = jnp.clip(0.5 + 0.5 * normalized_target, 0.0, 1.0)
        expected_memory, retention_updates = self._controller._updated_memory_utilities(
            source.memory,
            source.pending,
            retention_target,
            jnp.asarray(learning_eligible, dtype=jnp.bool_),
        )
        expected_candidate = LearnedExperientialMemoryControllerState(
            memory=expected_memory,
            admission_weights=expected_weights,
            transaction_words=source.transaction_words,
            feedback_count=_saturating_increment(source.feedback_count, applies),
            learned_feedback_count=_saturating_increment(
                source.learned_feedback_count,
                learning_eligible,
            ),
            positive_feedback_count=_saturating_increment(
                source.positive_feedback_count,
                learning_eligible and delta > 0.0,
            ),
            nonpositive_feedback_count=_saturating_increment(
                source.nonpositive_feedback_count,
                learning_eligible and delta <= 0.0,
            ),
            pending=self._controller._empty_pending(),
            config_digest_words=source.config_digest_words,
        )
        candidate_valid = _host_bool(
            self._controller.state_valid(expected_candidate)
        )
        committed = bool(applies and candidate_valid)
        expected_state = expected_candidate if committed else source
        diagnostics = result.diagnostics
        return bool(
            _tree_exact_equal(result.state, expected_state)
            and _exact_bool(diagnostics.source_state_valid, source_valid)
            and _exact_bool(diagnostics.pending_available, pending_available)
            and _exact_bool(diagnostics.receipt_matches, receipt_matches)
            and _exact_bool(diagnostics.feedback_valid, feedback_valid)
            and _exact_bool(
                diagnostics.learning_eligible,
                learning_eligible and committed,
            )
            and _exact_bool(
                diagnostics.admission_updated,
                learning_eligible and committed,
            )
            and _array_exact_equal(
                diagnostics.retention_rows_updated,
                jnp.where(
                    committed,
                    retention_updates,
                    jnp.asarray(0, dtype=jnp.int32),
                ).astype(jnp.int32),
            )
            and _exact_bool(diagnostics.transaction_applied, committed)
            and _exact_bool(
                diagnostics.counterfactual_feedback_authenticated,
                False,
            )
        )

    def settle(
        self,
        state: HCCLFeatureBoundMemoryState,
        feedback: LearnedExperientialMemoryFeedback,
    ) -> HCCLFeatureBoundMemorySettleResult:
        """Settle one current-ledger pending receipt or return ``state`` exactly.

        The controller is called exactly once.  The counterfactual meaning of
        ``feedback`` remains caller asserted; successful integrity binding does
        not authenticate it.
        """

        self._state_static_contract(state)
        if type(feedback) is not LearnedExperientialMemoryFeedback:
            raise TypeError(
                "feedback must be an exact LearnedExperientialMemoryFeedback"
            )
        self._controller._validate_feedback_static(
            state.controller_state,
            feedback,
        )
        if _contains_tracer((state, feedback)):
            raise TypeError("HCCL feature-bound memory settle is host/eager-only")

        source_state_valid = _host_bool(self.state_valid(state))
        pending_available = _host_bool(state.controller_state.pending.available)
        zero_token = jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8)
        zero_generation = jnp.zeros((_COUNTER_WORDS,), dtype=jnp.uint32)
        expected_pending_token = (
            state.feature_ledger.content_token if pending_available else zero_token
        )
        expected_pending_generation = (
            state.feature_ledger.semantic_generation_words
            if pending_available
            else zero_generation
        )
        pending_stamp_matches = bool(
            _array_exact_equal(
                state.pending_ledger_content_token,
                expected_pending_token,
            )
            and _array_exact_equal(
                state.pending_semantic_generation_words,
                expected_pending_generation,
            )
        )

        controller_result = self._controller.settle(
            state.controller_state,
            feedback,
        )
        self._settle_result_static_contract(controller_result)
        donor_result_valid = self._settle_donor_result_valid(
            state.controller_state,
            feedback,
            controller_result,
        )
        donor_transaction_applied = _host_bool(
            controller_result.diagnostics.transaction_applied
        )
        candidate = self._wrap_controller_state(
            controller_result.state,
            state.feature_ledger,
        )
        candidate_controller_state_valid = _host_bool(
            self._controller.state_valid(candidate.controller_state)
        )
        candidate_stamps_valid = self._stamp_invariants(candidate)
        candidate_state_valid = _host_bool(
            self.state_valid(candidate, state.feature_ledger)
        )
        transaction_applied = bool(
            source_state_valid
            and pending_stamp_matches
            and donor_result_valid
            and donor_transaction_applied
            and candidate_controller_state_valid
            and candidate_stamps_valid
            and candidate_state_valid
        )
        selected = candidate if transaction_applied else state
        false = jnp.asarray(False, dtype=jnp.bool_)
        diagnostics = HCCLFeatureBoundMemorySettleDiagnostics(
            source_state_valid=jnp.asarray(source_state_valid, dtype=jnp.bool_),
            pending_available=jnp.asarray(pending_available, dtype=jnp.bool_),
            pending_stamp_matches_current_ledger=jnp.asarray(
                pending_stamp_matches,
                dtype=jnp.bool_,
            ),
            donor_result_valid=jnp.asarray(donor_result_valid, dtype=jnp.bool_),
            donor_transaction_applied=jnp.asarray(
                donor_transaction_applied,
                dtype=jnp.bool_,
            ),
            candidate_controller_state_valid=jnp.asarray(
                candidate_controller_state_valid,
                dtype=jnp.bool_,
            ),
            candidate_stamps_valid=jnp.asarray(
                candidate_stamps_valid,
                dtype=jnp.bool_,
            ),
            candidate_state_valid=jnp.asarray(
                candidate_state_valid,
                dtype=jnp.bool_,
            ),
            transaction_applied=jnp.asarray(transaction_applied, dtype=jnp.bool_),
            complete_source_returned=jnp.asarray(
                not transaction_applied,
                dtype=jnp.bool_,
            ),
            counterfactual_feedback_authenticated=false,
            action_dispatch_authority=false,
            outer_transaction_authority=false,
            evidence_authority=false,
        )
        zero = jnp.asarray(0, dtype=jnp.int32)
        one = jnp.asarray(1, dtype=jnp.int32)
        work = HCCLFeatureBoundMemorySettleWork(
            controller_settle_calls=one,
            settlement_candidate_reconstructions=one,
            retention_identity_checks=jnp.asarray(
                self._config.controller.memory.top_k,
                dtype=jnp.int32,
            ),
            row_stamp_evaluations=jnp.asarray(_CAPACITY, dtype=jnp.int32),
            pending_stamp_evaluations=one,
            controller_step_calls=zero,
            representation_pair_products_evaluated=zero,
            rng_draws=zero,
        )
        return HCCLFeatureBoundMemorySettleResult(
            state=selected,
            candidate_state=candidate,
            controller_result=controller_result,
            diagnostics=diagnostics,
            work=work,
        )

    def step(
        self,
        state: HCCLFeatureBoundMemoryState,
        query_key: Float[Array, " 35"],
        query_uncertainty: Float[Array, ""],
        query_uncertainty_available: Bool[Array, ""],
        entry: ExperientialMemoryEntry,
    ) -> HCCLFeatureBoundMemoryStepResult:
        """Query/gate/write once in the current bank, without routing.

        The caller supplies no query representation-version authority.  The
        version used for both the query and stored entry is derived from the
        currently bound ledger's exact generation words; the version field in
        ``entry`` is structurally checked but deliberately ignored.
        """

        self._state_static_contract(state)
        if type(entry) is not ExperientialMemoryEntry:
            raise TypeError("entry must be an exact ExperientialMemoryEntry")
        self._controller.memory._validate_entry_static_contract(entry)
        _require_array(
            query_key,
            name="query_key",
            shape=(HCCL_FEATURE_TOTAL_DIM,),
            dtype=jnp.float32,
        )
        _require_array(
            query_uncertainty,
            name="query_uncertainty",
            shape=(),
            dtype=jnp.float32,
        )
        _require_array(
            query_uncertainty_available,
            name="query_uncertainty_available",
            shape=(),
            dtype=jnp.bool_,
        )
        if _contains_tracer(
            (
                state,
                query_key,
                query_uncertainty,
                query_uncertainty_available,
                entry,
            )
        ):
            raise TypeError("HCCL feature-bound memory step is host/eager-only")

        ledger_valid = _host_bool(self._route.ledger_valid(state.feature_ledger))
        source_state_valid = _host_bool(self.state_valid(state))
        derived_version = _telemetry_from_words(
            state.feature_ledger.semantic_generation_words
        )
        controlled_entry = cast(
            ExperientialMemoryEntry,
            cast(Any, entry).replace(
                representation_version=derived_version,
            ),
        )
        query_representation_valid = self._representation_semantics_valid(
            query_key,
            state.feature_ledger,
        )
        entry_observation_valid = self._representation_semantics_valid(
            controlled_entry.observation,
            state.feature_ledger,
        )
        entry_key_valid = self._representation_semantics_valid(
            controlled_entry.key,
            state.feature_ledger,
        )
        entry_outcome_valid = self._representation_semantics_valid(
            controlled_entry.outcome,
            state.feature_ledger,
        )
        entry_key_matches_observation = _array_exact_equal(
            controlled_entry.key,
            controlled_entry.observation,
        )
        entry_action_one_hot = self._categorical_one_hot_valid(
            controlled_entry.action
        )
        entry_source_local = _exact_int32(
            controlled_entry.source_id,
            self._config.agent_index,
        )
        entry_provenance_nonnegative = int(
            _host(controlled_entry.provenance_id)
        ) >= 0
        input_semantics_valid = bool(
            ledger_valid
            and query_representation_valid
            and entry_observation_valid
            and entry_key_valid
            and entry_outcome_valid
            and entry_key_matches_observation
            and entry_action_one_hot
            and entry_source_local
            and entry_provenance_nonnegative
        )
        _, controller_clock_available = _checked_words_increment_host(
            state.controller_state.transaction_words
        )
        controller_write_attempted = bool(
            _host_bool(self._controller.state_valid(state.controller_state))
            and _host_bool(
                self._controller.memory._entry_is_valid(
                    self._controller._controlled_entry(controlled_entry)
                )
            )
            and not _host_bool(state.controller_state.pending.available)
            and controller_clock_available
        )

        controller_result = self._controller.step(
            state.controller_state,
            query_key,
            derived_version,
            query_uncertainty,
            query_uncertainty_available,
            controlled_entry,
        )
        self._step_result_static_contract(controller_result)
        (
            donor_result_valid,
            step_candidate_reconstructed,
            donor_query_replayed,
        ) = self._step_donor_result_valid(
            state.controller_state,
            query_key,
            derived_version,
            query_uncertainty,
            query_uncertainty_available,
            controlled_entry,
            controller_result,
        )
        donor_transaction_applied = _host_bool(
            controller_result.diagnostics.transaction_applied
        )
        candidate = self._wrap_controller_state(
            controller_result.state,
            state.feature_ledger,
        )
        candidate_controller_state_valid = _host_bool(
            self._controller.state_valid(candidate.controller_state)
        )
        candidate_matches_donor = _tree_exact_equal(
            candidate.controller_state,
            controller_result.state,
        )
        candidate_stamps_valid = self._stamp_invariants(candidate)
        candidate_state_valid = _host_bool(
            self.state_valid(candidate, state.feature_ledger)
        )
        transaction_applied = bool(
            source_state_valid
            and input_semantics_valid
            and donor_result_valid
            and donor_transaction_applied
            and candidate_controller_state_valid
            and candidate_matches_donor
            and candidate_stamps_valid
            and candidate_state_valid
        )
        selected = candidate if transaction_applied else state
        false = jnp.asarray(False, dtype=jnp.bool_)
        diagnostics = HCCLFeatureBoundMemoryStepDiagnostics(
            source_state_valid=jnp.asarray(source_state_valid, dtype=jnp.bool_),
            ledger_valid=jnp.asarray(ledger_valid, dtype=jnp.bool_),
            query_representation_valid=jnp.asarray(
                query_representation_valid,
                dtype=jnp.bool_,
            ),
            entry_observation_valid=jnp.asarray(
                entry_observation_valid,
                dtype=jnp.bool_,
            ),
            entry_key_valid=jnp.asarray(entry_key_valid, dtype=jnp.bool_),
            entry_outcome_valid=jnp.asarray(
                entry_outcome_valid,
                dtype=jnp.bool_,
            ),
            entry_key_matches_observation=jnp.asarray(
                entry_key_matches_observation,
                dtype=jnp.bool_,
            ),
            entry_action_is_categorical_one_hot=jnp.asarray(
                entry_action_one_hot,
                dtype=jnp.bool_,
            ),
            entry_source_is_local=jnp.asarray(
                entry_source_local,
                dtype=jnp.bool_,
            ),
            entry_provenance_nonnegative=jnp.asarray(
                entry_provenance_nonnegative,
                dtype=jnp.bool_,
            ),
            input_semantics_valid=jnp.asarray(
                input_semantics_valid,
                dtype=jnp.bool_,
            ),
            derived_representation_version=derived_version,
            donor_result_valid=jnp.asarray(donor_result_valid, dtype=jnp.bool_),
            donor_transaction_applied=jnp.asarray(
                donor_transaction_applied,
                dtype=jnp.bool_,
            ),
            candidate_controller_state_valid=jnp.asarray(
                candidate_controller_state_valid,
                dtype=jnp.bool_,
            ),
            candidate_matches_donor_controller=jnp.asarray(
                candidate_matches_donor,
                dtype=jnp.bool_,
            ),
            candidate_stamps_valid=jnp.asarray(
                candidate_stamps_valid,
                dtype=jnp.bool_,
            ),
            candidate_state_valid=jnp.asarray(
                candidate_state_valid,
                dtype=jnp.bool_,
            ),
            transaction_applied=jnp.asarray(transaction_applied, dtype=jnp.bool_),
            complete_source_returned=jnp.asarray(
                not transaction_applied,
                dtype=jnp.bool_,
            ),
            action_dispatch_authority=false,
            outer_transaction_authority=false,
            evidence_authority=false,
        )
        zero = jnp.asarray(0, dtype=jnp.int32)
        one = jnp.asarray(1, dtype=jnp.int32)
        work = HCCLFeatureBoundMemoryStepWork(
            controller_step_calls=one,
            controller_query_kernels=one,
            controller_write_attempts=jnp.asarray(
                int(controller_write_attempted),
                dtype=jnp.int32,
            ),
            controller_settle_calls=zero,
            step_candidate_reconstructions=jnp.asarray(
                int(step_candidate_reconstructed),
                dtype=jnp.int32,
            ),
            memory_write_validation_replays=jnp.asarray(
                int(step_candidate_reconstructed),
                dtype=jnp.int32,
            ),
            representations_validated=jnp.asarray(
                _STEP_REPRESENTATIONS_VALIDATED,
                dtype=jnp.int32,
            ),
            representation_pair_products_evaluated=jnp.asarray(
                _STEP_PAIR_PRODUCTS,
                dtype=jnp.int32,
            ),
            row_stamp_evaluations=jnp.asarray(_CAPACITY, dtype=jnp.int32),
            pending_stamp_evaluations=one,
            donor_query_replays=jnp.asarray(
                int(donor_query_replayed),
                dtype=jnp.int32,
            ),
            rng_draws=zero,
        )
        return HCCLFeatureBoundMemoryStepResult(
            state=selected,
            candidate_state=candidate,
            controller_result=controller_result,
            diagnostics=diagnostics,
            work=work,
        )

    def rebind(
        self,
        state: HCCLFeatureBoundMemoryState,
        source_ledger: HCCLFeatureBirthLedger,
        route_result: HCCLFeatureConsumerRouteResult,
    ) -> HCCLFeatureBoundMemoryRebindResult:
        """Apply one exact v2 route rebind or return ``state`` bit-for-bit."""

        self._state_static_contract(state)
        self._result_static_contract(route_result)
        if _contains_tracer((state, source_ledger, route_result)):
            raise TypeError("HCCL feature-bound memory rebind is host/eager-only")

        source_state_valid = _host_bool(self.state_valid(state))
        source_ledger_matches = _tree_exact_equal(
            state.feature_ledger,
            source_ledger,
        )
        route_result_valid = _host_bool(
            self._route.result_integrity_valid(source_ledger, route_result)
        )
        route_transaction_applied = _host_bool(
            route_result.witness.transaction_applied
        )
        prerequisites = bool(
            source_state_valid
            and source_ledger_matches
            and route_result_valid
            and route_transaction_applied
        )
        candidate, candidate_values_finite = self._candidate_state(
            state,
            route_result,
        )
        candidate_controller_state_valid = _host_bool(
            self._controller.state_valid(candidate.controller_state)
        )
        nonrepresentation_preserved = self._nonrepresentation_preserved(
            state.controller_state,
            candidate.controller_state,
        )
        candidate_state_valid = _host_bool(
            self.state_valid(candidate, route_result.ledger)
        ) and nonrepresentation_preserved
        transaction_applied = bool(
            prerequisites
            and candidate_values_finite
            and candidate_controller_state_valid
            and candidate_state_valid
        )
        selected = candidate if transaction_applied else state
        valid_rows_before = int(
            np.sum(_host(state.controller_state.memory.entries.valid))
        )
        valid_rows_after = int(
            np.sum(_host(selected.controller_state.memory.entries.valid))
        )
        context_newborn_slots = int(
            np.sum(
                _host(
                    route_result.witness.route_map.newborn_mask[
                        HCCL_FEATURE_CONTEXT_START:HCCL_FEATURE_FAST_START
                    ]
                )
            )
        )
        diagnostics = HCCLFeatureBoundMemoryRebindDiagnostics(
            source_state_valid=jnp.asarray(source_state_valid, dtype=jnp.bool_),
            source_ledger_matches=jnp.asarray(
                source_ledger_matches,
                dtype=jnp.bool_,
            ),
            route_result_valid=jnp.asarray(route_result_valid, dtype=jnp.bool_),
            route_transaction_applied=jnp.asarray(
                route_transaction_applied,
                dtype=jnp.bool_,
            ),
            reencode_attempted=jnp.asarray(True, dtype=jnp.bool_),
            candidate_values_finite=jnp.asarray(
                candidate_values_finite,
                dtype=jnp.bool_,
            ),
            candidate_controller_state_valid=jnp.asarray(
                candidate_controller_state_valid,
                dtype=jnp.bool_,
            ),
            candidate_state_valid=jnp.asarray(
                candidate_state_valid,
                dtype=jnp.bool_,
            ),
            transaction_applied=jnp.asarray(transaction_applied, dtype=jnp.bool_),
            complete_source_returned=jnp.asarray(
                not transaction_applied,
                dtype=jnp.bool_,
            ),
            valid_rows_before=jnp.asarray(valid_rows_before, dtype=jnp.int32),
            valid_rows_reencoded=jnp.asarray(
                valid_rows_before if transaction_applied else 0,
                dtype=jnp.int32,
            ),
            valid_rows_after=jnp.asarray(valid_rows_after, dtype=jnp.int32),
            context_newborn_slots=jnp.asarray(
                context_newborn_slots,
                dtype=jnp.int32,
            ),
            controller_transaction_words_before=(
                state.controller_state.transaction_words
            ),
            controller_transaction_words_after=(
                selected.controller_state.transaction_words
            ),
            memory_step_words_before=state.controller_state.memory.step_words,
            memory_step_words_after=selected.controller_state.memory.step_words,
        )
        return HCCLFeatureBoundMemoryRebindResult(
            state=selected,
            candidate_state=candidate,
            diagnostics=diagnostics,
            work=_fixed_work(),
        )


__all__ = [
    "HCCL_FEATURE_BOUND_MEMORY_ACTION_DISPATCH_AUTHORITY",
    "HCCL_FEATURE_BOUND_MEMORY_CONFIG_SCHEMA",
    "HCCL_FEATURE_BOUND_MEMORY_COUNTERFACTUAL_FEEDBACK_AUTHENTICATED",
    "HCCL_FEATURE_BOUND_MEMORY_EVIDENCE_AUTHORITY",
    "HCCL_FEATURE_BOUND_MEMORY_FULL_INTEGRATION_CLAIMED",
    "HCCL_FEATURE_BOUND_MEMORY_OUTER_TRANSACTION_AUTHORITY",
    "HCCL_FEATURE_BOUND_MEMORY_SCIENTIFIC_PROMOTION_ALLOWED",
    "HCCL_FEATURE_BOUND_MEMORY_STATE_SCHEMA",
    "HCCL_FEATURE_BOUND_MEMORY_STATUS",
    "HCCLFeatureBoundMemory",
    "HCCLFeatureBoundMemoryConfig",
    "HCCLFeatureBoundMemoryRebindDiagnostics",
    "HCCLFeatureBoundMemoryRebindResult",
    "HCCLFeatureBoundMemoryRebindWork",
    "HCCLFeatureBoundMemorySettleDiagnostics",
    "HCCLFeatureBoundMemorySettleResult",
    "HCCLFeatureBoundMemorySettleWork",
    "HCCLFeatureBoundMemoryState",
    "HCCLFeatureBoundMemoryStepDiagnostics",
    "HCCLFeatureBoundMemoryStepResult",
    "HCCLFeatureBoundMemoryStepWork",
]
