# mypy: disable-error-code="arg-type,attr-defined,call-arg,no-any-return,type-var"
"""L0 whole-agent harness for one semantic owner, envelope, plant, and shadow.

The harness is development-only orchestration. It owns exactly one
``PrototypeEmbodiedCommandAdapterState``, one bounded deterministic synthetic
plant state, and one ``GroundedImaginationCompositionState``. The grounded
composition remains a non-dispatching shadow learner; it never owns or updates
Prototype, semantic memory, the embodied envelope, or the real plant.

``prepare`` stages the adapter command and one grounded-imagination result over
the exact pre-plant observation. It persists only a content tag for that
transient shadow result, not a second shadow state. ``settle`` verifies the
exact envelope result, proposes a plant transition from the envelope-mapped
primitive, then invokes the adapter's post-envelope semantic settlement. An
accepted/fallback transaction then advances the one real semantic owner with
the exact deterministic plant transition and adopts its plant-bound successor,
plant, and one shadow candidate atomically. Exact no-action or stop-only
outcomes adopt only the adapter/envelope state and leave semantic owner, plant,
and shadow bit-exact unchanged.

Plant capacity is a harness scheduling bound, not an inferred environment
termination or truncation. The final admitted transition remains an ordinary
continuing transition and produces one live, plant-bound semantic successor;
subsequent preparation is an exact no-op because no plant slot remains.

The shadow result tag and all checksums are unkeyed accidental-corruption
sentinels. Settlement does not rerun the shadow planner and therefore does not
authenticate a caller capable of rewriting both a result and its persisted
tag. The harness has no physical dispatch, safety, evidence, promotion, or
scientific authority.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any, ClassVar, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core.embodied_safety_envelope import (
    EmbodiedCommand,
    EmbodiedEnvelopeDecision,
)
from alberta_framework.core.grounded_imagination_composition import (
    GroundedImaginationComposition,
    GroundedImaginationCompositionResourceBudget,
    GroundedImaginationCompositionResult,
    GroundedImaginationCompositionState,
)
from alberta_framework.core.prototype_consolidated_semantic_memory import (
    PrototypeConsolidatedSemanticMemoryState,
    PrototypeConsolidatedSemanticTransition,
)
from alberta_framework.core.prototype_embodied_command_adapter import (
    PrototypeEmbodiedCommandAdapter,
    PrototypeEmbodiedCommandAdapterState,
    PrototypeEmbodiedCommandPreparationInput,
    PrototypeEmbodiedCommandPreparationResult,
    PrototypeEmbodiedCommandSettlementResult,
)
from alberta_framework.core.world_model_ensemble import WorldModelEnsembleState

PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_CONFIG_SCHEMA = (
    "alberta.prototype-embodied-development-harness.config.v1"
)
PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_CHECKPOINT_SCHEMA = (
    "alberta.prototype-embodied-development-harness.state.v1"
)
PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_MECHANISM_STATUS = (
    "l0-development-only-not-assessed"
)
PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_CHECKPOINT_HOST_ONLY = True
PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_PREPARE_HOST_ORCHESTRATED = True
PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_GROUNDED_INTERNAL_JIT = True
PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_JIT_SETTLE_SUPPORTED = True
PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_SHADOW_RESULT_AUTHENTICATED = False
PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_PHYSICAL_DISPATCH_AUTHORITY = False
PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_SAFETY_AUTHORITY = False
PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_EVIDENCE_AUTHORITY = False
PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_PROMOTION_AUTHORITY = False
PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_SCIENTIFIC_PROMOTION_ALLOWED = False

_IDENTITY_WORDS = 2
_DECISION_WORDS = 4
_DIGEST_WORDS = 8
_DIGEST_BYTES = 32
_UINT32_MAX = 2**32 - 1
_INT32_MAX = 2**31 - 1


def _require_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must expose array shape and dtype")
    array = cast(Array, value)
    if tuple(array.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}; got {tuple(array.shape)}")
    expected = jnp.dtype(dtype)
    if jnp.dtype(array.dtype) != expected:
        raise TypeError(f"{name} must have dtype {expected}; got {array.dtype}")
    return array


def _float_tuple(value: object, *, name: str, length: int) -> tuple[float, ...]:
    if type(value) is not tuple or len(value) != length:
        raise ValueError(f"{name} must be an exact tuple of length {length}")
    result: list[float] = []
    for index, item in enumerate(value):
        if type(item) is not float or not math.isfinite(item):
            raise ValueError(f"{name}[{index}] must be a finite exact Python float")
        if not math.isfinite(float(np.float32(item))):
            raise ValueError(f"{name}[{index}] must remain finite in float32")
        result.append(item)
    return tuple(result)


def _canonical_digest(value: object) -> Array:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    raw = hashlib.sha256(encoded.encode("utf-8")).digest()
    return jnp.asarray(
        tuple(
            int.from_bytes(raw[offset : offset + 4], "little")
            for offset in range(0, _DIGEST_BYTES, 4)
        ),
        dtype=jnp.uint32,
    )


def _tree_words(value: object) -> Array:
    parts: list[Array] = []
    for leaf in jax.tree_util.tree_leaves(value):
        if leaf is None:
            continue
        array = jnp.asarray(leaf)
        if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
            words = jr.key_data(array)
        elif array.dtype in (jnp.float32, jnp.int32):
            words = jax.lax.bitcast_convert_type(array, jnp.uint32)
        elif array.dtype == jnp.uint32:
            words = array
        else:
            words = array.astype(jnp.uint32)
        parts.append(words.reshape((-1,)))
    if not parts:
        return jnp.zeros((0,), dtype=jnp.uint32)
    return jnp.concatenate(tuple(parts))


def _content_tag(value: object, *, salt: int) -> Array:
    words = _tree_words(value)

    def body(index: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        acc0, acc1 = carry
        position = jnp.asarray(index + 1, dtype=jnp.uint32)
        word = words[index]
        acc0 = (acc0 ^ word ^ (position * jnp.uint32(0x9E3779B9))) * jnp.uint32(
            0x01000193
        )
        acc1 = acc1 + (word ^ (position * jnp.uint32(0x85EBCA6B)))
        acc1 = (acc1 << jnp.uint32(13)) | (acc1 >> jnp.uint32(19))
        return acc0, acc1

    initial = (
        jnp.uint32(0x811C9DC5 ^ (salt & _UINT32_MAX)),
        jnp.uint32(0x9E3779B9 ^ ((salt * 17) & _UINT32_MAX)),
    )
    first, second = jax.lax.fori_loop(0, words.shape[0], body, initial)
    return jnp.stack((first, second), dtype=jnp.uint32)


def _tree_array_equal(left: object, right: object) -> Array:
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    if cast(object, left_tree) != cast(object, right_tree):
        return jnp.asarray(False, dtype=jnp.bool_)
    if len(left_leaves) != len(right_leaves):
        return jnp.asarray(False, dtype=jnp.bool_)
    valid = jnp.asarray(True, dtype=jnp.bool_)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = jnp.asarray(left_leaf)
        right_array = jnp.asarray(right_leaf)
        if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
            return jnp.asarray(False, dtype=jnp.bool_)
        if jax.dtypes.issubdtype(left_array.dtype, jax.dtypes.prng_key):
            valid = valid & jnp.array_equal(
                jr.key_data(left_array),
                jr.key_data(right_array),
            )
        elif left_array.dtype == jnp.float32:
            valid = valid & jnp.array_equal(
                jax.lax.bitcast_convert_type(left_array, jnp.uint32),
                jax.lax.bitcast_convert_type(right_array, jnp.uint32),
            )
        else:
            valid = valid & jnp.array_equal(left_array, right_array)
    return valid


def _tree_nbytes(value: object) -> int:
    total = 0
    for leaf in jax.tree_util.tree_leaves(value):
        array = jnp.asarray(leaf)
        if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
            array = jr.key_data(array)
        total += int(array.size) * int(array.dtype.itemsize)
    return total


def _tree_sha256(value: object) -> Array:
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(value):
        array = jnp.asarray(leaf)
        if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
            array = jr.key_data(array)
        host = np.asarray(jax.device_get(array))
        digest.update(host.dtype.str.encode("ascii"))
        digest.update(np.asarray(host.shape, dtype=np.int64).tobytes())
        digest.update(host.tobytes(order="C"))
    return jnp.asarray(tuple(digest.digest()), dtype=jnp.uint8)


def _increment_decision_words(words: Array) -> tuple[Array, Array]:
    """Increment one big-endian uint32 word vector without wrapping."""

    result = words
    carry = jnp.asarray(True, dtype=jnp.bool_)
    for index in range(words.shape[0] - 1, -1, -1):
        value = result[index] + carry.astype(jnp.uint32)
        result = result.at[index].set(value)
        carry = carry & (value == jnp.uint32(0))
    available = ~jnp.all(words == jnp.uint32(_UINT32_MAX))
    return jnp.where(available, result, words), available


@dataclasses.dataclass(frozen=True, slots=True)
class DeterministicPrimitivePlantConfig:
    """Bounded synthetic action deltas; no physical or geometry claim."""

    observation_lower: tuple[float, ...]
    observation_upper: tuple[float, ...]
    primitive_deltas: tuple[tuple[float, ...], ...]
    primitive_rewards: tuple[float, ...]
    max_transitions: int

    SCHEMA_VERSION: ClassVar[str] = (
        "alberta.deterministic-primitive-plant.config.v1"
    )

    def __post_init__(self) -> None:
        if type(self.observation_lower) is not tuple or len(self.observation_lower) < 1:
            raise ValueError("observation_lower must be one nonempty exact tuple")
        dimension = len(self.observation_lower)
        lower = _float_tuple(
            self.observation_lower,
            name="observation_lower",
            length=dimension,
        )
        upper = _float_tuple(
            self.observation_upper,
            name="observation_upper",
            length=dimension,
        )
        if any(lo >= hi for lo, hi in zip(lower, upper, strict=True)):
            raise ValueError("every plant lower bound must be below its upper bound")
        if type(self.primitive_deltas) is not tuple or len(self.primitive_deltas) < 1:
            raise ValueError("primitive_deltas must be one nonempty exact tuple")
        for index, delta in enumerate(self.primitive_deltas):
            _float_tuple(delta, name=f"primitive_deltas[{index}]", length=dimension)
        _float_tuple(
            self.primitive_rewards,
            name="primitive_rewards",
            length=len(self.primitive_deltas),
        )
        if (
            type(self.max_transitions) is not int
            or self.max_transitions < 1
            or self.max_transitions > _INT32_MAX
        ):
            raise ValueError("max_transitions must be a positive exact int32 Python int")

    @property
    def observation_dim(self) -> int:
        return len(self.observation_lower)

    @property
    def n_actions(self) -> int:
        return len(self.primitive_deltas)

    def to_config(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA_VERSION,
            "observation_lower": list(self.observation_lower),
            "observation_upper": list(self.observation_upper),
            "primitive_deltas": [list(row) for row in self.primitive_deltas],
            "primitive_rewards": list(self.primitive_rewards),
            "max_transitions": self.max_transitions,
            "deterministic": True,
            "synthetic": True,
            "transition_discount": 1.0,
            "terminated_always_false": True,
            "truncated_always_false": True,
            "capacity_exhaustion": "scheduling_halt_no_new_receipt",
            "physical_dispatch_authority": False,
            "geometry_claim": False,
            "rng_draws_per_step": 0,
        }

    @classmethod
    def from_config(cls, value: Mapping[str, object]) -> DeterministicPrimitivePlantConfig:
        if type(value) is not dict:
            raise ValueError("plant config must be an exact dict")
        raw = dict(value)
        fixed: dict[str, object] = {
            "schema": cls.SCHEMA_VERSION,
            "deterministic": True,
            "synthetic": True,
            "transition_discount": 1.0,
            "terminated_always_false": True,
            "truncated_always_false": True,
            "capacity_exhaustion": "scheduling_halt_no_new_receipt",
            "physical_dispatch_authority": False,
            "geometry_claim": False,
            "rng_draws_per_step": 0,
        }
        expected = {
            "observation_lower",
            "observation_upper",
            "primitive_deltas",
            "primitive_rewards",
            "max_transitions",
            *fixed,
        }
        if set(raw) != expected:
            raise ValueError("plant config fields differ from schema v1")
        for name, expected_value in fixed.items():
            if type(raw[name]) is not type(expected_value) or raw.pop(name) != expected_value:
                raise ValueError(f"plant config fixed field {name} differs")

        def row(name: str) -> tuple[float, ...]:
            item = raw[name]
            if type(item) is not list or any(type(value) is not float for value in item):
                raise ValueError(f"plant {name} must be an exact float list")
            return tuple(cast(list[float], item))

        deltas_raw = raw["primitive_deltas"]
        if type(deltas_raw) is not list:
            raise ValueError("plant primitive_deltas must be an exact list")
        deltas: list[tuple[float, ...]] = []
        for item in deltas_raw:
            if type(item) is not list or any(type(value) is not float for value in item):
                raise ValueError("each plant primitive delta must be an exact float list")
            deltas.append(tuple(cast(list[float], item)))
        maximum = raw["max_transitions"]
        if type(maximum) is not int:
            raise ValueError("plant max_transitions must be an exact int")
        result = cls(
            observation_lower=row("observation_lower"),
            observation_upper=row("observation_upper"),
            primitive_deltas=tuple(deltas),
            primitive_rewards=row("primitive_rewards"),
            max_transitions=maximum,
        )
        if result.to_config() != value:
            raise ValueError("plant config is noncanonical")
        return result


@chex.dataclass(frozen=True)
class DeterministicPrimitivePlantState:
    observation: Float[Array, " observation_dim"]
    transition_count: Int[Array, ""]
    has_transition: Bool[Array, ""]
    last_receipt_words: UInt[Array, " 2"]
    last_action: Int[Array, ""]
    last_pre_observation: Float[Array, " observation_dim"]
    last_post_observation: Float[Array, " observation_dim"]
    checksum: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class DeterministicPrimitivePlantTransition:
    requested: Bool[Array, ""]
    proposal_applied: Bool[Array, ""]
    committed: Bool[Array, ""]
    receipt_words: UInt[Array, " 2"]
    primitive_action: Int[Array, ""]
    pre_observation: Float[Array, " observation_dim"]
    post_observation: Float[Array, " observation_dim"]
    reward: Float[Array, ""]
    discount: Float[Array, ""]
    terminated: Bool[Array, ""]
    truncated: Bool[Array, ""]
    clipped: Bool[Array, ""]
    pre_transition_count: Int[Array, ""]
    post_transition_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class DeterministicPrimitivePlantStepResult:
    state: DeterministicPrimitivePlantState
    transition: DeterministicPrimitivePlantTransition
    source_state_valid: Bool[Array, ""]
    action_contract_valid: Bool[Array, ""]
    capacity_available: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]


class DeterministicPrimitivePlant:
    """Pure bounded synthetic transition kernel."""

    def __init__(self, config: DeterministicPrimitivePlantConfig) -> None:
        if type(config) is not DeterministicPrimitivePlantConfig:
            raise TypeError("config must be an exact DeterministicPrimitivePlantConfig")
        self._config = config
        self._lower = jnp.asarray(config.observation_lower, dtype=jnp.float32)
        self._upper = jnp.asarray(config.observation_upper, dtype=jnp.float32)
        self._deltas = jnp.asarray(config.primitive_deltas, dtype=jnp.float32)
        self._rewards = jnp.asarray(config.primitive_rewards, dtype=jnp.float32)

    @property
    def config(self) -> DeterministicPrimitivePlantConfig:
        return self._config

    def to_config(self) -> dict[str, object]:
        return self._config.to_config()

    def _payload(self, state: DeterministicPrimitivePlantState) -> object:
        return (
            state.observation,
            state.transition_count,
            state.has_transition,
            state.last_receipt_words,
            state.last_action,
            state.last_pre_observation,
            state.last_post_observation,
        )

    def _with_checksum(
        self,
        state: DeterministicPrimitivePlantState,
    ) -> DeterministicPrimitivePlantState:
        return state.replace(checksum=_content_tag(self._payload(state), salt=0x504C414E))

    def _check_state_contract(self, state: DeterministicPrimitivePlantState) -> None:
        if type(state) is not DeterministicPrimitivePlantState:
            raise TypeError("plant state has the wrong exact type")
        dimension = self._config.observation_dim
        contracts = {
            "observation": ((dimension,), jnp.float32),
            "transition_count": ((), jnp.int32),
            "has_transition": ((), jnp.bool_),
            "last_receipt_words": ((_IDENTITY_WORDS,), jnp.uint32),
            "last_action": ((), jnp.int32),
            "last_pre_observation": ((dimension,), jnp.float32),
            "last_post_observation": ((dimension,), jnp.float32),
            "checksum": ((_IDENTITY_WORDS,), jnp.uint32),
        }
        for name, (shape, dtype) in contracts.items():
            _require_array(
                getattr(state, name),
                name=f"plant.{name}",
                shape=shape,
                dtype=dtype,
            )

    def state_valid(self, state: DeterministicPrimitivePlantState) -> Array:
        self._check_state_contract(state)
        observation_valid = (
            jnp.all(jnp.isfinite(state.observation))
            & jnp.all(state.observation >= self._lower)
            & jnp.all(state.observation <= self._upper)
        )
        count_valid = (
            (state.transition_count >= 0)
            & (state.transition_count <= self._config.max_transitions)
        )
        history_valid = jnp.where(
            state.has_transition,
            _tree_array_equal(state.observation, state.last_post_observation)
            & jnp.any(state.last_receipt_words != jnp.uint32(0))
            & (state.last_action >= 0)
            & (state.last_action < self._config.n_actions)
            & jnp.all(jnp.isfinite(state.last_pre_observation))
            & jnp.all(jnp.isfinite(state.last_post_observation))
            & (state.transition_count > 0),
            jnp.all(state.last_receipt_words == jnp.uint32(0))
            & (state.last_action == -1)
            & jnp.array_equal(state.last_pre_observation, state.observation)
            & jnp.array_equal(state.last_post_observation, state.observation)
            & (state.transition_count == 0),
        )
        return (
            observation_valid
            & count_valid
            & history_valid
            & jnp.array_equal(
                state.checksum,
                _content_tag(self._payload(state), salt=0x504C414E),
            )
        )

    def init(self, observation: Array) -> DeterministicPrimitivePlantState:
        exact = _require_array(
            observation,
            name="plant initial observation",
            shape=(self._config.observation_dim,),
            dtype=jnp.float32,
        )
        state = DeterministicPrimitivePlantState(
            observation=exact,
            transition_count=jnp.asarray(0, dtype=jnp.int32),
            has_transition=jnp.asarray(False, dtype=jnp.bool_),
            last_receipt_words=jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32),
            last_action=jnp.asarray(-1, dtype=jnp.int32),
            last_pre_observation=exact,
            last_post_observation=exact,
            checksum=jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32),
        )
        state = self._with_checksum(state)
        if not bool(jax.device_get(self.state_valid(state))):
            raise ValueError("plant initial observation is outside its finite bounds")
        return state

    def propose_step(
        self,
        state: DeterministicPrimitivePlantState,
        *,
        requested: Array,
        receipt_words: Array,
        primitive_action: Array,
    ) -> DeterministicPrimitivePlantStepResult:
        self._check_state_contract(state)
        _require_array(requested, name="plant requested", shape=(), dtype=jnp.bool_)
        _require_array(
            receipt_words,
            name="plant receipt_words",
            shape=(_IDENTITY_WORDS,),
            dtype=jnp.uint32,
        )
        _require_array(
            primitive_action,
            name="plant primitive_action",
            shape=(),
            dtype=jnp.int32,
        )
        source_valid = self.state_valid(state)
        action_valid = (
            (primitive_action >= 0)
            & (primitive_action < self._config.n_actions)
        )
        safe_action = jnp.clip(primitive_action, 0, self._config.n_actions - 1)
        capacity = state.transition_count < self._config.max_transitions
        receipt_valid = jnp.any(receipt_words != jnp.uint32(0))
        raw_next = state.observation + self._deltas[safe_action]
        next_observation = jnp.clip(raw_next, self._lower, self._upper)
        clipped = jnp.any(
            jax.lax.bitcast_convert_type(raw_next, jnp.uint32)
            != jax.lax.bitcast_convert_type(next_observation, jnp.uint32)
        )
        apply_pre = requested & source_valid & action_valid & capacity & receipt_valid
        candidate = self._with_checksum(
            state.replace(
                observation=next_observation,
                transition_count=state.transition_count + jnp.int32(1),
                has_transition=jnp.asarray(True, dtype=jnp.bool_),
                last_receipt_words=receipt_words,
                last_action=primitive_action,
                last_pre_observation=state.observation,
                last_post_observation=next_observation,
                checksum=jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32),
            )
        )
        candidate_valid = self.state_valid(candidate)
        applied = apply_pre & candidate_valid
        next_state = cast(
            DeterministicPrimitivePlantState,
            jax.lax.cond(applied, lambda _: candidate, lambda _: state, operand=None),
        )
        transition = DeterministicPrimitivePlantTransition(
            requested=requested,
            proposal_applied=applied,
            committed=jnp.asarray(False, dtype=jnp.bool_),
            receipt_words=jnp.where(
                applied,
                receipt_words,
                jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32),
            ),
            primitive_action=jnp.where(
                applied,
                primitive_action,
                jnp.asarray(-1, dtype=jnp.int32),
            ).astype(jnp.int32),
            pre_observation=state.observation,
            post_observation=jnp.where(
                applied,
                next_observation,
                state.observation,
            ),
            reward=jnp.where(
                applied,
                self._rewards[safe_action],
                jnp.asarray(0.0, dtype=jnp.float32),
            ).astype(jnp.float32),
            discount=jnp.where(
                applied,
                jnp.asarray(1.0, dtype=jnp.float32),
                jnp.asarray(0.0, dtype=jnp.float32),
            ).astype(jnp.float32),
            terminated=jnp.asarray(False, dtype=jnp.bool_),
            truncated=jnp.asarray(False, dtype=jnp.bool_),
            clipped=applied & clipped,
            pre_transition_count=state.transition_count,
            post_transition_count=jnp.where(
                applied,
                state.transition_count + jnp.int32(1),
                state.transition_count,
            ).astype(jnp.int32),
        )
        return DeterministicPrimitivePlantStepResult(
            state=next_state,
            transition=transition,
            source_state_valid=source_valid,
            action_contract_valid=action_valid,
            capacity_available=capacity,
            candidate_state_valid=candidate_valid,
        )


@chex.dataclass(frozen=True)
class PrototypeEmbodiedDevelopmentHarnessPreparationInput:
    envelope: PrototypeEmbodiedCommandPreparationInput
    model_state: WorldModelEnsembleState
    action_support_counts: Int[Array, " n_actions"]
    source_revision_words: UInt[Array, " 2"]
    region_ids: Int[Array, "rollout_budget rollout_horizon"]
    safety_admitted: Bool[Array, "rollout_budget rollout_horizon"]
    protected: Bool[Array, "rollout_budget rollout_horizon"]


@chex.dataclass(frozen=True)
class PrototypeEmbodiedDevelopmentHarnessPendingState:
    available: Bool[Array, ""]
    receipt_words: UInt[Array, " 2"]
    prototype_decision_id: UInt[Array, " 4"]
    selected_action: Int[Array, ""]
    adapter_binding_checksum: UInt[Array, " 2"]
    adapter_state_content_tag: UInt[Array, " 2"]
    adapter_pending_checksum: UInt[Array, " 2"]
    plant_source_checksum: UInt[Array, " 2"]
    shadow_source_integrity_tag: UInt[Array, ""]
    shadow_source_transaction_words: UInt[Array, " 2"]
    proposed_command: EmbodiedCommand
    shadow_result_content_tag: UInt[Array, " 2"]
    harness_config_digest: UInt[Array, " 8"]
    checksum: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class PrototypeEmbodiedDevelopmentHarnessCommitRecord:
    available: Bool[Array, ""]
    receipt_words: UInt[Array, " 2"]
    prototype_decision_id: UInt[Array, " 4"]
    selected_action: Int[Array, ""]
    executed_action: Int[Array, ""]
    successor_available: Bool[Array, ""]
    successor_decision_id: UInt[Array, " 4"]
    successor_action: Int[Array, ""]
    semantic_successor_content_tag: UInt[Array, " 2"]
    proposed_command: EmbodiedCommand
    executed_command: EmbodiedCommand
    envelope_result_content_tag: UInt[Array, " 2"]
    envelope_decision_id: UInt[Array, " 2"]
    envelope_action_id: UInt[Array, " 2"]
    telemetry_id: UInt[Array, " 2"]
    model_version: UInt[Array, " 8"]
    optimizer_version: UInt[Array, " 8"]
    lifecycle_version: UInt[Array, " 8"]
    plant_transition: DeterministicPrimitivePlantTransition
    shadow_source_integrity_tag: UInt[Array, ""]
    shadow_result_content_tag: UInt[Array, " 2"]
    shadow_post_integrity_tag: UInt[Array, ""]
    checksum: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class PrototypeEmbodiedDevelopmentHarnessState:
    adapter: PrototypeEmbodiedCommandAdapterState
    plant: DeterministicPrimitivePlantState
    shadow: GroundedImaginationCompositionState
    harness_config_digest: UInt[Array, " 8"]
    pending: PrototypeEmbodiedDevelopmentHarnessPendingState
    last_commit: PrototypeEmbodiedDevelopmentHarnessCommitRecord
    binding_checksum: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class PrototypeEmbodiedDevelopmentHarnessPreparationDiagnostics:
    source_state_valid: Bool[Array, ""]
    receipt_slot_available: Bool[Array, ""]
    plant_capacity_available: Bool[Array, ""]
    adapter_prepared: Bool[Array, ""]
    shadow_source_bound: Bool[Array, ""]
    shadow_result_state_valid: Bool[Array, ""]
    shadow_result_content_bound: Bool[Array, ""]
    shadow_result_authenticated: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    prepared: Bool[Array, ""]
    physical_dispatch_authority: Bool[Array, ""]
    safety_authority: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeEmbodiedDevelopmentHarnessPreparationResult:
    state: PrototypeEmbodiedDevelopmentHarnessState
    command: EmbodiedCommand
    receipt_words: UInt[Array, " 2"]
    adapter: PrototypeEmbodiedCommandPreparationResult
    shadow: GroundedImaginationCompositionResult
    diagnostics: PrototypeEmbodiedDevelopmentHarnessPreparationDiagnostics


@chex.dataclass(frozen=True)
class PrototypeEmbodiedDevelopmentHarnessSettlementDiagnostics:
    source_state_valid: Bool[Array, ""]
    pending_receipt_available: Bool[Array, ""]
    envelope_result_exact: Bool[Array, ""]
    mapped_action_unique: Bool[Array, ""]
    mapped_action_admitted: Bool[Array, ""]
    plant_proposal_requested: Bool[Array, ""]
    plant_proposal_applied: Bool[Array, ""]
    plant_transition_committed: Bool[Array, ""]
    semantic_settlement_after_plant_proposal: Bool[Array, ""]
    semantic_transition_requested: Bool[Array, ""]
    semantic_transition_committed: Bool[Array, ""]
    semantic_prototype_learning_retained: Bool[Array, ""]
    semantic_successor_plant_bound: Bool[Array, ""]
    semantic_successor_rearmed: Bool[Array, ""]
    plant_capacity_exhausted_after_commit: Bool[Array, ""]
    adapter_action_receipt_consumed: Bool[Array, ""]
    adapter_envelope_only_committed: Bool[Array, ""]
    shadow_result_content_matches_receipt: Bool[Array, ""]
    shadow_result_source_bound: Bool[Array, ""]
    shadow_result_state_valid: Bool[Array, ""]
    shadow_result_authenticated: Bool[Array, ""]
    shadow_adopted: Bool[Array, ""]
    no_action_plant_unchanged: Bool[Array, ""]
    no_action_shadow_unchanged: Bool[Array, ""]
    emergency_stop_latch_preserved: Bool[Array, ""]
    action_transaction_committed: Bool[Array, ""]
    envelope_only_transaction_committed: Bool[Array, ""]
    transaction_committed: Bool[Array, ""]
    learning_updates_applied_by_real_adapter: Int[Array, ""]
    prototype_learning_updates_adopted: Int[Array, ""]
    shadow_backward_evaluations_per_prepare: Int[Array, ""]
    physical_dispatch_authority: Bool[Array, ""]
    safety_authority: Bool[Array, ""]
    evidence_authority: Bool[Array, ""]
    promotion_authority: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeEmbodiedDevelopmentHarnessSettlementResult:
    state: PrototypeEmbodiedDevelopmentHarnessState
    action: Int[Array, ""]
    transition: DeterministicPrimitivePlantTransition
    adapter: PrototypeEmbodiedCommandSettlementResult
    envelope: EmbodiedEnvelopeDecision
    shadow: GroundedImaginationCompositionResult
    diagnostics: PrototypeEmbodiedDevelopmentHarnessSettlementDiagnostics


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeEmbodiedDevelopmentHarnessResourceBudget:
    persistent_state_nbytes: int
    adapter_state_nbytes: int
    plant_state_nbytes: int
    shadow_state_nbytes: int
    pending_receipts: int
    last_commit_records: int
    plant_observation_dim: int
    primitive_actions: int
    maximum_plant_transitions: int
    plant_capacity_halts_prepare: bool
    budget_exhaustion_environment_boundaries_inferred: int
    logical_adapter_prepare_calls_per_prepare: int
    logical_shadow_steps_per_prepare: int
    logical_adapter_settlements_per_settle: int
    logical_plant_proposals_per_settle: int
    maximum_semantic_transition_calls_per_settle: int
    maximum_prototype_learning_updates_per_settle: int
    no_action_semantic_transition_calls_per_settle: int
    maximum_plant_transitions_per_settle: int
    maximum_shadow_planner_calls_per_prepare: int
    maximum_shadow_authorization_calls_per_prepare: int
    maximum_shadow_actor_critic_commits_per_prepare: int
    maximum_shadow_backward_transitions_per_prepare: int
    maximum_shadow_autodiff_passes_per_prepare: int
    maximum_shadow_rng_splits_per_prepare: int
    maximum_shadow_rng_draws_per_prepare: int
    shadow_recomputations_per_settle: int
    physical_dispatches_per_operation: int
    real_adapter_learning_updates_per_settle: int
    evidence_writes_per_operation: int
    persistent_growth_per_operation_bytes: int
    checkpoint_host_only: bool
    prepare_host_orchestrated: bool
    grounded_internal_jit: bool
    jit_settle_supported: bool
    shadow_result_authenticated: bool
    shadow_content_integrity_only: bool
    shadow_dispatch_authority: bool
    plant_physical_dispatch_authority: bool
    safety_authority: bool
    evidence_authority: bool
    promotion_authority: bool
    scientific_promotion_allowed: bool
    grounded: GroundedImaginationCompositionResourceBudget


class PrototypeEmbodiedDevelopmentHarness:
    """One-owner, two-phase development composition over real and shadow state."""

    def __init__(
        self,
        adapter: PrototypeEmbodiedCommandAdapter,
        plant: DeterministicPrimitivePlant,
        grounded: GroundedImaginationComposition,
    ) -> None:
        if type(adapter) is not PrototypeEmbodiedCommandAdapter:
            raise TypeError("adapter must be an exact PrototypeEmbodiedCommandAdapter")
        if type(plant) is not DeterministicPrimitivePlant:
            raise TypeError("plant must be an exact DeterministicPrimitivePlant")
        if type(grounded) is not GroundedImaginationComposition:
            raise TypeError("grounded must be an exact GroundedImaginationComposition")
        n_actions = adapter.config.n_actions
        observation_dim = adapter.config.semantic.raw_observation_dim
        if plant.config.n_actions != n_actions:
            raise ValueError("plant n_actions must equal the adapter command bank")
        if grounded.planner.n_actions != n_actions:
            raise ValueError("grounded planner n_actions must equal the adapter command bank")
        if plant.config.observation_dim != observation_dim:
            raise ValueError("plant observation_dim must equal semantic raw_observation_dim")
        if grounded.planner.observation_dim != observation_dim:
            raise ValueError("grounded observation_dim must equal semantic raw_observation_dim")
        self._adapter = adapter
        self._plant = plant
        self._grounded = grounded
        self._config_digest = _canonical_digest(self.to_config())

    @property
    def adapter(self) -> PrototypeEmbodiedCommandAdapter:
        return self._adapter

    @property
    def plant(self) -> DeterministicPrimitivePlant:
        return self._plant

    @property
    def grounded(self) -> GroundedImaginationComposition:
        return self._grounded

    @property
    def config_digest(self) -> Array:
        return self._config_digest

    def to_config(self) -> dict[str, object]:
        return {
            "schema": PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_CONFIG_SCHEMA,
            "mechanism_status": (
                PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_MECHANISM_STATUS
            ),
            "adapter": self._adapter.to_config(),
            "plant": self._plant.to_config(),
            "grounded": self._grounded.to_config(),
            "owned_adapter_states": 1,
            "owned_plant_states": 1,
            "owned_grounded_composition_states": 1,
            "owned_prototype_states_outside_adapter": 0,
            "owned_semantic_states_outside_adapter": 0,
            "owned_envelope_states_outside_adapter": 0,
            "shadow_candidate_state_persisted_while_pending": False,
            "shadow_result_binding": "unkeyed_content_tag_from_prepare",
            "shadow_result_authenticated": False,
            "shadow_recomputed_during_settle": False,
            "real_successor_transition": "one_exact_plant_transition",
            "successor_semantic_input": "none_zero_semantic_tail",
            "successor_decision_input": "none_base_prototype_dispatch_owner",
            "synthetic_reward_source": "plant_primitive_reward",
            "plant_capacity_exhaustion": "halt_prepare_with_live_successor",
            "budget_exhaustion_implies_environment_boundary": False,
            "composition_order": [
                "semantic_dispatch_owner",
                "adapter_prepare",
                "grounded_shadow_over_preplant_source",
                "real_envelope_result",
                "envelope_mapped_plant_proposal",
                "post_plant_adapter_semantic_settlement",
                "post_settlement_real_semantic_transition",
                "plant_bound_successor_dispatch_owner",
                "atomic_adoption",
            ],
            "no_action_semantics": (
                "adapter_envelope_only_commit_plant_and_shadow_unchanged"
            ),
            "physical_dispatch_authority": False,
            "safety_authority": False,
            "evidence_authority": False,
            "promotion_authority": False,
            "scientific_promotion_allowed": False,
            "prepare_host_orchestrated": True,
            "grounded_internal_jit": True,
            "jit_settle_supported": True,
        }

    @classmethod
    def from_config(
        cls,
        value: Mapping[str, object],
    ) -> PrototypeEmbodiedDevelopmentHarness:
        if type(value) is not dict:
            raise ValueError("development harness config must be an exact dict")
        raw = dict(value)
        fixed: dict[str, object] = {
            "schema": PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_CONFIG_SCHEMA,
            "mechanism_status": (
                PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_MECHANISM_STATUS
            ),
            "owned_adapter_states": 1,
            "owned_plant_states": 1,
            "owned_grounded_composition_states": 1,
            "owned_prototype_states_outside_adapter": 0,
            "owned_semantic_states_outside_adapter": 0,
            "owned_envelope_states_outside_adapter": 0,
            "shadow_candidate_state_persisted_while_pending": False,
            "shadow_result_binding": "unkeyed_content_tag_from_prepare",
            "shadow_result_authenticated": False,
            "shadow_recomputed_during_settle": False,
            "real_successor_transition": "one_exact_plant_transition",
            "successor_semantic_input": "none_zero_semantic_tail",
            "successor_decision_input": "none_base_prototype_dispatch_owner",
            "synthetic_reward_source": "plant_primitive_reward",
            "plant_capacity_exhaustion": "halt_prepare_with_live_successor",
            "budget_exhaustion_implies_environment_boundary": False,
            "composition_order": [
                "semantic_dispatch_owner",
                "adapter_prepare",
                "grounded_shadow_over_preplant_source",
                "real_envelope_result",
                "envelope_mapped_plant_proposal",
                "post_plant_adapter_semantic_settlement",
                "post_settlement_real_semantic_transition",
                "plant_bound_successor_dispatch_owner",
                "atomic_adoption",
            ],
            "no_action_semantics": (
                "adapter_envelope_only_commit_plant_and_shadow_unchanged"
            ),
            "physical_dispatch_authority": False,
            "safety_authority": False,
            "evidence_authority": False,
            "promotion_authority": False,
            "scientific_promotion_allowed": False,
            "prepare_host_orchestrated": True,
            "grounded_internal_jit": True,
            "jit_settle_supported": True,
        }
        expected = {"adapter", "plant", "grounded", *fixed}
        if set(raw) != expected:
            raise ValueError("development harness config fields differ from schema v1")
        for name, expected_value in fixed.items():
            if type(raw[name]) is not type(expected_value) or raw.pop(name) != expected_value:
                raise ValueError(f"development harness fixed field {name} differs")
        adapter_raw = raw.pop("adapter")
        plant_raw = raw.pop("plant")
        grounded_raw = raw.pop("grounded")
        if not all(type(item) is dict for item in (adapter_raw, plant_raw, grounded_raw)):
            raise ValueError("development harness nested configs must be exact dicts")
        result = cls(
            PrototypeEmbodiedCommandAdapter.from_config(
                cast(Mapping[str, object], adapter_raw)
            ),
            DeterministicPrimitivePlant(
                DeterministicPrimitivePlantConfig.from_config(
                    cast(Mapping[str, object], plant_raw)
                )
            ),
            GroundedImaginationComposition.from_config(
                cast(Mapping[str, object], grounded_raw)
            ),
        )
        if result.to_config() != value:
            raise ValueError("development harness config is noncanonical")
        return result

    def _zero_command(self) -> EmbodiedCommand:
        n = self._adapter.config.n_joints
        return EmbodiedCommand(
            joint_position=jnp.zeros((n,), dtype=jnp.float32),
            joint_velocity=jnp.zeros((n,), dtype=jnp.float32),
            joint_torque=jnp.zeros((n,), dtype=jnp.float32),
            workspace_position=jnp.zeros((3,), dtype=jnp.float32),
            collision_clearance=jnp.asarray(0.0, dtype=jnp.float32),
        )

    def _blank_transition(self) -> DeterministicPrimitivePlantTransition:
        zero_obs = jnp.zeros(
            (self._plant.config.observation_dim,), dtype=jnp.float32
        )
        return DeterministicPrimitivePlantTransition(
            requested=jnp.asarray(False, dtype=jnp.bool_),
            proposal_applied=jnp.asarray(False, dtype=jnp.bool_),
            committed=jnp.asarray(False, dtype=jnp.bool_),
            receipt_words=jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32),
            primitive_action=jnp.asarray(-1, dtype=jnp.int32),
            pre_observation=zero_obs,
            post_observation=zero_obs,
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            discount=jnp.asarray(0.0, dtype=jnp.float32),
            terminated=jnp.asarray(False, dtype=jnp.bool_),
            truncated=jnp.asarray(False, dtype=jnp.bool_),
            clipped=jnp.asarray(False, dtype=jnp.bool_),
            pre_transition_count=jnp.asarray(-1, dtype=jnp.int32),
            post_transition_count=jnp.asarray(-1, dtype=jnp.int32),
        )

    def _pending_payload(
        self,
        pending: PrototypeEmbodiedDevelopmentHarnessPendingState,
    ) -> object:
        return tuple(
            getattr(pending, field.name)
            for field in dataclasses.fields(PrototypeEmbodiedDevelopmentHarnessPendingState)
            if field.name != "checksum"
        )

    def _with_pending_checksum(
        self,
        pending: PrototypeEmbodiedDevelopmentHarnessPendingState,
    ) -> PrototypeEmbodiedDevelopmentHarnessPendingState:
        return pending.replace(
            checksum=_content_tag(self._pending_payload(pending), salt=0x48525045)
        )

    def _blank_pending(self) -> PrototypeEmbodiedDevelopmentHarnessPendingState:
        pending = PrototypeEmbodiedDevelopmentHarnessPendingState(
            available=jnp.asarray(False, dtype=jnp.bool_),
            receipt_words=jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32),
            prototype_decision_id=jnp.zeros((_DECISION_WORDS,), dtype=jnp.uint32),
            selected_action=jnp.asarray(-1, dtype=jnp.int32),
            adapter_binding_checksum=jnp.zeros(
                (_IDENTITY_WORDS,), dtype=jnp.uint32
            ),
            adapter_state_content_tag=jnp.zeros(
                (_IDENTITY_WORDS,), dtype=jnp.uint32
            ),
            adapter_pending_checksum=jnp.zeros(
                (_IDENTITY_WORDS,), dtype=jnp.uint32
            ),
            plant_source_checksum=jnp.zeros(
                (_IDENTITY_WORDS,), dtype=jnp.uint32
            ),
            shadow_source_integrity_tag=jnp.asarray(0, dtype=jnp.uint32),
            shadow_source_transaction_words=jnp.zeros(
                (_IDENTITY_WORDS,), dtype=jnp.uint32
            ),
            proposed_command=self._zero_command(),
            shadow_result_content_tag=jnp.zeros(
                (_IDENTITY_WORDS,), dtype=jnp.uint32
            ),
            harness_config_digest=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32),
            checksum=jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32),
        )
        return self._with_pending_checksum(pending)

    def _commit_payload(
        self,
        record: PrototypeEmbodiedDevelopmentHarnessCommitRecord,
    ) -> object:
        return tuple(
            getattr(record, field.name)
            for field in dataclasses.fields(PrototypeEmbodiedDevelopmentHarnessCommitRecord)
            if field.name != "checksum"
        )

    def _with_commit_checksum(
        self,
        record: PrototypeEmbodiedDevelopmentHarnessCommitRecord,
    ) -> PrototypeEmbodiedDevelopmentHarnessCommitRecord:
        return record.replace(
            checksum=_content_tag(self._commit_payload(record), salt=0x4852434D)
        )

    def _blank_commit(self) -> PrototypeEmbodiedDevelopmentHarnessCommitRecord:
        zero2 = jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32)
        zero8 = jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32)
        record = PrototypeEmbodiedDevelopmentHarnessCommitRecord(
            available=jnp.asarray(False, dtype=jnp.bool_),
            receipt_words=zero2,
            prototype_decision_id=jnp.zeros((_DECISION_WORDS,), dtype=jnp.uint32),
            selected_action=jnp.asarray(-1, dtype=jnp.int32),
            executed_action=jnp.asarray(-1, dtype=jnp.int32),
            successor_available=jnp.asarray(False, dtype=jnp.bool_),
            successor_decision_id=jnp.zeros(
                (_DECISION_WORDS,), dtype=jnp.uint32
            ),
            successor_action=jnp.asarray(-1, dtype=jnp.int32),
            semantic_successor_content_tag=zero2,
            proposed_command=self._zero_command(),
            executed_command=self._zero_command(),
            envelope_result_content_tag=zero2,
            envelope_decision_id=zero2,
            envelope_action_id=zero2,
            telemetry_id=zero2,
            model_version=zero8,
            optimizer_version=zero8,
            lifecycle_version=zero8,
            plant_transition=self._blank_transition(),
            shadow_source_integrity_tag=jnp.asarray(0, dtype=jnp.uint32),
            shadow_result_content_tag=zero2,
            shadow_post_integrity_tag=jnp.asarray(0, dtype=jnp.uint32),
            checksum=zero2,
        )
        return self._with_commit_checksum(record)

    def _binding_payload(
        self,
        state: PrototypeEmbodiedDevelopmentHarnessState,
    ) -> object:
        return (
            state.harness_config_digest,
            state.adapter.binding_checksum,
            _content_tag(state.adapter, salt=0x48524144),
            state.plant.checksum,
            state.shadow.state_integrity_tag,
            state.pending.checksum,
            state.last_commit.checksum,
        )

    def _with_binding_checksum(
        self,
        state: PrototypeEmbodiedDevelopmentHarnessState,
    ) -> PrototypeEmbodiedDevelopmentHarnessState:
        return state.replace(
            binding_checksum=_content_tag(self._binding_payload(state), salt=0x48525354)
        )

    def _check_pending_contract(
        self,
        pending: PrototypeEmbodiedDevelopmentHarnessPendingState,
    ) -> None:
        if type(pending) is not PrototypeEmbodiedDevelopmentHarnessPendingState:
            raise TypeError("harness pending state has the wrong exact type")
        n = self._adapter.config.n_joints
        contracts = {
            "available": ((), jnp.bool_),
            "receipt_words": ((_IDENTITY_WORDS,), jnp.uint32),
            "prototype_decision_id": ((_DECISION_WORDS,), jnp.uint32),
            "selected_action": ((), jnp.int32),
            "adapter_binding_checksum": ((_IDENTITY_WORDS,), jnp.uint32),
            "adapter_state_content_tag": ((_IDENTITY_WORDS,), jnp.uint32),
            "adapter_pending_checksum": ((_IDENTITY_WORDS,), jnp.uint32),
            "plant_source_checksum": ((_IDENTITY_WORDS,), jnp.uint32),
            "shadow_source_integrity_tag": ((), jnp.uint32),
            "shadow_source_transaction_words": ((_IDENTITY_WORDS,), jnp.uint32),
            "shadow_result_content_tag": ((_IDENTITY_WORDS,), jnp.uint32),
            "harness_config_digest": ((_DIGEST_WORDS,), jnp.uint32),
            "checksum": ((_IDENTITY_WORDS,), jnp.uint32),
        }
        for name, (shape, dtype) in contracts.items():
            _require_array(
                getattr(pending, name),
                name=f"harness.pending.{name}",
                shape=shape,
                dtype=dtype,
            )
        command = pending.proposed_command
        if type(command) is not EmbodiedCommand:
            raise TypeError("harness.pending.proposed_command has the wrong type")
        for name, shape in (
            ("joint_position", (n,)),
            ("joint_velocity", (n,)),
            ("joint_torque", (n,)),
            ("workspace_position", (3,)),
            ("collision_clearance", ()),
        ):
            _require_array(
                getattr(command, name),
                name=f"harness.pending.proposed_command.{name}",
                shape=shape,
                dtype=jnp.float32,
            )

    def _check_commit_contract(
        self,
        record: PrototypeEmbodiedDevelopmentHarnessCommitRecord,
    ) -> None:
        if type(record) is not PrototypeEmbodiedDevelopmentHarnessCommitRecord:
            raise TypeError("harness last_commit has the wrong exact type")
        contracts = {
            "available": ((), jnp.bool_),
            "receipt_words": ((_IDENTITY_WORDS,), jnp.uint32),
            "prototype_decision_id": ((_DECISION_WORDS,), jnp.uint32),
            "selected_action": ((), jnp.int32),
            "executed_action": ((), jnp.int32),
            "successor_available": ((), jnp.bool_),
            "successor_decision_id": ((_DECISION_WORDS,), jnp.uint32),
            "successor_action": ((), jnp.int32),
            "semantic_successor_content_tag": ((_IDENTITY_WORDS,), jnp.uint32),
            "envelope_result_content_tag": ((_IDENTITY_WORDS,), jnp.uint32),
            "envelope_decision_id": ((_IDENTITY_WORDS,), jnp.uint32),
            "envelope_action_id": ((_IDENTITY_WORDS,), jnp.uint32),
            "telemetry_id": ((_IDENTITY_WORDS,), jnp.uint32),
            "model_version": ((_DIGEST_WORDS,), jnp.uint32),
            "optimizer_version": ((_DIGEST_WORDS,), jnp.uint32),
            "lifecycle_version": ((_DIGEST_WORDS,), jnp.uint32),
            "shadow_source_integrity_tag": ((), jnp.uint32),
            "shadow_result_content_tag": ((_IDENTITY_WORDS,), jnp.uint32),
            "shadow_post_integrity_tag": ((), jnp.uint32),
            "checksum": ((_IDENTITY_WORDS,), jnp.uint32),
        }
        for name, (shape, dtype) in contracts.items():
            _require_array(
                getattr(record, name),
                name=f"harness.last_commit.{name}",
                shape=shape,
                dtype=dtype,
            )
        self._check_pending_contract(
            self._blank_pending().replace(proposed_command=record.proposed_command)
        )
        self._check_pending_contract(
            self._blank_pending().replace(proposed_command=record.executed_command)
        )
        transition = record.plant_transition
        if type(transition) is not DeterministicPrimitivePlantTransition:
            raise TypeError("harness last_commit transition has the wrong type")
        dimension = self._plant.config.observation_dim
        for name, shape, dtype in (
            ("requested", (), jnp.bool_),
            ("proposal_applied", (), jnp.bool_),
            ("committed", (), jnp.bool_),
            ("receipt_words", (_IDENTITY_WORDS,), jnp.uint32),
            ("primitive_action", (), jnp.int32),
            ("pre_observation", (dimension,), jnp.float32),
            ("post_observation", (dimension,), jnp.float32),
            ("reward", (), jnp.float32),
            ("discount", (), jnp.float32),
            ("terminated", (), jnp.bool_),
            ("truncated", (), jnp.bool_),
            ("clipped", (), jnp.bool_),
            ("pre_transition_count", (), jnp.int32),
            ("post_transition_count", (), jnp.int32),
        ):
            _require_array(
                getattr(transition, name),
                name=f"harness.last_commit.transition.{name}",
                shape=shape,
                dtype=dtype,
            )

    def _check_state_contract(
        self,
        state: PrototypeEmbodiedDevelopmentHarnessState,
    ) -> None:
        if type(state) is not PrototypeEmbodiedDevelopmentHarnessState:
            raise TypeError("harness state has the wrong exact type")
        if type(state.adapter) is not PrototypeEmbodiedCommandAdapterState:
            raise TypeError("harness adapter state has the wrong exact type")
        if type(state.plant) is not DeterministicPrimitivePlantState:
            raise TypeError("harness plant state has the wrong exact type")
        if type(state.shadow) is not GroundedImaginationCompositionState:
            raise TypeError("harness shadow state has the wrong exact type")
        _require_array(
            state.harness_config_digest,
            name="harness.harness_config_digest",
            shape=(_DIGEST_WORDS,),
            dtype=jnp.uint32,
        )
        self._check_pending_contract(state.pending)
        self._check_commit_contract(state.last_commit)
        _require_array(
            state.binding_checksum,
            name="harness.binding_checksum",
            shape=(_IDENTITY_WORDS,),
            dtype=jnp.uint32,
        )

    def _pending_valid(
        self,
        state: PrototypeEmbodiedDevelopmentHarnessState,
    ) -> Array:
        pending = state.pending
        adapter_pending = state.adapter.pending
        selected_valid = (
            (pending.selected_action >= 0)
            & (pending.selected_action < self._adapter.config.n_actions)
        )
        proposed = self._adapter.map_command(pending.proposed_command)
        return (
            pending.available
            & (state.plant.transition_count < self._plant.config.max_transitions)
            & jnp.any(pending.receipt_words != jnp.uint32(0))
            & jnp.array_equal(pending.receipt_words, adapter_pending.receipt_words)
            & jnp.array_equal(
                pending.prototype_decision_id,
                adapter_pending.prototype_decision_id,
            )
            & (pending.selected_action == adapter_pending.selected_action)
            & jnp.array_equal(
                pending.adapter_binding_checksum,
                state.adapter.binding_checksum,
            )
            & jnp.array_equal(
                pending.adapter_state_content_tag,
                _content_tag(state.adapter, salt=0x48524144),
            )
            & jnp.array_equal(
                pending.adapter_pending_checksum,
                adapter_pending.checksum,
            )
            & adapter_pending.available
            & jnp.array_equal(pending.plant_source_checksum, state.plant.checksum)
            & (
                pending.shadow_source_integrity_tag
                == state.shadow.state_integrity_tag
            )
            & jnp.array_equal(
                pending.shadow_source_transaction_words,
                state.shadow.transaction_count_words,
            )
            & _tree_array_equal(
                pending.proposed_command,
                self._adapter.command_for_action(adapter_pending.selected_action),
            )
            & proposed.maps_exactly_one_primitive
            & (proposed.action == pending.selected_action)
            & selected_valid
            & jnp.any(pending.shadow_result_content_tag != jnp.uint32(0))
            & jnp.array_equal(pending.harness_config_digest, self._config_digest)
            & jnp.array_equal(
                pending.checksum,
                _content_tag(self._pending_payload(pending), salt=0x48525045),
            )
        )

    def _commit_valid(
        self,
        state: PrototypeEmbodiedDevelopmentHarnessState,
    ) -> Array:
        record = state.last_commit
        transition = record.plant_transition
        proposed_mapping = self._adapter.map_command(record.proposed_command)
        executed_mapping = self._adapter.map_command(record.executed_command)
        safe_executed = jnp.clip(
            record.executed_action,
            0,
            self._adapter.config.n_actions - 1,
        )
        expected_reward = jnp.asarray(
            self._plant.config.primitive_rewards,
            dtype=jnp.float32,
        )[safe_executed]
        prototype = state.adapter.semantic.composition.prototype
        owner = state.adapter.semantic.composition.dispatch_owner
        expected_successor_id, successor_clock_available = (
            _increment_decision_words(record.prototype_decision_id)
        )
        live_successor = (
            record.successor_available
            & (~transition.truncated)
            & successor_clock_available
            & prototype.started
            & owner.available
            & jnp.array_equal(
                record.successor_decision_id,
                expected_successor_id,
            )
            & jnp.array_equal(
                record.successor_decision_id,
                prototype.current_decision_id,
            )
            & jnp.array_equal(
                record.successor_decision_id,
                owner.prototype_decision_id,
            )
            & (record.successor_action == prototype.current_action)
            & (record.successor_action == owner.selected_action)
        )
        return (
            record.available
            & jnp.any(record.receipt_words != jnp.uint32(0))
            & (record.selected_action >= 0)
            & (record.selected_action < self._adapter.config.n_actions)
            & (record.executed_action >= 0)
            & (record.executed_action < self._adapter.config.n_actions)
            & proposed_mapping.maps_exactly_one_primitive
            & (proposed_mapping.action == record.selected_action)
            & executed_mapping.maps_exactly_one_primitive
            & (executed_mapping.action == record.executed_action)
            & transition.requested
            & transition.proposal_applied
            & transition.committed
            & jnp.isfinite(transition.reward)
            & (
                jax.lax.bitcast_convert_type(transition.reward, jnp.uint32)
                == jax.lax.bitcast_convert_type(expected_reward, jnp.uint32)
            )
            & jnp.isfinite(transition.discount)
            & (transition.discount >= 0.0)
            & (transition.discount <= 1.0)
            & (~transition.terminated)
            & (~transition.truncated)
            & (transition.discount == jnp.asarray(1.0, dtype=jnp.float32))
            & jnp.array_equal(transition.receipt_words, record.receipt_words)
            & (transition.primitive_action == record.executed_action)
            & state.plant.has_transition
            & jnp.array_equal(state.plant.last_receipt_words, record.receipt_words)
            & (state.plant.last_action == record.executed_action)
            & jnp.array_equal(
                state.plant.last_pre_observation,
                transition.pre_observation,
            )
            & jnp.array_equal(
                state.plant.last_post_observation,
                transition.post_observation,
            )
            & state.adapter.has_settled_prototype_decision
            & jnp.array_equal(
                state.adapter.last_settled_prototype_decision_id,
                record.prototype_decision_id,
            )
            & live_successor
            & jnp.array_equal(
                record.semantic_successor_content_tag,
                _content_tag(state.adapter.semantic, salt=0x48525345),
            )
            & (record.shadow_post_integrity_tag == state.shadow.state_integrity_tag)
            & jnp.array_equal(
                record.checksum,
                _content_tag(self._commit_payload(record), salt=0x4852434D),
            )
        )

    def _semantic_plant_bound(
        self,
        state: PrototypeEmbodiedDevelopmentHarnessState,
    ) -> Array:
        prototype = state.adapter.semantic.composition.prototype
        owner = state.adapter.semantic.composition.dispatch_owner
        raw = prototype.current_raw_observation[
            : self._adapter.config.semantic.raw_observation_dim
        ]
        armed = (
            prototype.started
            & owner.available
            & jnp.array_equal(owner.prototype_decision_id, prototype.current_decision_id)
            & (owner.selected_action == prototype.current_action)
            & _tree_array_equal(raw, state.plant.observation)
        )
        return armed

    def state_valid(self, state: PrototypeEmbodiedDevelopmentHarnessState) -> Array:
        self._check_state_contract(state)
        pending_layout = jnp.where(
            state.pending.available,
            self._pending_valid(state),
            _tree_array_equal(state.pending, self._blank_pending()),
        )
        commit_layout = jnp.where(
            state.last_commit.available,
            self._commit_valid(state),
            _tree_array_equal(state.last_commit, self._blank_commit()),
        )
        return (
            self._adapter.state_valid(state.adapter)
            & self._plant.state_valid(state.plant)
            & self._grounded.state_valid(state.shadow)
            & self._semantic_plant_bound(state)
            & jnp.array_equal(state.harness_config_digest, self._config_digest)
            & pending_layout
            & commit_layout
            & jnp.array_equal(
                state.binding_checksum,
                _content_tag(self._binding_payload(state), salt=0x48525354),
            )
        )

    def init(
        self,
        adapter_state: PrototypeEmbodiedCommandAdapterState,
        plant_state: DeterministicPrimitivePlantState,
        grounded_state: GroundedImaginationCompositionState,
    ) -> PrototypeEmbodiedDevelopmentHarnessState:
        state = PrototypeEmbodiedDevelopmentHarnessState(
            adapter=adapter_state,
            plant=plant_state,
            shadow=grounded_state,
            harness_config_digest=self._config_digest,
            pending=self._blank_pending(),
            last_commit=self._blank_commit(),
            binding_checksum=jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32),
        )
        state = self._with_binding_checksum(state)
        if not bool(jax.device_get(self.state_valid(state))):
            raise ValueError("cannot initialize from invalid or incompatible child states")
        return state

    def _check_preparation_contract(
        self,
        preparation: PrototypeEmbodiedDevelopmentHarnessPreparationInput,
    ) -> None:
        if type(preparation) is not PrototypeEmbodiedDevelopmentHarnessPreparationInput:
            raise TypeError("harness preparation input has the wrong exact type")
        if type(preparation.envelope) is not PrototypeEmbodiedCommandPreparationInput:
            raise TypeError("harness envelope preparation has the wrong exact type")
        if type(preparation.model_state) is not WorldModelEnsembleState:
            raise TypeError("harness model_state has the wrong exact type")
        planner = self._grounded.planner
        shape = (
            self._grounded.gauge.rollout_budget,
            self._grounded.gauge.rollout_horizon,
        )
        for name, value, expected_shape, dtype in (
            (
                "action_support_counts",
                preparation.action_support_counts,
                (planner.n_actions,),
                jnp.int32,
            ),
            (
                "source_revision_words",
                preparation.source_revision_words,
                (_IDENTITY_WORDS,),
                jnp.uint32,
            ),
            ("region_ids", preparation.region_ids, shape, jnp.int32),
            ("safety_admitted", preparation.safety_admitted, shape, jnp.bool_),
            ("protected", preparation.protected, shape, jnp.bool_),
        ):
            _require_array(
                value,
                name=f"harness preparation {name}",
                shape=expected_shape,
                dtype=dtype,
            )

    def prepare(
        self,
        state: PrototypeEmbodiedDevelopmentHarnessState,
        preparation: PrototypeEmbodiedDevelopmentHarnessPreparationInput,
    ) -> PrototypeEmbodiedDevelopmentHarnessPreparationResult:
        """Host-orchestrate one adapter receipt and one internally-JIT shadow step."""

        self._check_state_contract(state)
        self._check_preparation_contract(preparation)
        source_valid = self.state_valid(state)
        adapter_result = self._adapter.prepare(state.adapter, preparation.envelope)
        prototype_decision = adapter_result.prototype_decision_id
        shadow_result = self._grounded.step(
            state.shadow,
            model_state=preparation.model_state,
            action_support_counts=preparation.action_support_counts,
            source_revision_words=preparation.source_revision_words,
            real_observation=state.plant.observation,
            decision_id_words=prototype_decision[-2:],
            region_ids=preparation.region_ids,
            safety_admitted=preparation.safety_admitted,
            protected=preparation.protected,
        )
        shadow_state_valid = self._grounded.state_valid(shadow_result.state)
        shadow_source_bound = jnp.array_equal(
            shadow_result.diagnostics.pre_transaction_count_words,
            state.shadow.transaction_count_words,
        ) & jnp.where(
            shadow_result.diagnostics.transaction_applied,
            ~_tree_array_equal(shadow_result.state, state.shadow),
            _tree_array_equal(shadow_result.state, state.shadow),
        )
        shadow_tag = _content_tag(shadow_result, salt=0x48525348)
        shadow_content_bound = jnp.any(shadow_tag != jnp.uint32(0))
        plant_capacity_available = (
            state.plant.transition_count < self._plant.config.max_transitions
        )
        prepare_pre = (
            source_valid
            & (~state.pending.available)
            & plant_capacity_available
            & adapter_result.diagnostics.prepared
            & shadow_state_valid
            & shadow_source_bound
            & shadow_content_bound
        )
        adapter_candidate = adapter_result.state
        pending = self._with_pending_checksum(
            PrototypeEmbodiedDevelopmentHarnessPendingState(
                available=jnp.asarray(True, dtype=jnp.bool_),
                receipt_words=adapter_result.receipt_words,
                prototype_decision_id=prototype_decision,
                selected_action=adapter_result.selected_action,
                adapter_binding_checksum=adapter_candidate.binding_checksum,
                adapter_state_content_tag=_content_tag(
                    adapter_candidate,
                    salt=0x48524144,
                ),
                adapter_pending_checksum=adapter_candidate.pending.checksum,
                plant_source_checksum=state.plant.checksum,
                shadow_source_integrity_tag=state.shadow.state_integrity_tag,
                shadow_source_transaction_words=state.shadow.transaction_count_words,
                proposed_command=adapter_result.command,
                shadow_result_content_tag=shadow_tag,
                harness_config_digest=self._config_digest,
                checksum=jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32),
            )
        )
        candidate = self._with_binding_checksum(
            state.replace(
                adapter=adapter_candidate,
                pending=pending,
                binding_checksum=jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32),
            )
        )
        candidate_valid = self.state_valid(candidate)
        prepared = prepare_pre & candidate_valid
        final_state = cast(
            PrototypeEmbodiedDevelopmentHarnessState,
            jax.lax.cond(prepared, lambda _: candidate, lambda _: state, operand=None),
        )
        return PrototypeEmbodiedDevelopmentHarnessPreparationResult(
            state=final_state,
            command=cast(
                EmbodiedCommand,
                jax.lax.cond(
                    prepared,
                    lambda _: adapter_result.command,
                    lambda _: self._zero_command(),
                    operand=None,
                ),
            ),
            receipt_words=jnp.where(
                prepared,
                adapter_result.receipt_words,
                jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32),
            ),
            adapter=adapter_result,
            shadow=shadow_result,
            diagnostics=PrototypeEmbodiedDevelopmentHarnessPreparationDiagnostics(
                source_state_valid=source_valid,
                receipt_slot_available=~state.pending.available,
                plant_capacity_available=plant_capacity_available,
                adapter_prepared=adapter_result.diagnostics.prepared,
                shadow_source_bound=shadow_source_bound,
                shadow_result_state_valid=shadow_state_valid,
                shadow_result_content_bound=shadow_content_bound,
                shadow_result_authenticated=jnp.asarray(False, dtype=jnp.bool_),
                candidate_state_valid=candidate_valid,
                prepared=prepared,
                physical_dispatch_authority=jnp.asarray(False, dtype=jnp.bool_),
                safety_authority=jnp.asarray(False, dtype=jnp.bool_),
            ),
        )

    def evaluate_pending_envelope(
        self,
        state: PrototypeEmbodiedDevelopmentHarnessState,
    ) -> EmbodiedEnvelopeDecision:
        """Evaluate the exact nested adapter receipt without state adoption."""

        self._check_state_contract(state)
        return self._adapter.evaluate_pending(state.adapter)

    def settle(
        self,
        state: PrototypeEmbodiedDevelopmentHarnessState,
        envelope_result: EmbodiedEnvelopeDecision,
        shadow_result: GroundedImaginationCompositionResult,
    ) -> PrototypeEmbodiedDevelopmentHarnessSettlementResult:
        """Propose plant work, then settle semantic ownership, then adopt atomically."""

        self._check_state_contract(state)
        if type(envelope_result) is not EmbodiedEnvelopeDecision:
            raise TypeError("envelope_result must be an exact EmbodiedEnvelopeDecision")
        if type(shadow_result) is not GroundedImaginationCompositionResult:
            raise TypeError(
                "shadow_result must be an exact GroundedImaginationCompositionResult"
            )
        source_valid = self.state_valid(state)
        pending = state.pending
        adapter_pending = state.adapter.pending

        expected_envelope = self._adapter.evaluate_pending(state.adapter)
        envelope_exact = _tree_array_equal(envelope_result, expected_envelope)
        mapping = self._adapter.map_command(envelope_result.command)
        safe_action = jnp.clip(mapping.action, 0, self._adapter.config.n_actions - 1)
        mapped_admitted = (
            mapping.maps_exactly_one_primitive
            & adapter_pending.hard_safety_action_mask[safe_action]
        )
        accepted_mapping = (
            expected_envelope.transaction_applied
            & expected_envelope.action_available
            & expected_envelope.proposed_accepted
            & (~expected_envelope.fallback_used)
            & (mapping.action == adapter_pending.selected_action)
        )
        fallback_mapping = (
            expected_envelope.transaction_applied
            & expected_envelope.action_available
            & (~expected_envelope.proposed_accepted)
            & expected_envelope.fallback_used
            & expected_envelope.fallback_certified
        )
        plant_requested = (
            source_valid
            & pending.available
            & envelope_exact
            & mapping.maps_exactly_one_primitive
            & mapped_admitted
            & (accepted_mapping | fallback_mapping)
        )

        # This source order is load-bearing: the plant transition is proposed
        # before the adapter invokes post-envelope semantic settlement. Both
        # remain pure candidates until the outer transaction adopts them.
        plant_result = self._plant.propose_step(
            state.plant,
            requested=plant_requested,
            receipt_words=pending.receipt_words,
            primitive_action=mapping.action,
        )
        adapter_result = self._adapter.settle(state.adapter, envelope_result)

        settled_prototype = adapter_result.state.semantic.composition.prototype
        settled_raw_observation = settled_prototype.current_raw_observation[
            : self._adapter.config.semantic.raw_observation_dim
        ]
        semantic_transition_requested = (
            source_valid
            & pending.available
            & adapter_result.diagnostics.receipt_consumed
            & plant_result.transition.proposal_applied
            & (plant_result.transition.primitive_action == adapter_result.action)
            & jnp.array_equal(
                pending.prototype_decision_id,
                settled_prototype.current_decision_id,
            )
            & _tree_array_equal(
                settled_raw_observation,
                plant_result.transition.pre_observation,
            )
        )
        semantic_transition = PrototypeConsolidatedSemanticTransition(
            observation=plant_result.transition.pre_observation,
            action=adapter_result.action,
            decision_id=pending.prototype_decision_id,
            reward=plant_result.transition.reward,
            discount=plant_result.transition.discount,
            terminated=plant_result.transition.terminated,
            truncated=plant_result.transition.truncated,
            next_observation=plant_result.transition.post_observation,
            next_decision_observation=plant_result.transition.post_observation,
        )

        def advance_semantic(
            _: None,
        ) -> tuple[
            PrototypeConsolidatedSemanticMemoryState,
            Array,
            Array,
            Array,
            Array,
        ]:
            result = self._adapter.semantic.update_transition(
                adapter_result.state.semantic,
                semantic_transition,
            )
            return (
                result.state,
                result.diagnostics.outer_transaction_committed,
                result.diagnostics.prototype_learning_retained,
                result.diagnostics.action_available,
                result.action,
            )

        def preserve_semantic(
            _: None,
        ) -> tuple[
            PrototypeConsolidatedSemanticMemoryState,
            Array,
            Array,
            Array,
            Array,
        ]:
            return (
                adapter_result.state.semantic,
                jnp.asarray(False, dtype=jnp.bool_),
                jnp.asarray(False, dtype=jnp.bool_),
                jnp.asarray(False, dtype=jnp.bool_),
                jnp.asarray(-1, dtype=jnp.int32),
            )

        (
            semantic_successor_object,
            semantic_transition_committed,
            semantic_prototype_learning_retained,
            semantic_action_available,
            semantic_successor_action,
        ) = jax.lax.cond(
            semantic_transition_requested,
            advance_semantic,
            preserve_semantic,
            operand=None,
        )
        semantic_successor = cast(
            PrototypeConsolidatedSemanticMemoryState,
            semantic_successor_object,
        )
        successor_prototype = semantic_successor.composition.prototype
        successor_owner = semantic_successor.composition.dispatch_owner
        expected_successor_id, successor_clock_available = (
            _increment_decision_words(pending.prototype_decision_id)
        )
        successor_raw_observation = successor_prototype.current_raw_observation[
            : self._adapter.config.semantic.raw_observation_dim
        ]
        semantic_successor_plant_bound = (
            semantic_transition_committed
            & successor_prototype.started
            & _tree_array_equal(
                successor_raw_observation,
                plant_result.transition.post_observation,
            )
        )
        semantic_successor_rearmed = (
            semantic_transition_committed
            & semantic_prototype_learning_retained
            & semantic_action_available
            & successor_clock_available
            & semantic_successor_plant_bound
            & jnp.array_equal(
                successor_prototype.current_decision_id,
                expected_successor_id,
            )
            & successor_owner.available
            & jnp.array_equal(
                successor_owner.prototype_decision_id,
                successor_prototype.current_decision_id,
            )
            & (successor_owner.selected_action == successor_prototype.current_action)
            & (semantic_successor_action == successor_prototype.current_action)
        )
        successor_adapter = adapter_result.state.replace(
            semantic=semantic_successor,
        )
        successor_adapter_valid = self._adapter.state_valid(successor_adapter)
        semantic_successor_valid = (
            successor_adapter_valid & semantic_successor_rearmed
        )

        shadow_tag = _content_tag(shadow_result, salt=0x48525348)
        shadow_content_matches = jnp.array_equal(
            shadow_tag,
            pending.shadow_result_content_tag,
        )
        shadow_source_bound = (
            pending.available
            & (
                pending.shadow_source_integrity_tag
                == state.shadow.state_integrity_tag
            )
            & jnp.array_equal(
                pending.shadow_source_transaction_words,
                state.shadow.transaction_count_words,
            )
            & jnp.array_equal(
                shadow_result.diagnostics.pre_transaction_count_words,
                state.shadow.transaction_count_words,
            )
            & jnp.where(
                shadow_result.diagnostics.transaction_applied,
                ~_tree_array_equal(shadow_result.state, state.shadow),
                _tree_array_equal(shadow_result.state, state.shadow),
            )
        )
        shadow_state_valid = self._grounded.state_valid(shadow_result.state)
        shadow_exact = (
            shadow_content_matches & shadow_source_bound & shadow_state_valid
        )

        action_pre = semantic_transition_requested & semantic_successor_valid & shadow_exact
        committed_transition = plant_result.transition.replace(
            committed=jnp.asarray(True, dtype=jnp.bool_)
        )
        commit_record = self._with_commit_checksum(
            PrototypeEmbodiedDevelopmentHarnessCommitRecord(
                available=jnp.asarray(True, dtype=jnp.bool_),
                receipt_words=pending.receipt_words,
                prototype_decision_id=pending.prototype_decision_id,
                selected_action=pending.selected_action,
                executed_action=adapter_result.action,
                successor_available=semantic_successor_rearmed,
                successor_decision_id=jnp.where(
                    semantic_successor_rearmed,
                    successor_prototype.current_decision_id,
                    jnp.zeros((_DECISION_WORDS,), dtype=jnp.uint32),
                ),
                successor_action=jnp.where(
                    semantic_successor_rearmed,
                    semantic_successor_action,
                    jnp.asarray(-1, dtype=jnp.int32),
                ).astype(jnp.int32),
                semantic_successor_content_tag=_content_tag(
                    semantic_successor,
                    salt=0x48525345,
                ),
                proposed_command=pending.proposed_command,
                executed_command=envelope_result.command,
                envelope_result_content_tag=_content_tag(
                    envelope_result,
                    salt=0x4852454E,
                ),
                envelope_decision_id=adapter_pending.envelope_decision_id,
                envelope_action_id=adapter_pending.envelope_action_id,
                telemetry_id=adapter_pending.telemetry.telemetry_id,
                model_version=adapter_pending.model_version,
                optimizer_version=adapter_pending.optimizer_version,
                lifecycle_version=adapter_pending.lifecycle_version,
                plant_transition=committed_transition,
                shadow_source_integrity_tag=(
                    pending.shadow_source_integrity_tag
                ),
                shadow_result_content_tag=shadow_tag,
                shadow_post_integrity_tag=(
                    shadow_result.state.state_integrity_tag
                ),
                checksum=jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32),
            )
        )
        action_candidate = self._with_binding_checksum(
            state.replace(
                adapter=successor_adapter,
                plant=plant_result.state,
                shadow=shadow_result.state,
                pending=self._blank_pending(),
                last_commit=commit_record,
                binding_checksum=jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32),
            )
        )
        action_candidate_valid = self.state_valid(action_candidate)
        action_committed = action_pre & action_candidate_valid

        envelope_only_pre = (
            source_valid
            & pending.available
            & adapter_result.diagnostics.envelope_only_state_committed
            & _tree_array_equal(adapter_result.state.semantic, state.adapter.semantic)
            & _tree_array_equal(plant_result.state, state.plant)
        )
        envelope_only_candidate = self._with_binding_checksum(
            state.replace(
                adapter=adapter_result.state,
                pending=self._blank_pending(),
                binding_checksum=jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32),
            )
        )
        envelope_only_candidate_valid = self.state_valid(envelope_only_candidate)
        envelope_only_committed = (
            envelope_only_pre & envelope_only_candidate_valid
        )
        final_state = cast(
            PrototypeEmbodiedDevelopmentHarnessState,
            jax.lax.cond(
                action_committed,
                lambda _: action_candidate,
                lambda _: jax.lax.cond(
                    envelope_only_committed,
                    lambda __: envelope_only_candidate,
                    lambda __: state,
                    operand=None,
                ),
                operand=None,
            ),
        )
        final_action = jnp.where(
            action_committed,
            adapter_result.action,
            jnp.asarray(-1, dtype=jnp.int32),
        ).astype(jnp.int32)
        transaction_committed = action_committed | envelope_only_committed
        returned_transition = cast(
            DeterministicPrimitivePlantTransition,
            jax.lax.cond(
                action_committed,
                lambda _: committed_transition,
                lambda _: plant_result.transition,
                operand=None,
            ),
        )
        exposed_adapter_state = cast(
            PrototypeEmbodiedCommandAdapterState,
            jax.lax.cond(
                action_committed,
                lambda _: successor_adapter,
                lambda _: adapter_result.state,
                operand=None,
            ),
        )
        exposed_adapter_result = adapter_result.replace(state=exposed_adapter_state)
        return PrototypeEmbodiedDevelopmentHarnessSettlementResult(
            state=final_state,
            action=final_action,
            transition=returned_transition,
            adapter=exposed_adapter_result,
            envelope=envelope_result,
            shadow=shadow_result,
            diagnostics=PrototypeEmbodiedDevelopmentHarnessSettlementDiagnostics(
                source_state_valid=source_valid,
                pending_receipt_available=pending.available,
                envelope_result_exact=envelope_exact,
                mapped_action_unique=mapping.maps_exactly_one_primitive,
                mapped_action_admitted=mapped_admitted,
                plant_proposal_requested=plant_requested,
                plant_proposal_applied=(
                    plant_result.transition.proposal_applied
                ),
                plant_transition_committed=action_committed,
                semantic_settlement_after_plant_proposal=(
                    source_valid & pending.available
                ),
                semantic_transition_requested=semantic_transition_requested,
                semantic_transition_committed=semantic_transition_committed,
                semantic_prototype_learning_retained=(
                    semantic_prototype_learning_retained
                ),
                semantic_successor_plant_bound=semantic_successor_plant_bound,
                semantic_successor_rearmed=semantic_successor_rearmed,
                plant_capacity_exhausted_after_commit=(
                    action_committed
                    & (
                        final_state.plant.transition_count
                        == self._plant.config.max_transitions
                    )
                ),
                adapter_action_receipt_consumed=(
                    adapter_result.diagnostics.receipt_consumed
                ),
                adapter_envelope_only_committed=(
                    adapter_result.diagnostics.envelope_only_state_committed
                ),
                shadow_result_content_matches_receipt=shadow_content_matches,
                shadow_result_source_bound=shadow_source_bound,
                shadow_result_state_valid=shadow_state_valid,
                shadow_result_authenticated=jnp.asarray(False, dtype=jnp.bool_),
                shadow_adopted=action_committed,
                no_action_plant_unchanged=(
                    envelope_only_committed
                    & _tree_array_equal(final_state.plant, state.plant)
                ),
                no_action_shadow_unchanged=(
                    envelope_only_committed
                    & _tree_array_equal(final_state.shadow, state.shadow)
                ),
                emergency_stop_latch_preserved=(
                    envelope_only_committed
                    & envelope_result.emergency_stop_latch_applied
                    & final_state.adapter.envelope.emergency_stop_latched
                ),
                action_transaction_committed=action_committed,
                envelope_only_transaction_committed=envelope_only_committed,
                transaction_committed=transaction_committed,
                learning_updates_applied_by_real_adapter=jnp.asarray(
                    0, dtype=jnp.int32
                ),
                prototype_learning_updates_adopted=action_committed.astype(
                    jnp.int32
                ),
                shadow_backward_evaluations_per_prepare=(
                    shadow_result.diagnostics.commit_autodiff_pass_count
                ),
                physical_dispatch_authority=jnp.asarray(False, dtype=jnp.bool_),
                safety_authority=jnp.asarray(False, dtype=jnp.bool_),
                evidence_authority=jnp.asarray(False, dtype=jnp.bool_),
                promotion_authority=jnp.asarray(False, dtype=jnp.bool_),
            ),
        )

    def checkpoint_payload(
        self,
        state: PrototypeEmbodiedDevelopmentHarnessState,
    ) -> dict[str, object]:
        """Return one strict host-only checkpoint; SHA-256 is not authentication."""

        if not bool(jax.device_get(self.state_valid(state))):
            raise ValueError("cannot checkpoint an invalid development harness state")
        return {
            "schema": PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_CHECKPOINT_SCHEMA,
            "config": self.to_config(),
            "adapter": self._adapter.checkpoint_payload(state.adapter),
            "plant": state.plant,
            "shadow": state.shadow,
            "harness_config_digest": state.harness_config_digest,
            "pending": state.pending,
            "last_commit": state.last_commit,
            "binding_checksum": state.binding_checksum,
            "state_sha256": _tree_sha256(state),
        }

    def restore_checkpoint(
        self,
        payload: object,
        *,
        semantic_source_digest: Array,
        semantic_namespace_digest: Array,
        semantic_representation_revision: int | Array,
        semantic_source_revision: int | Array,
        envelope_source_digest: Array,
        trusted_envelope_state_revision: int | Array,
        trusted_envelope_state_digest: Array,
        trusted_adapter_state_digest: Array,
        trusted_harness_state_digest: Array,
    ) -> PrototypeEmbodiedDevelopmentHarnessState:
        if type(payload) is not dict:
            raise ValueError("development harness checkpoint must be an exact dict")
        raw = cast(dict[object, object], payload)
        expected = {
            "schema",
            "config",
            "adapter",
            "plant",
            "shadow",
            "harness_config_digest",
            "pending",
            "last_commit",
            "binding_checksum",
            "state_sha256",
        }
        if set(raw) != expected:
            raise ValueError("development harness checkpoint fields differ from schema v1")
        if raw["schema"] != PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_CHECKPOINT_SCHEMA:
            raise ValueError("development harness checkpoint schema differs")
        if raw["config"] != self.to_config():
            raise ValueError("development harness checkpoint config differs")
        adapter = self._adapter.restore_checkpoint(
            raw["adapter"],
            semantic_source_digest=semantic_source_digest,
            semantic_namespace_digest=semantic_namespace_digest,
            semantic_representation_revision=semantic_representation_revision,
            semantic_source_revision=semantic_source_revision,
            envelope_source_digest=envelope_source_digest,
            trusted_envelope_state_revision=trusted_envelope_state_revision,
            trusted_envelope_state_digest=trusted_envelope_state_digest,
            trusted_adapter_state_digest=trusted_adapter_state_digest,
        )
        plant = raw["plant"]
        shadow = raw["shadow"]
        pending = raw["pending"]
        last_commit = raw["last_commit"]
        if type(plant) is not DeterministicPrimitivePlantState:
            raise ValueError("development harness checkpoint plant type differs")
        if type(shadow) is not GroundedImaginationCompositionState:
            raise ValueError("development harness checkpoint shadow type differs")
        if type(pending) is not PrototypeEmbodiedDevelopmentHarnessPendingState:
            raise ValueError("development harness checkpoint pending type differs")
        if type(last_commit) is not PrototypeEmbodiedDevelopmentHarnessCommitRecord:
            raise ValueError("development harness checkpoint commit type differs")
        config_digest = _require_array(
            raw["harness_config_digest"],
            name="checkpoint.harness_config_digest",
            shape=(_DIGEST_WORDS,),
            dtype=jnp.uint32,
        )
        binding = _require_array(
            raw["binding_checksum"],
            name="checkpoint.binding_checksum",
            shape=(_IDENTITY_WORDS,),
            dtype=jnp.uint32,
        )
        persisted_sha = _require_array(
            raw["state_sha256"],
            name="checkpoint.state_sha256",
            shape=(_DIGEST_BYTES,),
            dtype=jnp.uint8,
        )
        trusted_sha = _require_array(
            trusted_harness_state_digest,
            name="trusted_harness_state_digest",
            shape=(_DIGEST_BYTES,),
            dtype=jnp.uint8,
        )
        restored = PrototypeEmbodiedDevelopmentHarnessState(
            adapter=adapter,
            plant=plant,
            shadow=shadow,
            harness_config_digest=config_digest,
            pending=pending,
            last_commit=last_commit,
            binding_checksum=binding,
        )
        valid = (
            jnp.array_equal(config_digest, self._config_digest)
            & jnp.array_equal(persisted_sha, trusted_sha)
            & jnp.array_equal(persisted_sha, _tree_sha256(restored))
            & self.state_valid(restored)
        )
        if not bool(jax.device_get(valid)):
            raise ValueError("development harness checkpoint is invalid, stale, or tampered")
        return restored

    def resource_budget(
        self,
        state: PrototypeEmbodiedDevelopmentHarnessState,
    ) -> PrototypeEmbodiedDevelopmentHarnessResourceBudget:
        self._check_state_contract(state)
        if not bool(jax.device_get(self.state_valid(state))):
            raise ValueError("cannot measure an invalid development harness state")
        grounded = self._grounded.resource_budget
        planner = self._grounded.planner.resource_budget
        return PrototypeEmbodiedDevelopmentHarnessResourceBudget(
            persistent_state_nbytes=_tree_nbytes(state),
            adapter_state_nbytes=_tree_nbytes(state.adapter),
            plant_state_nbytes=_tree_nbytes(state.plant),
            shadow_state_nbytes=_tree_nbytes(state.shadow),
            pending_receipts=1,
            last_commit_records=1,
            plant_observation_dim=self._plant.config.observation_dim,
            primitive_actions=self._plant.config.n_actions,
            maximum_plant_transitions=self._plant.config.max_transitions,
            plant_capacity_halts_prepare=True,
            budget_exhaustion_environment_boundaries_inferred=0,
            logical_adapter_prepare_calls_per_prepare=1,
            logical_shadow_steps_per_prepare=1,
            logical_adapter_settlements_per_settle=1,
            logical_plant_proposals_per_settle=1,
            maximum_semantic_transition_calls_per_settle=1,
            maximum_prototype_learning_updates_per_settle=1,
            no_action_semantic_transition_calls_per_settle=0,
            maximum_plant_transitions_per_settle=1,
            maximum_shadow_planner_calls_per_prepare=(
                grounded.max_planner_calls_per_call
            ),
            maximum_shadow_authorization_calls_per_prepare=(
                grounded.max_authorization_calls_per_call
            ),
            maximum_shadow_actor_critic_commits_per_prepare=(
                grounded.max_actor_critic_commits_per_call
            ),
            maximum_shadow_backward_transitions_per_prepare=(
                grounded.max_backward_transitions_per_call
            ),
            maximum_shadow_autodiff_passes_per_prepare=(
                grounded.max_autodiff_passes_per_call
            ),
            maximum_shadow_rng_splits_per_prepare=planner.max_rng_splits_per_call,
            maximum_shadow_rng_draws_per_prepare=planner.max_rng_draws_per_call,
            shadow_recomputations_per_settle=0,
            physical_dispatches_per_operation=0,
            real_adapter_learning_updates_per_settle=0,
            evidence_writes_per_operation=0,
            persistent_growth_per_operation_bytes=0,
            checkpoint_host_only=True,
            prepare_host_orchestrated=True,
            grounded_internal_jit=True,
            jit_settle_supported=True,
            shadow_result_authenticated=False,
            shadow_content_integrity_only=True,
            shadow_dispatch_authority=False,
            plant_physical_dispatch_authority=False,
            safety_authority=False,
            evidence_authority=False,
            promotion_authority=False,
            scientific_promotion_allowed=False,
            grounded=grounded,
        )


__all__ = [
    "PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_CHECKPOINT_HOST_ONLY",
    "PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_CHECKPOINT_SCHEMA",
    "PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_CONFIG_SCHEMA",
    "PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_EVIDENCE_AUTHORITY",
    "PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_GROUNDED_INTERNAL_JIT",
    "PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_JIT_SETTLE_SUPPORTED",
    "PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_MECHANISM_STATUS",
    "PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_PHYSICAL_DISPATCH_AUTHORITY",
    "PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_PREPARE_HOST_ORCHESTRATED",
    "PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_PROMOTION_AUTHORITY",
    "PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_SAFETY_AUTHORITY",
    "PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_SCIENTIFIC_PROMOTION_ALLOWED",
    "PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_SHADOW_RESULT_AUTHENTICATED",
    "DeterministicPrimitivePlant",
    "DeterministicPrimitivePlantConfig",
    "DeterministicPrimitivePlantState",
    "DeterministicPrimitivePlantStepResult",
    "DeterministicPrimitivePlantTransition",
    "PrototypeEmbodiedDevelopmentHarness",
    "PrototypeEmbodiedDevelopmentHarnessCommitRecord",
    "PrototypeEmbodiedDevelopmentHarnessPendingState",
    "PrototypeEmbodiedDevelopmentHarnessPreparationDiagnostics",
    "PrototypeEmbodiedDevelopmentHarnessPreparationInput",
    "PrototypeEmbodiedDevelopmentHarnessPreparationResult",
    "PrototypeEmbodiedDevelopmentHarnessResourceBudget",
    "PrototypeEmbodiedDevelopmentHarnessSettlementDiagnostics",
    "PrototypeEmbodiedDevelopmentHarnessSettlementResult",
    "PrototypeEmbodiedDevelopmentHarnessState",
]
