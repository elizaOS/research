# mypy: disable-error-code="attr-defined,call-arg"
"""Exact experiential-memory rebinding for Prototype pair-feature banks.

The standalone wrapper in this module closes one deliberately narrow
Prototype integration boundary.  It binds an :class:`ExperientialMemoryState`
to the exact :class:`PrototypeFeatureConsumerBinding` whose pair descriptors
encoded its representation-bearing rows.  At a lifecycle curation boundary,
``rebind`` reconstructs every valid row from its stable Identity-builder base
prefix and the destination pair bank:

* observations and keys are ``[base, pair-products]`` and bit-identical;
* outcomes are ``[base, pair-products, reward]`` with a bit-identical reward;
* the int32 representation-version telemetry is updated to the destination
  generation while the wrapper retains the exact two-word identity; and
* all non-representation payload, counters, timestamps, and memory clocks are
  preserved bit-for-bit.

This is a fixed-memory migration mechanism, not evidence that feature
replacement improves recall.  The SHA-256 schema digest detects accidental
composition drift; it is not a cryptographic authenticity boundary.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
from typing import Any, ClassVar, cast

import chex
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Int, UInt

from alberta_framework.core.experiential_memory import (
    ExperientialMemory,
    ExperientialMemoryConfig,
    ExperientialMemoryEntries,
    ExperientialMemoryState,
)
from alberta_framework.core.feature_bank_router import (
    FeatureBankRouter,
    FeatureBankRouterConfig,
)
from alberta_framework.core.prototype_feature_lifecycle import (
    PrototypeFeatureConsumerBinding,
    PrototypeFeatureLifecycleConfig,
)
from alberta_framework.core.state_builder import IdentityStateBuilderConfig

PROTOTYPE_FEATURE_MEMORY_CONFIG_SCHEMA = "alberta.prototype-feature-memory.config.v1"
PROTOTYPE_FEATURE_MEMORY_STATE_SCHEMA = "alberta.prototype-feature-memory.state.v1"
PROTOTYPE_FEATURE_MEMORY_MECHANISM_STATUS = "development_mechanism_only"
PROTOTYPE_FEATURE_MEMORY_SCIENTIFIC_PROMOTION_ALLOWED = False

_INT32_MAX = 2_147_483_647
_UINT32_MAX = 4_294_967_295
_SCHEMA_DIGEST_NBYTES = 32
_BASE_SEMANTICS = "IdentityStateBuilder stable raw-observation prefix"
_OBSERVATION_SEMANTICS = "[stable-base,pair-products]"
_KEY_SEMANTICS = "bit-identical-to-observation"
_OUTCOME_SEMANTICS = "[stable-base,pair-products,bit-identical-reward]"
_MIGRATION_SEMANTICS = (
    "valid-rows-only;recompute-from-stable-prefix;preserve-all-other-bits"
)


def _tree_nbytes(tree: object) -> int:
    """Return exact bytes in all persistent array leaves."""

    return sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(tree)
    )


def _require_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    """Reject shape or dtype drift without coercion."""

    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must be an array with exact shape and dtype")
    array = cast(Array, value)
    if tuple(array.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}; got {tuple(array.shape)}")
    expected_dtype = jnp.dtype(dtype)
    if jnp.dtype(array.dtype) != expected_dtype:
        raise TypeError(f"{name} must have dtype {expected_dtype}; got {array.dtype}")
    return array


def _telemetry_from_words(words: Array) -> Array:
    """Project an exact generation identity to saturating int32 telemetry."""

    below_saturation = (words[0] == jnp.uint32(0)) & (
        words[1] < jnp.uint32(_INT32_MAX)
    )
    result: Array = jnp.where(
        below_saturation,
        words[1].astype(jnp.int32),
        jnp.int32(_INT32_MAX),
    )
    return result


def _words_successor(source: Array, destination: Array) -> Bool[Array, ""]:
    """Return whether ``destination`` is the non-wrapping uint64 successor."""

    available = ~jnp.all(source == jnp.uint32(_UINT32_MAX))
    low = source[1] + jnp.uint32(1)
    carry = (low == jnp.uint32(0)).astype(jnp.uint32)
    candidate = jnp.stack((source[0] + carry, low)).astype(jnp.uint32)
    result: Bool[Array, ""] = available & jnp.all(destination == candidate)
    return result


def _float32_bits(value: Array) -> Array:
    """Expose float32 payload bits for signed-zero-sensitive comparisons."""

    return jax.lax.bitcast_convert_type(value, jnp.uint32)


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeFeatureMemoryConfig:
    """Exact feature-lifecycle, memory, and stable-base composition.

    Only an exact :class:`IdentityStateBuilderConfig` is accepted.  A learned
    or history-dependent builder would not provide the immutable base prefix
    required to reconstruct old rows after feature-bank replacement.
    """

    feature_lifecycle: PrototypeFeatureLifecycleConfig
    experiential_memory: ExperientialMemoryConfig
    base_state_builder: IdentityStateBuilderConfig

    SCHEMA_VERSION: ClassVar[str] = PROTOTYPE_FEATURE_MEMORY_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if type(self.feature_lifecycle) is not PrototypeFeatureLifecycleConfig:
            raise TypeError(
                "feature_lifecycle must be an exact PrototypeFeatureLifecycleConfig"
            )
        if type(self.experiential_memory) is not ExperientialMemoryConfig:
            raise TypeError(
                "experiential_memory must be an exact ExperientialMemoryConfig"
            )
        if type(self.base_state_builder) is not IdentityStateBuilderConfig:
            raise TypeError(
                "base_state_builder must be an exact IdentityStateBuilderConfig"
            )

        feature = self.feature_lifecycle
        memory = self.experiential_memory
        if memory.capacity > _INT32_MAX:
            raise ValueError(
                "experiential-memory capacity exceeds exact int32 row diagnostics"
            )
        if 3 * feature.active_pair_slots * memory.capacity > _INT32_MAX:
            raise ValueError(
                "feature-memory rebind pair-product work exceeds exact int32 diagnostics"
            )
        # ExperientialMemoryConfig predates dataclass-level validation.
        ExperientialMemory(memory)
        if self.base_state_builder.observation_dim != feature.base_feature_dim:
            raise ValueError(
                "Identity base width must equal feature_lifecycle.base_feature_dim"
            )
        if memory.observation_dim != feature.total_feature_dim:
            raise ValueError(
                "experiential-memory observation_dim must equal lifecycle total width"
            )
        if memory.key_dim != feature.total_feature_dim:
            raise ValueError(
                "experiential-memory key_dim must equal lifecycle total width"
            )
        if memory.outcome_dim != feature.total_feature_dim + 1:
            raise ValueError(
                "experiential-memory outcome_dim must equal lifecycle total width plus reward"
            )

    def _digest_payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA_VERSION,
            "state_schema": PROTOTYPE_FEATURE_MEMORY_STATE_SCHEMA,
            "type": "PrototypeFeatureMemory",
            "mechanism_status": PROTOTYPE_FEATURE_MEMORY_MECHANISM_STATUS,
            "scientific_promotion_allowed": False,
            "feature_lifecycle": self.feature_lifecycle.to_config(),
            "experiential_memory": self.experiential_memory.to_config(),
            "base_state_builder": self.base_state_builder.to_config(),
            "base_semantics": _BASE_SEMANTICS,
            "observation_semantics": _OBSERVATION_SEMANTICS,
            "key_semantics": _KEY_SEMANTICS,
            "outcome_semantics": _OUTCOME_SEMANTICS,
            "migration_semantics": _MIGRATION_SEMANTICS,
            "memory_clock_advances_per_rebind": 0,
            "rng_draws_per_rebind": 0,
        }

    @property
    def schema_digest(self) -> bytes:
        """Canonical SHA-256 bytes for the exact adapter composition."""

        encoded = json.dumps(
            self._digest_payload(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).digest()

    @property
    def schema_digest_hex(self) -> str:
        """Canonical digest as checkpoint-friendly lowercase hexadecimal."""

        return self.schema_digest.hex()

    def to_config(self) -> dict[str, object]:
        """Return the strict v1 composition record and its digest."""

        return {
            **self._digest_payload(),
            "schema_digest_sha256": self.schema_digest_hex,
        }

    @classmethod
    def from_config(cls, payload: object) -> PrototypeFeatureMemoryConfig:
        """Reconstruct only an exact current-schema composition."""

        if type(payload) is not dict:
            raise TypeError("prototype feature-memory config must be an exact dict")
        raw = cast(dict[str, object], payload)
        expected = {
            "schema",
            "state_schema",
            "type",
            "mechanism_status",
            "scientific_promotion_allowed",
            "feature_lifecycle",
            "experiential_memory",
            "base_state_builder",
            "base_semantics",
            "observation_semantics",
            "key_semantics",
            "outcome_semantics",
            "migration_semantics",
            "memory_clock_advances_per_rebind",
            "rng_draws_per_rebind",
            "schema_digest_sha256",
        }
        if set(raw) != expected:
            missing = sorted(expected - set(raw))
            extra = sorted(set(raw) - expected)
            raise ValueError(
                "prototype feature-memory config fields differ from v1; "
                f"missing={missing}, extra={extra}"
            )
        fixed = {
            "schema": cls.SCHEMA_VERSION,
            "state_schema": PROTOTYPE_FEATURE_MEMORY_STATE_SCHEMA,
            "type": "PrototypeFeatureMemory",
            "mechanism_status": PROTOTYPE_FEATURE_MEMORY_MECHANISM_STATUS,
            "scientific_promotion_allowed": False,
            "base_semantics": _BASE_SEMANTICS,
            "observation_semantics": _OBSERVATION_SEMANTICS,
            "key_semantics": _KEY_SEMANTICS,
            "outcome_semantics": _OUTCOME_SEMANTICS,
            "migration_semantics": _MIGRATION_SEMANTICS,
            "memory_clock_advances_per_rebind": 0,
            "rng_draws_per_rebind": 0,
        }
        if any(raw[name] != value for name, value in fixed.items()):
            raise ValueError("prototype feature-memory fixed semantics differ")
        feature_raw = raw["feature_lifecycle"]
        memory_raw = raw["experiential_memory"]
        builder_raw = raw["base_state_builder"]
        if type(feature_raw) is not dict:
            raise ValueError("feature_lifecycle must be an exact config dict")
        if type(memory_raw) is not dict:
            raise ValueError("experiential_memory must be an exact config dict")
        if type(builder_raw) is not dict:
            raise ValueError("base_state_builder must be an exact config dict")
        if type(builder_raw.get("observation_dim")) is not int:
            raise ValueError("serialized Identity observation_dim must be an exact int")
        result = cls(
            feature_lifecycle=PrototypeFeatureLifecycleConfig.from_config(feature_raw),
            experiential_memory=ExperientialMemoryConfig.from_config(memory_raw),
            base_state_builder=IdentityStateBuilderConfig.from_config(builder_raw),
        )
        if raw["schema_digest_sha256"] != result.schema_digest_hex:
            raise ValueError("prototype feature-memory schema digest differs")
        return result


@chex.dataclass(frozen=True)
class PrototypeFeatureMemoryState:
    """Experiential memory plus its exact current feature-bank identity."""

    memory_state: ExperientialMemoryState
    consumer_binding: PrototypeFeatureConsumerBinding
    schema_digest: UInt[Array, " 32"]


@chex.dataclass(frozen=True)
class PrototypeFeatureMemoryRebindDiagnostics:
    """Fixed-shape audit for one no-op, rejected, or committed rebind."""

    source_state_valid: Bool[Array, ""]
    source_binding_valid: Bool[Array, ""]
    source_binding_matches: Bool[Array, ""]
    destination_binding_valid: Bool[Array, ""]
    destination_descriptors_changed: Bool[Array, ""]
    destination_generation_changed: Bool[Array, ""]
    generation_is_successor: Bool[Array, ""]
    transition_consistent: Bool[Array, ""]
    rebind_required: Bool[Array, ""]
    reencode_attempted: Bool[Array, ""]
    candidate_values_finite: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]
    transaction_noop: Bool[Array, ""]
    valid_rows_reencoded: Int[Array, ""]
    pair_products_evaluated: Int[Array, ""]
    memory_clock_advance_count: Int[Array, ""]
    rng_draw_count: Int[Array, ""]
    source_generation_words: UInt[Array, " 2"]
    requested_generation_words: UInt[Array, " 2"]
    committed_generation_words: UInt[Array, " 2"]
    memory_step_words_before: UInt[Array, " 2"]
    memory_step_words_after: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class PrototypeFeatureMemoryRebindResult:
    """Atomically selected wrapper and complete migration diagnostics."""

    state: PrototypeFeatureMemoryState
    diagnostics: PrototypeFeatureMemoryRebindDiagnostics


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeFeatureMemoryResourceBudget:
    """Exact persistent bytes and history-independent rebind work bound."""

    mechanism_status: str
    scientific_promotion_allowed: bool
    base_feature_dim: int
    active_pair_slots: int
    total_feature_dim: int
    capacity_entries: int
    memory_state_nbytes: int
    consumer_binding_generation_nbytes: int
    consumer_binding_descriptor_nbytes: int
    consumer_binding_nbytes: int
    schema_digest_nbytes: int
    wrapper_metadata_nbytes: int
    wrapper_state_nbytes: int
    max_valid_rows_reencoded: int
    max_pair_products_per_rebind: int
    memory_clock_advances_per_rebind: int
    memory_operation_counter_advances_per_rebind: int
    rng_draws_per_rebind: int

    def to_config(self) -> dict[str, str | int | bool]:
        """Return an exact JSON-compatible resource declaration."""

        return dataclasses.asdict(self)


class PrototypeFeatureMemory:
    """Exact, atomic feature-bank adapter around experiential memory."""

    def __init__(self, config: PrototypeFeatureMemoryConfig):
        if type(config) is not PrototypeFeatureMemoryConfig:
            raise TypeError("config must be an exact PrototypeFeatureMemoryConfig")
        self._config = config
        self._memory = ExperientialMemory(config.experiential_memory)
        self._router = FeatureBankRouter(
            FeatureBankRouterConfig(
                base_dim=config.feature_lifecycle.base_feature_dim,
                active_slots=config.feature_lifecycle.active_pair_slots,
            )
        )
        self._schema_digest = jnp.asarray(
            tuple(config.schema_digest),
            dtype=jnp.uint8,
        )

    @property
    def config(self) -> PrototypeFeatureMemoryConfig:
        """Return the exact static composition."""

        return self._config

    @property
    def memory(self) -> ExperientialMemory:
        """Return the bound memory implementation for ordinary query/write use."""

        return self._memory

    @property
    def schema_digest(self) -> UInt[Array, " 32"]:
        """Return the fixed SHA-256 composition digest as uint8 bytes."""

        return self._schema_digest

    @property
    def schema_digest_hex(self) -> str:
        """Return the checkpoint-friendly composition digest."""

        return self._config.schema_digest_hex

    def to_config(self) -> dict[str, object]:
        """Serialize the strict static composition."""

        return self._config.to_config()

    @classmethod
    def from_config(cls, payload: object) -> PrototypeFeatureMemory:
        """Construct from one strict current-schema record."""

        return cls(PrototypeFeatureMemoryConfig.from_config(payload))

    def _validate_binding_static_contract(
        self,
        binding: PrototypeFeatureConsumerBinding,
        *,
        name: str,
    ) -> None:
        if type(binding) is not PrototypeFeatureConsumerBinding:
            raise TypeError(f"{name} must be an exact PrototypeFeatureConsumerBinding")
        _require_array(
            binding.semantic_generation,
            name=f"{name}.semantic_generation",
            shape=(),
            dtype=jnp.int32,
        )
        _require_array(
            binding.semantic_generation_words,
            name=f"{name}.semantic_generation_words",
            shape=(2,),
            dtype=jnp.uint32,
        )
        _require_array(
            binding.descriptors,
            name=f"{name}.descriptors",
            shape=(self._config.feature_lifecycle.active_pair_slots, 2),
            dtype=jnp.int32,
        )

    def _validate_state_static_contract(self, state: PrototypeFeatureMemoryState) -> None:
        if type(state) is not PrototypeFeatureMemoryState:
            raise TypeError("state must be an exact PrototypeFeatureMemoryState")
        self._memory._validate_state_static_contract(state.memory_state)
        self._validate_binding_static_contract(
            state.consumer_binding,
            name="state.consumer_binding",
        )
        _require_array(
            state.schema_digest,
            name="state.schema_digest",
            shape=(_SCHEMA_DIGEST_NBYTES,),
            dtype=jnp.uint8,
        )

    def _binding_valid(self, binding: PrototypeFeatureConsumerBinding) -> Array:
        validation = self._router.validate_descriptors(binding.descriptors)
        result: Array = (
            (binding.semantic_generation >= jnp.int32(0))
            & (
                binding.semantic_generation
                == _telemetry_from_words(binding.semantic_generation_words)
            )
            & validation.valid
            & jnp.all(validation.live_mask)
        )
        return result

    @staticmethod
    def _bindings_equal(
        left: PrototypeFeatureConsumerBinding,
        right: PrototypeFeatureConsumerBinding,
    ) -> Array:
        return (
            (left.semantic_generation == right.semantic_generation)
            & jnp.all(
                left.semantic_generation_words == right.semantic_generation_words
            )
            & jnp.all(left.descriptors == right.descriptors)
        )

    def _encoded_row_invariants(
        self,
        memory_state: ExperientialMemoryState,
        binding: PrototypeFeatureConsumerBinding,
    ) -> Array:
        entries = memory_state.entries
        valid = entries.valid
        base = self._config.feature_lifecycle.base_feature_dim
        pairs = self._config.feature_lifecycle.active_pair_slots
        safe_left = jnp.clip(binding.descriptors[:, 0], 0, base - 1)
        safe_right = jnp.clip(binding.descriptors[:, 1], 0, base - 1)

        observation_base = entries.observations[:, :base]
        outcome_base = entries.outcomes[:, :base]
        expected_observation_pairs = (
            observation_base[:, safe_left] * observation_base[:, safe_right]
        )
        expected_outcome_pairs = outcome_base[:, safe_left] * outcome_base[:, safe_right]
        stored_observation_pairs = entries.observations[:, base : base + pairs]
        stored_outcome_pairs = entries.outcomes[:, base : base + pairs]

        row_mask = valid[:, None]
        observations_match_keys = jnp.all(
            (~row_mask)
            | (_float32_bits(entries.observations) == _float32_bits(entries.keys))
        )
        observation_pairs_match = jnp.all(
            (~row_mask)
            | (
                _float32_bits(expected_observation_pairs)
                == _float32_bits(stored_observation_pairs)
            )
        )
        outcome_pairs_match = jnp.all(
            (~row_mask)
            | (
                _float32_bits(expected_outcome_pairs)
                == _float32_bits(stored_outcome_pairs)
            )
        )
        outcome_rewards_match = jnp.all(
            (~valid)
            | (
                _float32_bits(entries.outcomes[:, -1])
                == _float32_bits(entries.rewards)
            )
        )
        versions_match = jnp.all(
            (~valid)
            | (entries.representation_versions == binding.semantic_generation)
        )
        products_finite = jnp.all(
            (~row_mask)
            | (
                jnp.isfinite(expected_observation_pairs)
                & jnp.isfinite(expected_outcome_pairs)
            )
        )
        return (
            observations_match_keys
            & observation_pairs_match
            & outcome_pairs_match
            & outcome_rewards_match
            & versions_match
            & products_finite
        )

    def _state_is_valid(
        self,
        state: PrototypeFeatureMemoryState,
        expected_binding: PrototypeFeatureConsumerBinding | None = None,
    ) -> Array:
        expected_matches = (
            jnp.asarray(True, dtype=jnp.bool_)
            if expected_binding is None
            else self._bindings_equal(state.consumer_binding, expected_binding)
        )
        return (
            self._memory._state_is_valid(state.memory_state)
            & self._binding_valid(state.consumer_binding)
            & jnp.all(state.schema_digest == self._schema_digest)
            & self._encoded_row_invariants(
                state.memory_state,
                state.consumer_binding,
            )
            & expected_matches
        )

    def state_valid(
        self,
        state: PrototypeFeatureMemoryState,
        expected_binding: PrototypeFeatureConsumerBinding | None = None,
    ) -> Bool[Array, ""]:
        """Validate structure, digest, exact identity, and every valid row."""

        self._validate_state_static_contract(state)
        if expected_binding is not None:
            self._validate_binding_static_contract(
                expected_binding,
                name="expected_binding",
            )
        return cast(
            Bool[Array, ""],
            self._state_valid_jit(state, expected_binding),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _state_valid_jit(
        self,
        state: PrototypeFeatureMemoryState,
        expected_binding: PrototypeFeatureConsumerBinding | None,
    ) -> Array:
        return self._state_is_valid(state, expected_binding)

    def init(
        self,
        binding: PrototypeFeatureConsumerBinding,
        memory_state: ExperientialMemoryState | None = None,
    ) -> PrototypeFeatureMemoryState:
        """Bind an empty or already correctly encoded memory to one bank."""

        self._validate_binding_static_contract(binding, name="binding")
        measured_memory = self._memory.init() if memory_state is None else memory_state
        self._memory._validate_state_static_contract(measured_memory)
        state = PrototypeFeatureMemoryState(
            memory_state=measured_memory,
            consumer_binding=binding,
            schema_digest=self._schema_digest,
        )
        if not bool(jax.device_get(self.state_valid(state, binding))):
            raise ValueError("initial feature-memory composition is invalid")
        return state

    def _reencoded_memory_candidate(
        self,
        state: ExperientialMemoryState,
        destination_binding: PrototypeFeatureConsumerBinding,
    ) -> tuple[ExperientialMemoryState, Array]:
        entries = state.entries
        valid = entries.valid
        base = self._config.feature_lifecycle.base_feature_dim
        safe_left = jnp.clip(destination_binding.descriptors[:, 0], 0, base - 1)
        safe_right = jnp.clip(destination_binding.descriptors[:, 1], 0, base - 1)

        observation_base = entries.observations[:, :base]
        observation_pairs = (
            observation_base[:, safe_left] * observation_base[:, safe_right]
        )
        encoded_observations = jnp.concatenate(
            (observation_base, observation_pairs),
            axis=1,
        )
        candidate_observations = jnp.where(
            valid[:, None],
            encoded_observations,
            entries.observations,
        )
        # Use the exact same candidate array so valid key/observation payloads
        # cannot diverge through a second floating-point computation.
        candidate_keys = jnp.where(
            valid[:, None],
            candidate_observations,
            entries.keys,
        )

        outcome_base = entries.outcomes[:, :base]
        outcome_pairs = outcome_base[:, safe_left] * outcome_base[:, safe_right]
        encoded_outcomes = jnp.concatenate(
            (outcome_base, outcome_pairs, entries.rewards[:, None]),
            axis=1,
        )
        candidate_outcomes = jnp.where(
            valid[:, None],
            encoded_outcomes,
            entries.outcomes,
        )
        candidate_versions = jnp.where(
            valid,
            destination_binding.semantic_generation,
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
        candidate = cast(
            ExperientialMemoryState,
            state.replace(entries=candidate_entries),
        )
        candidate_values_finite = (
            jnp.all((~valid[:, None]) | jnp.isfinite(observation_pairs))
            & jnp.all((~valid[:, None]) | jnp.isfinite(outcome_pairs))
            & jnp.all((~valid[:, None]) | jnp.isfinite(candidate_observations))
            & jnp.all((~valid[:, None]) | jnp.isfinite(candidate_outcomes))
        )
        return candidate, candidate_values_finite

    def rebind(
        self,
        state: PrototypeFeatureMemoryState,
        source_binding: PrototypeFeatureConsumerBinding,
        destination_binding: PrototypeFeatureConsumerBinding,
    ) -> PrototypeFeatureMemoryRebindResult:
        """Atomically re-encode valid rows for one exact bank successor.

        Dynamic corruption, a stale source binding, an invalid destination,
        an incoherent generation/descriptor transition, or non-finite pair
        products returns the original wrapper bit-for-bit.
        """

        self._validate_state_static_contract(state)
        self._validate_binding_static_contract(source_binding, name="source_binding")
        self._validate_binding_static_contract(
            destination_binding,
            name="destination_binding",
        )
        return cast(
            PrototypeFeatureMemoryRebindResult,
            self._rebind_jit(state, source_binding, destination_binding),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _rebind_jit(
        self,
        state: PrototypeFeatureMemoryState,
        source_binding: PrototypeFeatureConsumerBinding,
        destination_binding: PrototypeFeatureConsumerBinding,
    ) -> PrototypeFeatureMemoryRebindResult:
        source_state_valid = self._state_is_valid(state)
        source_binding_valid = self._binding_valid(source_binding)
        source_binding_matches = self._bindings_equal(
            state.consumer_binding,
            source_binding,
        )
        destination_binding_valid = self._binding_valid(destination_binding)
        descriptors_changed = jnp.any(
            source_binding.descriptors != destination_binding.descriptors
        )
        generation_changed = jnp.any(
            source_binding.semantic_generation_words
            != destination_binding.semantic_generation_words
        )
        generation_is_successor = _words_successor(
            source_binding.semantic_generation_words,
            destination_binding.semantic_generation_words,
        )
        no_change = (~descriptors_changed) & (~generation_changed)
        changed_consistently = (
            descriptors_changed & generation_changed & generation_is_successor
        )
        transition_consistent = no_change | changed_consistently
        prerequisites = (
            source_state_valid
            & source_binding_valid
            & source_binding_matches
            & destination_binding_valid
            & transition_consistent
        )
        rebind_required = prerequisites & changed_consistently
        transaction_noop = prerequisites & no_change

        candidate_memory, candidate_values_finite = (
            self._reencoded_memory_candidate(
                state.memory_state,
                destination_binding,
            )
        )
        candidate = PrototypeFeatureMemoryState(
            memory_state=candidate_memory,
            consumer_binding=destination_binding,
            schema_digest=state.schema_digest,
        )
        candidate_state_valid = self._state_is_valid(
            candidate,
            destination_binding,
        )
        transaction_applied = (
            rebind_required & candidate_values_finite & candidate_state_valid
        )
        selected = jax.lax.cond(
            transaction_applied,
            lambda _: candidate,
            lambda _: state,
            operand=None,
        )
        valid_rows = jnp.sum(
            state.memory_state.entries.valid.astype(jnp.int32),
            dtype=jnp.int32,
        )
        # Observation and key rows share the exact same encoded array, so the
        # kernel evaluates one observation product and one outcome product per
        # slot/row.  Counting the key as a third multiplication would overstate
        # the work performed by `_reencoded_memory_candidate`.
        pair_products = (
            2
            * self._config.feature_lifecycle.active_pair_slots
            * self._config.experiential_memory.capacity
        )
        diagnostics = PrototypeFeatureMemoryRebindDiagnostics(
            source_state_valid=source_state_valid,
            source_binding_valid=source_binding_valid,
            source_binding_matches=source_binding_matches,
            destination_binding_valid=destination_binding_valid,
            destination_descriptors_changed=descriptors_changed,
            destination_generation_changed=generation_changed,
            generation_is_successor=generation_is_successor,
            transition_consistent=transition_consistent,
            rebind_required=rebind_required,
            reencode_attempted=rebind_required,
            candidate_values_finite=candidate_values_finite,
            candidate_state_valid=candidate_state_valid,
            transaction_applied=transaction_applied,
            transaction_noop=transaction_noop,
            valid_rows_reencoded=jnp.where(
                transaction_applied,
                valid_rows,
                jnp.int32(0),
            ),
            pair_products_evaluated=jnp.int32(pair_products),
            memory_clock_advance_count=jnp.int32(0),
            rng_draw_count=jnp.int32(0),
            source_generation_words=source_binding.semantic_generation_words,
            requested_generation_words=destination_binding.semantic_generation_words,
            committed_generation_words=selected.consumer_binding.semantic_generation_words,
            memory_step_words_before=state.memory_state.step_words,
            memory_step_words_after=selected.memory_state.step_words,
        )
        return PrototypeFeatureMemoryRebindResult(
            state=selected,
            diagnostics=diagnostics,
        )

    def resource_budget(self) -> PrototypeFeatureMemoryResourceBudget:
        """Return exact persistent allocation and worst-case rebind work."""

        active = self._config.feature_lifecycle.active_pair_slots
        capacity = self._config.experiential_memory.capacity
        generation_nbytes = 4 + 2 * 4
        descriptor_nbytes = active * 2 * 4
        binding_nbytes = generation_nbytes + descriptor_nbytes
        digest_nbytes = int(self._schema_digest.size) * int(
            self._schema_digest.dtype.itemsize
        )
        memory_nbytes = self._memory.persistent_bytes
        metadata_nbytes = binding_nbytes + digest_nbytes
        wrapper_nbytes = memory_nbytes + metadata_nbytes

        # Cross-check formulas against concrete fixed-shape templates.
        binding_template = PrototypeFeatureConsumerBinding(
            semantic_generation=jnp.int32(0),
            semantic_generation_words=jnp.zeros((2,), dtype=jnp.uint32),
            descriptors=jnp.zeros((active, 2), dtype=jnp.int32),
        )
        wrapper_template = PrototypeFeatureMemoryState(
            memory_state=self._memory.init(),
            consumer_binding=binding_template,
            schema_digest=self._schema_digest,
        )
        if _tree_nbytes(binding_template) != binding_nbytes:
            raise RuntimeError("feature-memory binding byte formula drifted")
        if _tree_nbytes(wrapper_template) != wrapper_nbytes:
            raise RuntimeError("feature-memory wrapper byte formula drifted")
        return PrototypeFeatureMemoryResourceBudget(
            mechanism_status=PROTOTYPE_FEATURE_MEMORY_MECHANISM_STATUS,
            scientific_promotion_allowed=(
                PROTOTYPE_FEATURE_MEMORY_SCIENTIFIC_PROMOTION_ALLOWED
            ),
            base_feature_dim=self._config.feature_lifecycle.base_feature_dim,
            active_pair_slots=active,
            total_feature_dim=self._config.feature_lifecycle.total_feature_dim,
            capacity_entries=capacity,
            memory_state_nbytes=memory_nbytes,
            consumer_binding_generation_nbytes=generation_nbytes,
            consumer_binding_descriptor_nbytes=descriptor_nbytes,
            consumer_binding_nbytes=binding_nbytes,
            schema_digest_nbytes=digest_nbytes,
            wrapper_metadata_nbytes=metadata_nbytes,
            wrapper_state_nbytes=wrapper_nbytes,
            max_valid_rows_reencoded=capacity,
            max_pair_products_per_rebind=2 * active * capacity,
            memory_clock_advances_per_rebind=0,
            memory_operation_counter_advances_per_rebind=0,
            rng_draws_per_rebind=0,
        )


__all__ = [
    "PROTOTYPE_FEATURE_MEMORY_CONFIG_SCHEMA",
    "PROTOTYPE_FEATURE_MEMORY_MECHANISM_STATUS",
    "PROTOTYPE_FEATURE_MEMORY_SCIENTIFIC_PROMOTION_ALLOWED",
    "PROTOTYPE_FEATURE_MEMORY_STATE_SCHEMA",
    "PrototypeFeatureMemory",
    "PrototypeFeatureMemoryConfig",
    "PrototypeFeatureMemoryRebindDiagnostics",
    "PrototypeFeatureMemoryRebindResult",
    "PrototypeFeatureMemoryResourceBudget",
    "PrototypeFeatureMemoryState",
]
