# mypy: disable-error-code="attr-defined,call-arg"
"""Fail-closed deployed representation for compositional feature banks.

``CompositionalFeatureLearner`` reserves an active raw prefix and returns that
prefix from :meth:`constructed_features`, with every value clipped to the
learner's feature safety range.  That is useful inside the learner, but it is
not the representation contract needed by a continual agent: stable base
coordinates must remain byte-exact and must not be duplicated.

This development adapter therefore deploys ``[exact base | composed tail]``.
It also binds the complete active DAG to a monotonically advancing bank
generation and a per-slot birth generation.  Two equal local AST rows remain
different features because slot and birth identity are authoritative; hashes
and descriptor equality are never routing identities.

Stage one deliberately freezes every theta update.  A later integration may
add parameter-revision identity, but silently changing a feature's value while
keeping its consumer identity is forbidden here.  Structural learner updates
are adopted only when the complete public curation trace agrees with the
source, successor, final topology, dependency closure, and exact step
transition.  Persisted rows are rebound as one all-or-nothing transaction:
valid source rows are first reconstructed and bit-authenticated, then all are
re-encoded under exactly one successor bank.

An explicit source-bound prepare/commit boundary lets an outer agent defer
adoption until every routed consumer is ready.  Commit recomputes exactly one
learner update from the captured source and bit-authenticates every
JAX-authoritative learning/binding proposal leaf before applying it.  Legacy
host birth/uptime floats retain the repository-wide non-bit-exact outer-JIT
boundary.  Preparation plus commit therefore costs two logical learner-update
evaluations but advances persistent state, RNG, and clocks at most once.
Prepared updates also capture one dynamic boolean curation permission.  A
false permission consumes the learner's current cadence opportunity without
forming a structural event, allowing callers to cross unsafe consumer
boundaries without rolling back ordinary learning.

This module is development mechanism only.  It is not wired into
``PrototypeAgent``, grants no evidence-promotion authority, and performs no
artifact writes.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core.compositional_features import (
    NUM_OPS,
    OP_RAW,
    CompositionalCurationTrace,
    CompositionalFeatureLearner,
    CompositionalFeatureState,
)

COMPOSITIONAL_FEATURE_ADAPTER_CONFIG_SCHEMA = (
    "alberta.compositional-feature-adapter.config.v1"
)
COMPOSITIONAL_FEATURE_ADAPTER_STATE_SCHEMA = (
    "alberta.compositional-feature-adapter.state.v1"
)
COMPOSITIONAL_FEATURE_ADAPTER_MECHANISM_STATUS = "development_mechanism_only"
COMPOSITIONAL_FEATURE_ADAPTER_SCIENTIFIC_PROMOTION_ALLOWED = False
COMPOSITIONAL_FEATURE_ADAPTER_PREPARED_CURATION_PERMISSION_NBYTES = 1

_CONFIG_TYPE = "CompositionalFeatureAdapter"
_DEPLOYED_REPRESENTATION = "exact_base_then_active_composed_tail"
_IDENTITY_SEMANTICS = "full_bank_topology_plus_slot_and_birth_generation"
_THETA_SEMANTICS = "frozen_until_parameter_revision_identity_exists"
_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1
_MAX_FEATURES = 262_144
_MAX_REENCODE_ROWS = 2**31 - 1


def _strict_int(
    value: object,
    *,
    name: str,
    minimum: int,
    maximum: int = _INT32_MAX,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be a strict integer in [{minimum}, {maximum}]"
        )
    return value


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ValueError("configuration must be canonical finite ASCII JSON") from error


def _exact_json_equal(left: object, right: object) -> bool:
    return _canonical_json_bytes(left) == _canonical_json_bytes(right)


def _digest_array(value: object) -> UInt[Array, " 32"]:
    digest = hashlib.sha256(_canonical_json_bytes(value)).digest()
    return jnp.asarray(np.frombuffer(digest, dtype=np.uint8).copy())


def _theta_bits(theta: Array) -> UInt[Array, "n_features 2"]:
    return jax.lax.bitcast_convert_type(
        jnp.asarray(theta, dtype=jnp.float32),
        jnp.uint32,
    )


def _words_equal(left: Array, right: Array) -> Bool[Array, ""]:
    return jnp.all(left == right)


def _words_less_equal(left: Array, right: Array) -> Bool[Array, ""]:
    return (left[0] < right[0]) | ((left[0] == right[0]) & (left[1] <= right[1]))


def _words_successor(
    words: Array,
) -> tuple[UInt[Array, " 2"], Bool[Array, ""]]:
    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    capacity_available = ~jnp.all(words == maximum)
    low = words[1] + jnp.asarray(1, dtype=jnp.uint32)
    carry = (low == 0).astype(jnp.uint32)
    candidate = jnp.stack((words[0] + carry, low)).astype(jnp.uint32)
    return jnp.where(capacity_available, candidate, words), capacity_available


def _counter_telemetry(words: Array) -> Int[Array, ""]:
    maximum_i32 = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    maximum_u32 = jnp.asarray(_INT32_MAX, dtype=jnp.uint32)
    exact = (words[0] == 0) & (words[1] <= maximum_u32)
    return jnp.where(exact, words[1].astype(jnp.int32), maximum_i32)


def _words_mod_int32(words: Array, modulus: int) -> Int[Array, ""]:
    """Reduce two uint32 words modulo a positive int32 without uint64.

    The four 16-bit limbs are folded with modular doubling.  Both operands of
    each addition are below ``modulus <= INT32_MAX``, so their uint32 sum
    cannot overflow.
    """

    if modulus == 1:
        return jnp.asarray(0, dtype=jnp.int32)
    divisor = jnp.asarray(modulus, dtype=jnp.uint32)
    shift = jnp.asarray(16, dtype=jnp.uint32)
    mask = jnp.asarray(0xFFFF, dtype=jnp.uint32)
    limbs = (
        words[0] >> shift,
        words[0] & mask,
        words[1] >> shift,
        words[1] & mask,
    )

    def add_mod(left: Array, right: Array) -> Array:
        total = left + right
        return jnp.where(total >= divisor, total - divisor, total)

    remainder = jnp.asarray(0, dtype=jnp.uint32)
    for limb in limbs:
        for _ in range(16):
            remainder = add_mod(remainder, remainder)
        remainder = add_mod(remainder, limb % divisor)
    return remainder.astype(jnp.int32)


def _array_contract(value: object, shape: tuple[int, ...], dtype: Any) -> bool:
    return (
        getattr(value, "shape", None) == shape
        and getattr(value, "dtype", None) == jnp.dtype(dtype)
    )


def _curation_permission(value: Array | bool) -> Bool[Array, ""]:
    """Normalize only an exact dynamic boolean scalar."""

    if type(value) is bool:
        return jnp.asarray(value, dtype=jnp.bool_)
    if not _array_contract(value, (), jnp.bool_):
        raise ValueError("curation_allowed must be an exact bool scalar")
    return jnp.asarray(value, dtype=jnp.bool_)


def _leaf_contract(value: object, template: object) -> bool:
    template_shape = getattr(template, "shape", None)
    template_dtype = getattr(template, "dtype", None)
    if template_shape is not None and template_dtype is not None:
        return bool(
            getattr(value, "shape", None) == template_shape
            and getattr(value, "dtype", None) == template_dtype
        )
    if type(template) is float:
        if type(value) is float:
            return math.isfinite(value)
        return bool(
            getattr(value, "shape", None) == ()
            and jnp.issubdtype(
                getattr(value, "dtype", jnp.dtype(jnp.float32)),
                jnp.floating,
            )
        )
    return type(value) is type(template)


def _tree_contract(value: object, template: object) -> bool:
    if str(jax.tree_util.tree_structure(value)) != str(
        jax.tree_util.tree_structure(template)
    ):
        return False
    return all(
        _leaf_contract(actual, expected)
        for actual, expected in zip(
            jax.tree_util.tree_leaves(value),
            jax.tree_util.tree_leaves(template),
            strict=True,
        )
    )


def _tree_nbytes(value: object) -> int:
    total = 0
    for leaf in jax.tree_util.tree_leaves(value):
        shape = getattr(leaf, "shape", None)
        dtype = getattr(leaf, "dtype", None)
        if shape is not None and dtype is not None:
            count = math.prod(shape) if shape else 1
            total += count * int(dtype.itemsize)
        elif type(leaf) is float:
            total += 8
        elif type(leaf) is int:
            total += 8
        elif type(leaf) is bool:
            total += 1
        else:
            raise TypeError(f"unsupported persistent state leaf {type(leaf)!r}")
    return total


def _host_float_metadata_adjustment(state: CompositionalFeatureState) -> int:
    """Keep logical host-float accounting stable across outer JIT conversion."""

    adjustment = 0
    for value in (state.birth_timestamp, state.uptime_s):
        if type(value) is float:
            continue
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        if shape != () or dtype is None or not jnp.issubdtype(dtype, jnp.floating):
            raise TypeError("learner host-float metadata contract drifted")
        adjustment += 8 - int(dtype.itemsize)
    return adjustment


def _learner_state_nbytes(state: CompositionalFeatureState) -> int:
    return _tree_nbytes(state) + _host_float_metadata_adjustment(state)


def _adapter_state_nbytes(state: CompositionalFeatureAdapterState) -> int:
    return (
        _learner_state_nbytes(state.learner_state)
        + _tree_nbytes(state.binding)
    )


def _array_bits_equal(left: object, right: object) -> Bool[Array, ""]:
    """Return exact scalar/array equality, preserving floating-point bits.

    Python scalar metadata in the learner state becomes an array leaf under an
    outer ``jit``.  Normalizing both sides through JAX therefore keeps proposal
    authentication identical in eager and compiled callers.  Typed PRNG keys
    are compared through their public key-data representation.
    """

    try:
        left_dtype = getattr(left, "dtype", None)
        right_dtype = getattr(right, "dtype", None)
        if (
            left_dtype is not None
            and jax.dtypes.issubdtype(left_dtype, jax.dtypes.prng_key)
        ) or (
            right_dtype is not None
            and jax.dtypes.issubdtype(right_dtype, jax.dtypes.prng_key)
        ):
            if left_dtype != right_dtype:
                return jnp.asarray(False, dtype=jnp.bool_)
            return jnp.all(
                jr.key_data(cast(Array, left)) == jr.key_data(cast(Array, right))
            )
    except TypeError:
        return jnp.asarray(False, dtype=jnp.bool_)

    left_array = jnp.asarray(left)
    right_array = jnp.asarray(right)
    if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
        return jnp.asarray(False, dtype=jnp.bool_)
    dtype = left_array.dtype
    if dtype == jnp.dtype(jnp.float32):
        return jnp.all(
            jax.lax.bitcast_convert_type(left_array, jnp.uint32)
            == jax.lax.bitcast_convert_type(right_array, jnp.uint32)
        )
    if dtype in (jnp.dtype(jnp.float16), jnp.dtype(jnp.bfloat16)):
        return jnp.all(
            jax.lax.bitcast_convert_type(left_array, jnp.uint16)
            == jax.lax.bitcast_convert_type(right_array, jnp.uint16)
        )
    if dtype == jnp.dtype(jnp.float64):
        return jnp.all(
            jax.lax.bitcast_convert_type(left_array, jnp.uint64)
            == jax.lax.bitcast_convert_type(right_array, jnp.uint64)
        )
    return jnp.all(left_array == right_array)


def _tree_bits_equal(left: object, right: object) -> Bool[Array, ""]:
    """Bit-authenticate two equal-structure JAX pytrees without host reads."""

    left_structure = jax.tree_util.tree_structure(left)
    right_structure = jax.tree_util.tree_structure(right)
    if str(left_structure) != str(right_structure):
        return jnp.asarray(False, dtype=jnp.bool_)
    valid = jnp.asarray(True, dtype=jnp.bool_)
    for left_leaf, right_leaf in zip(
        jax.tree_util.tree_leaves(left),
        jax.tree_util.tree_leaves(right),
        strict=True,
    ):
        valid &= _array_bits_equal(left_leaf, right_leaf)
    return valid


@chex.dataclass(frozen=True)
class CompositionalFeatureBinding:
    """Full active-bank topology and slot-local lifetime identity."""

    semantic_generation: Int[Array, ""]
    semantic_generation_words: UInt[Array, " 2"]
    ops: Int[Array, " n_features"]
    parent_a: Int[Array, " n_features"]
    parent_b: Int[Array, " n_features"]
    theta_bits: UInt[Array, "n_features 2"]
    depth: Int[Array, " n_features"]
    slot_birth_words: UInt[Array, "n_features 2"]
    schema_digest: UInt[Array, " 32"]


@chex.dataclass(frozen=True)
class CompositionalFeatureAdapterState:
    """Learner state paired with its complete deployed identity binding."""

    learner_state: CompositionalFeatureState
    binding: CompositionalFeatureBinding


@chex.dataclass(frozen=True)
class CompositionalFeatureAdapterDiagnostics:
    """Authentication facts for one learner-step adoption."""

    source_state_valid: Bool[Array, ""]
    learner_step_committed: Bool[Array, ""]
    trace_valid: Bool[Array, ""]
    topology_change_authenticated: Bool[Array, ""]
    identity_capacity_available: Bool[Array, ""]
    active_bank_changed: Bool[Array, ""]
    active_change_mask: Bool[Array, " n_features"]
    transaction_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class CompositionalFeatureAdapterUpdateResult:
    """Atomic adapter result around one compositional learner update."""

    state: CompositionalFeatureAdapterState
    predictions: Float[Array, " n_tasks"]
    errors: Float[Array, " n_tasks"]
    metrics: Float[Array, " 7"]
    curation_trace: CompositionalCurationTrace
    diagnostics: CompositionalFeatureAdapterDiagnostics


@chex.dataclass(frozen=True)
class CompositionalFeatureAdapterPreparedUpdate:
    """Pure source-bound learner proposal awaiting all consumer transactions.

    The complete source is retained so commit can reject a stale destination.
    Commit also recomputes this proposal from the source observation, targets,
    and context and bit-authenticates the entire result.  The proposal is
    transient; it adds no persistent learner or adapter state.  Its captured
    curation permission is one exact boolean byte.
    """

    source_state: CompositionalFeatureAdapterState
    observation: Float[Array, " base_feature_dim"]
    targets: Float[Array, " n_tasks"]
    context_id: Int[Array, ""]
    curation_allowed: Bool[Array, ""]
    candidate_state: CompositionalFeatureAdapterState
    predictions: Float[Array, " n_tasks"]
    errors: Float[Array, " n_tasks"]
    metrics: Float[Array, " 7"]
    curation_trace: CompositionalCurationTrace
    diagnostics: CompositionalFeatureAdapterDiagnostics
    preparation_learner_update_evaluations: Int[Array, ""]


@chex.dataclass(frozen=True)
class CompositionalFeatureAdapterCommitDiagnostics:
    """Commit-time source, proposal, consumer, and exact-work facts."""

    proposal_integrity: Bool[Array, ""]
    source_matches: Bool[Array, ""]
    destination_state_valid: Bool[Array, ""]
    proposal_valid: Bool[Array, ""]
    consumers_ready: Bool[Array, ""]
    applied: Bool[Array, ""]
    rejected: Bool[Array, ""]
    preparation_learner_update_evaluations: Int[Array, ""]
    commit_recomputed_learner_update_evaluations: Int[Array, ""]
    total_learner_update_evaluations: Int[Array, ""]


@chex.dataclass(frozen=True)
class CompositionalFeatureAdapterCommitResult:
    """Atomic prepared-update commit result convenient as a scan carry."""

    state: CompositionalFeatureAdapterState
    diagnostics: CompositionalFeatureAdapterCommitDiagnostics


@chex.dataclass(frozen=True)
class CompositionalFeatureReencodeDiagnostics:
    """Checks performed by one fixed-work row-rebinding transaction."""

    source_state_valid: Bool[Array, ""]
    destination_state_valid: Bool[Array, ""]
    bindings_same: Bool[Array, ""]
    generation_is_successor: Bool[Array, ""]
    learner_step_is_successor: Bool[Array, ""]
    transition_consistent: Bool[Array, ""]
    source_rows_match: Bool[Array, ""]
    candidate_values_finite: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]
    transaction_noop: Bool[Array, ""]
    valid_rows_reencoded: Int[Array, ""]
    feature_slot_evaluations: Int[Array, ""]


@chex.dataclass(frozen=True)
class CompositionalFeatureReencodeResult:
    """Rows and diagnostics from an atomic source-to-successor re-encode."""

    values: Float[Array, "rows n_features"]
    diagnostics: CompositionalFeatureReencodeDiagnostics


@dataclasses.dataclass(frozen=True)
class CompositionalFeatureAdapterResourceBudget:
    """Exact persistent bytes and fixed replay work for one configured state."""

    schema: str
    mechanism_status: str
    scientific_promotion_allowed: bool
    base_feature_dim: int
    n_features: int
    dynamic_feature_slots: int
    candidate_slots: int
    n_tasks: int
    learner_persistent_nbytes: int
    binding_persistent_nbytes: int
    total_persistent_state_nbytes: int
    max_reencode_rows: int
    max_reencode_feature_slot_evaluations: int


class CompositionalFeatureAdapter:
    """Deploy and authenticate one fixed-width compositional feature bank."""

    def __init__(
        self,
        learner: CompositionalFeatureLearner,
        *,
        base_feature_dim: int,
    ) -> None:
        if type(learner) is not CompositionalFeatureLearner:
            raise TypeError("learner must be the exact CompositionalFeatureLearner class")
        base_feature_dim = _strict_int(
            base_feature_dim,
            name="base_feature_dim",
            minimum=1,
            maximum=_MAX_FEATURES - 1,
        )
        learner_config = learner.to_config()
        n_features = _strict_int(
            learner_config["n_features"],
            name="learner.n_features",
            minimum=1,
            maximum=_MAX_FEATURES,
        )
        if n_features <= base_feature_dim:
            raise ValueError("learner must expose at least one composed slot")
        if type(learner_config["step_size_theta"]) is not float or (
            learner_config["step_size_theta"] != 0.0
        ):
            raise ValueError("step_size_theta must be the exact frozen value 0.0")
        if learner_config["train_candidate_theta"] is not False:
            raise ValueError("train_candidate_theta must be False while theta is frozen")

        self._learner = learner
        self._base_feature_dim = base_feature_dim
        self._n_features = n_features
        self._n_tasks = int(learner_config["n_tasks"])
        self._candidate_count = int(learner_config["candidate_count"])
        self._max_depth = int(learner_config["max_depth"])
        self._replacement_interval = int(learner_config["replacement_interval"])
        self._learn_generator_resources = bool(
            learner_config["learn_generator_resources"]
        )
        self._learner_config = learner_config
        self._config = {
            "schema": COMPOSITIONAL_FEATURE_ADAPTER_CONFIG_SCHEMA,
            "type": _CONFIG_TYPE,
            "state_schema": COMPOSITIONAL_FEATURE_ADAPTER_STATE_SCHEMA,
            "mechanism_status": COMPOSITIONAL_FEATURE_ADAPTER_MECHANISM_STATUS,
            "scientific_promotion_allowed": (
                COMPOSITIONAL_FEATURE_ADAPTER_SCIENTIFIC_PROMOTION_ALLOWED
            ),
            "base_feature_dim": base_feature_dim,
            "deployed_representation": _DEPLOYED_REPRESENTATION,
            "identity_semantics": _IDENTITY_SEMANTICS,
            "theta_semantics": _THETA_SEMANTICS,
            "learner": learner_config,
        }
        self._schema_digest = _digest_array(self._config)
        self._learner_template = learner.init(base_feature_dim, jr.key(0))

    @property
    def learner(self) -> CompositionalFeatureLearner:
        """Exact learner whose steps this adapter authenticates."""

        return self._learner

    @property
    def base_feature_dim(self) -> int:
        """Stable, byte-exact deployed prefix width."""

        return self._base_feature_dim

    @property
    def n_features(self) -> int:
        """Total deployed representation width."""

        return self._n_features

    @property
    def dynamic_feature_slots(self) -> int:
        """Number of composed deployed tail slots."""

        return self._n_features - self._base_feature_dim

    def to_config(self) -> dict[str, Any]:
        """Return the exact static adapter and learner contract."""

        return cast(
            dict[str, Any],
            json.loads(_canonical_json_bytes(self._config).decode("ascii")),
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> CompositionalFeatureAdapter:
        """Reconstruct only an exact, non-default-expanded configuration."""

        if type(config) is not dict:
            raise ValueError("adapter configuration must be an exact dictionary")
        payload = dict(config)
        expected_keys = {
            "schema",
            "type",
            "state_schema",
            "mechanism_status",
            "scientific_promotion_allowed",
            "base_feature_dim",
            "deployed_representation",
            "identity_semantics",
            "theta_semantics",
            "learner",
        }
        if set(payload) != expected_keys:
            raise ValueError("adapter configuration field manifest is not exact")
        fixed = {
            "schema": COMPOSITIONAL_FEATURE_ADAPTER_CONFIG_SCHEMA,
            "type": _CONFIG_TYPE,
            "state_schema": COMPOSITIONAL_FEATURE_ADAPTER_STATE_SCHEMA,
            "mechanism_status": COMPOSITIONAL_FEATURE_ADAPTER_MECHANISM_STATUS,
            "scientific_promotion_allowed": False,
            "deployed_representation": _DEPLOYED_REPRESENTATION,
            "identity_semantics": _IDENTITY_SEMANTICS,
            "theta_semantics": _THETA_SEMANTICS,
        }
        if any(not _exact_json_equal(payload[name], value) for name, value in fixed.items()):
            raise ValueError("adapter configuration fixed disclosure drifted")
        learner_payload = payload["learner"]
        if type(learner_payload) is not dict:
            raise ValueError("learner configuration must be an exact dictionary")
        try:
            learner = CompositionalFeatureLearner.from_config(learner_payload)
        except (TypeError, ValueError, KeyError) as error:
            raise ValueError("learner configuration is invalid") from error
        if not _exact_json_equal(learner.to_config(), learner_payload):
            raise ValueError("learner configuration field manifest is not exact")
        adapter = cls(learner, base_feature_dim=payload["base_feature_dim"])
        if not _exact_json_equal(adapter.to_config(), payload):
            raise ValueError("adapter configuration does not round-trip exactly")
        return adapter

    def _binding_from_state(
        self,
        state: CompositionalFeatureState,
        *,
        generation_words: Array,
        slot_birth_words: Array,
    ) -> CompositionalFeatureBinding:
        return CompositionalFeatureBinding(
            semantic_generation=_counter_telemetry(generation_words),
            semantic_generation_words=jnp.asarray(
                generation_words, dtype=jnp.uint32
            ),
            ops=state.ops,
            parent_a=state.parent_a,
            parent_b=state.parent_b,
            theta_bits=_theta_bits(state.theta),
            depth=state.depth,
            slot_birth_words=jnp.asarray(slot_birth_words, dtype=jnp.uint32),
            schema_digest=self._schema_digest,
        )

    def init(self, key: Array) -> CompositionalFeatureAdapterState:
        """Initialize a zero-generation authenticated learner bank."""

        if getattr(key, "shape", None) != () or not jax.dtypes.issubdtype(
            getattr(key, "dtype", None),
            jax.dtypes.prng_key,
        ):
            raise ValueError("key must be a scalar typed PRNG key")
        learner_state = self._learner.init(self._base_feature_dim, key)
        zeros = jnp.zeros((self._n_features, 2), dtype=jnp.uint32)
        state = CompositionalFeatureAdapterState(
            learner_state=learner_state,
            binding=self._binding_from_state(
                learner_state,
                generation_words=jnp.zeros((2,), dtype=jnp.uint32),
                slot_birth_words=zeros,
            ),
        )
        if not bool(self.state_valid(state)):
            raise RuntimeError("learner produced an invalid initial compositional bank")
        return state

    def rebind_pristine_state(
        self,
        learner_state: CompositionalFeatureState,
    ) -> CompositionalFeatureAdapterState:
        """Bind a host-constructed genesis state for testing and migration.

        Only exact zero-step state is accepted.  This cannot relabel an active
        lifetime or bypass the authenticated :meth:`update` transition.
        """

        if not _tree_contract(learner_state, self._learner_template):
            raise ValueError("pristine learner state has the wrong tree contract")
        if int(learner_state.step_count) != 0 or bool(jnp.any(learner_state.step_words)):
            raise ValueError("pristine learner state must have zero learner lifetime")
        if int(learner_state.replacement_phase) != 0:
            raise ValueError("pristine learner state must have zero replacement phase")
        zeros = jnp.zeros((self._n_features, 2), dtype=jnp.uint32)
        state = CompositionalFeatureAdapterState(
            learner_state=learner_state,
            binding=self._binding_from_state(
                learner_state,
                generation_words=jnp.zeros((2,), dtype=jnp.uint32),
                slot_birth_words=zeros,
            ),
        )
        if not bool(self.state_valid(state)):
            raise ValueError("pristine learner state violates adapter topology")
        return state

    def _binding_contract(self, binding: object) -> bool:
        if type(binding) is not CompositionalFeatureBinding:
            return False
        return all(
            (
                _array_contract(binding.semantic_generation, (), jnp.int32),
                _array_contract(binding.semantic_generation_words, (2,), jnp.uint32),
                _array_contract(binding.ops, (self._n_features,), jnp.int32),
                _array_contract(binding.parent_a, (self._n_features,), jnp.int32),
                _array_contract(binding.parent_b, (self._n_features,), jnp.int32),
                _array_contract(
                    binding.theta_bits, (self._n_features, 2), jnp.uint32
                ),
                _array_contract(binding.depth, (self._n_features,), jnp.int32),
                _array_contract(
                    binding.slot_birth_words,
                    (self._n_features, 2),
                    jnp.uint32,
                ),
                _array_contract(binding.schema_digest, (32,), jnp.uint8),
            )
        )

    def _learner_state_contract(self, state: object) -> bool:
        return type(state) is CompositionalFeatureState and _tree_contract(
            state,
            self._learner_template,
        )

    def _active_topology_valid(
        self,
        state: CompositionalFeatureState,
    ) -> Bool[Array, ""]:
        valid = jnp.asarray(True, dtype=jnp.bool_)
        raw_indices = jnp.arange(self._base_feature_dim, dtype=jnp.int32)
        valid &= jnp.all(state.ops[: self._base_feature_dim] == OP_RAW)
        valid &= jnp.all(state.parent_a[: self._base_feature_dim] == raw_indices)
        valid &= jnp.all(state.parent_b[: self._base_feature_dim] == -1)
        valid &= jnp.all(state.depth[: self._base_feature_dim] == 0)
        valid &= jnp.all(_theta_bits(state.theta[: self._base_feature_dim]) == 0)
        for slot in range(self._base_feature_dim, self._n_features):
            parent_a = state.parent_a[slot]
            parent_b = state.parent_b[slot]
            safe_a = jnp.clip(parent_a, 0, slot - 1)
            safe_b = jnp.clip(parent_b, 0, slot - 1)
            expected_depth = jnp.maximum(
                state.depth[safe_a],
                state.depth[safe_b],
            ) + 1
            valid &= (state.ops[slot] > OP_RAW) & (state.ops[slot] < NUM_OPS)
            valid &= (parent_a >= 0) & (parent_a < slot)
            valid &= (parent_b >= 0) & (parent_b < slot)
            valid &= state.depth[slot] == expected_depth
            valid &= (state.depth[slot] > 0) & (
                state.depth[slot] <= self._max_depth
            )
        return valid

    def _counter_state_valid(
        self,
        state: CompositionalFeatureState,
    ) -> Bool[Array, ""]:
        words = state.step_words
        expected = _counter_telemetry(words)
        if self._replacement_interval == 0 or self._learn_generator_resources:
            phase_valid = state.replacement_phase == 0
        else:
            phase_valid = state.replacement_phase == _words_mod_int32(
                words,
                self._replacement_interval,
            )
        expected_manager_step = jnp.where(
            self._learn_generator_resources,
            state.step_count,
            jnp.asarray(0, dtype=jnp.int32),
        )
        return (
            (state.step_count >= 0)
            & (state.step_count == expected)
            & phase_valid
            & (
                state.generator_resource_state.step_count
                == expected_manager_step
            )
            & jnp.all(state.ages >= 0)
            & jnp.all(state.candidate_ages >= 0)
        )

    def _static_contract_valid(self) -> bool:
        try:
            return type(self._learner) is CompositionalFeatureLearner and (
                _exact_json_equal(self._learner.to_config(), self._learner_config)
            )
        except (TypeError, ValueError, UnicodeEncodeError):
            return False

    def state_valid(
        self,
        state: CompositionalFeatureAdapterState,
    ) -> Bool[Array, ""]:
        """Validate state, full binding, topology, counters, and birth history."""

        if type(state) is not CompositionalFeatureAdapterState:
            return jnp.asarray(False, dtype=jnp.bool_)
        if not self._static_contract_valid():
            return jnp.asarray(False, dtype=jnp.bool_)
        if not self._learner_state_contract(state.learner_state):
            return jnp.asarray(False, dtype=jnp.bool_)
        if not self._binding_contract(state.binding):
            return jnp.asarray(False, dtype=jnp.bool_)
        learner_state = state.learner_state
        binding = state.binding
        ranking_valid = self._learner.ranking_diagnostics(
            learner_state,
            self._base_feature_dim,
        ).contract_valid
        valid = ranking_valid & self._active_topology_valid(learner_state)
        valid &= self._counter_state_valid(learner_state)
        valid &= binding.semantic_generation == _counter_telemetry(
            binding.semantic_generation_words
        )
        valid &= jnp.all(binding.ops == learner_state.ops)
        valid &= jnp.all(binding.parent_a == learner_state.parent_a)
        valid &= jnp.all(binding.parent_b == learner_state.parent_b)
        valid &= jnp.all(binding.theta_bits == _theta_bits(learner_state.theta))
        valid &= jnp.all(binding.depth == learner_state.depth)
        valid &= jnp.all(binding.schema_digest == self._schema_digest)
        valid &= jnp.all(binding.slot_birth_words[: self._base_feature_dim] == 0)
        for slot in range(self._base_feature_dim, self._n_features):
            valid &= _words_less_equal(
                binding.slot_birth_words[slot],
                binding.semantic_generation_words,
            )
        generation_is_zero = _words_equal(
            binding.semantic_generation_words,
            jnp.zeros((2,), dtype=jnp.uint32),
        )
        latest_birth_present = jnp.any(
            jnp.all(
                binding.slot_birth_words[self._base_feature_dim :]
                == binding.semantic_generation_words[None, :],
                axis=1,
            )
        )
        valid &= generation_is_zero | latest_birth_present
        return cast(Array, valid)

    def _representation_unchecked(
        self,
        state: CompositionalFeatureState,
        observation: Array,
    ) -> Float[Array, " n_features"]:
        constructed = self._learner.constructed_features(state, observation)
        return jnp.concatenate(
            (observation, constructed[self._base_feature_dim :]),
            axis=0,
        )

    def representation(
        self,
        state: CompositionalFeatureAdapterState,
        observation: Array,
    ) -> Float[Array, " n_features"]:
        """Return ``[exact base | composed tail]`` or fail closed to zeros."""

        if not _array_contract(
            observation,
            (self._base_feature_dim,),
            jnp.float32,
        ):
            raise ValueError("observation must be exact float32 stable-base shape")
        if not self._learner_state_contract(state.learner_state):
            return jnp.zeros((self._n_features,), dtype=jnp.float32)
        candidate = self._representation_unchecked(state.learner_state, observation)
        valid = self.state_valid(state) & jnp.all(jnp.isfinite(observation))
        valid &= jnp.all(jnp.isfinite(candidate))
        return jnp.where(valid, candidate, jnp.zeros_like(candidate))

    def dynamic_slot_identity(
        self,
        state: CompositionalFeatureAdapterState,
        slot: int,
    ) -> tuple[object, ...]:
        """Return full host identity without reducing the bank to an AST hash."""

        slot = _strict_int(
            slot,
            name="slot",
            minimum=self._base_feature_dim,
            maximum=self._n_features - 1,
        )
        if not bool(self.state_valid(state)):
            raise ValueError("cannot inspect identity of invalid adapter state")
        binding = state.binding
        return (
            slot,
            int(binding.slot_birth_words[slot, 0]),
            int(binding.slot_birth_words[slot, 1]),
            tuple(int(value) for value in np.asarray(binding.ops)),
            tuple(int(value) for value in np.asarray(binding.parent_a)),
            tuple(int(value) for value in np.asarray(binding.parent_b)),
            tuple(
                tuple(int(word) for word in row)
                for row in np.asarray(binding.theta_bits)
            ),
            tuple(int(value) for value in np.asarray(binding.depth)),
            tuple(
                tuple(int(word) for word in row)
                for row in np.asarray(binding.slot_birth_words)
            ),
        )

    def _descriptor_change_mask(
        self,
        source: CompositionalFeatureState,
        destination: CompositionalFeatureState,
    ) -> Bool[Array, " n_features"]:
        return (
            (source.ops != destination.ops)
            | (source.parent_a != destination.parent_a)
            | (source.parent_b != destination.parent_b)
            | jnp.any(_theta_bits(source.theta) != _theta_bits(destination.theta), axis=1)
            | (source.depth != destination.depth)
        )

    def _dependency_closure_valid(
        self,
        destination: CompositionalFeatureState,
        active_change_mask: Array,
    ) -> Bool[Array, ""]:
        valid = ~jnp.any(active_change_mask[: self._base_feature_dim])
        for slot in range(self._base_feature_dim, self._n_features):
            safe_a = jnp.clip(destination.parent_a[slot], 0, slot - 1)
            safe_b = jnp.clip(destination.parent_b[slot], 0, slot - 1)
            parent_changed = active_change_mask[safe_a] | active_change_mask[safe_b]
            valid &= (~parent_changed) | active_change_mask[slot]
        return valid

    def _trace_valid(
        self,
        source: CompositionalFeatureState,
        destination: CompositionalFeatureState,
        trace: CompositionalCurationTrace,
        expected_step_words: Array,
        learner_capacity_available: Array,
    ) -> Bool[Array, ""]:
        active_change_mask = trace.active_change_mask
        descriptor_change = self._descriptor_change_mask(source, destination)
        valid = (trace.pre_step == source.step_count) & (
            trace.post_step == destination.step_count
        )
        valid &= _words_equal(trace.pre_step_words, source.step_words)
        valid &= _words_equal(trace.post_step_words, destination.step_words)
        valid &= _words_equal(destination.step_words, expected_step_words)
        valid &= trace.pre_replacement_phase == source.replacement_phase
        valid &= trace.post_replacement_phase == destination.replacement_phase
        valid &= trace.lifetime_counter_valid
        valid &= trace.lifetime_capacity_available == learner_capacity_available
        valid &= jnp.all(trace.cascade_final_ops == destination.ops)
        valid &= jnp.all(trace.cascade_final_parent_a == destination.parent_a)
        valid &= jnp.all(trace.cascade_final_parent_b == destination.parent_b)
        valid &= jnp.all(
            _theta_bits(trace.cascade_final_theta) == _theta_bits(destination.theta)
        )
        valid &= jnp.all(trace.cascade_final_depth == destination.depth)
        valid &= jnp.all(
            active_change_mask == (trace.root_change_mask | trace.cascade_refill_mask)
        )
        valid &= jnp.all((~descriptor_change) | active_change_mask)
        valid &= self._dependency_closure_valid(destination, active_change_mask)
        valid &= (~jnp.any(active_change_mask)) | trace.has_event
        return valid

    def update(
        self,
        state: CompositionalFeatureAdapterState,
        observation: Array,
        targets: Array,
        context_id: Array | int = 0,
        curation_allowed: Array | bool = True,
    ) -> CompositionalFeatureAdapterUpdateResult:
        """Authenticate and atomically adopt one exact learner successor."""

        if not _array_contract(
            observation,
            (self._base_feature_dim,),
            jnp.float32,
        ):
            raise ValueError("observation must be exact float32 stable-base shape")
        if not _array_contract(targets, (self._n_tasks,), jnp.float32):
            raise ValueError("targets must be exact float32 learner-task shape")
        if not self._learner_state_contract(state.learner_state):
            raise ValueError("learner state tree contract is not safe to execute")
        allow_curation = _curation_permission(curation_allowed)

        source_valid = self.state_valid(state)
        raw_result = self._learner.update(
            state.learner_state,
            observation,
            targets,
            context_id=context_id,
            curation_allowed=allow_curation,
        )
        expected_step_words, learner_capacity = _words_successor(
            state.learner_state.step_words
        )
        learner_step_committed = learner_capacity & _words_equal(
            raw_result.state.step_words,
            expected_step_words,
        )
        learner_step_committed &= raw_result.state.step_count == _counter_telemetry(
            expected_step_words
        )
        trace_valid = self._trace_valid(
            state.learner_state,
            raw_result.state,
            raw_result.curation_trace,
            expected_step_words,
            learner_capacity,
        )
        active_change_mask = raw_result.curation_trace.active_change_mask
        active_bank_changed = jnp.any(active_change_mask)
        next_generation_words, generation_capacity = _words_successor(
            state.binding.semantic_generation_words
        )
        identity_capacity = (~active_bank_changed) | generation_capacity
        adopted_generation_words = jnp.where(
            active_bank_changed,
            next_generation_words,
            state.binding.semantic_generation_words,
        )
        adopted_births = jnp.where(
            active_change_mask[:, None],
            adopted_generation_words[None, :],
            state.binding.slot_birth_words,
        ).astype(jnp.uint32)
        candidate_binding = self._binding_from_state(
            raw_result.state,
            generation_words=adopted_generation_words,
            slot_birth_words=adopted_births,
        )
        candidate_state = CompositionalFeatureAdapterState(
            learner_state=raw_result.state,
            binding=candidate_binding,
        )
        topology_change_authenticated = (
            self._dependency_closure_valid(raw_result.state, active_change_mask)
            & (~jnp.any(active_change_mask[: self._base_feature_dim]))
        )
        transaction_applied = (
            source_valid
            & learner_step_committed
            & trace_valid
            & topology_change_authenticated
            & identity_capacity
            & self.state_valid(candidate_state)
            & jnp.all(jnp.isfinite(observation))
            & jnp.all(jnp.isnan(targets) | jnp.isfinite(targets))
        )
        committed_state = jax.lax.cond(
            transaction_applied,
            lambda: candidate_state,
            lambda: state,
        )
        diagnostics = CompositionalFeatureAdapterDiagnostics(
            source_state_valid=source_valid,
            learner_step_committed=learner_step_committed,
            trace_valid=trace_valid,
            topology_change_authenticated=topology_change_authenticated,
            identity_capacity_available=identity_capacity,
            active_bank_changed=active_bank_changed & transaction_applied,
            active_change_mask=active_change_mask & transaction_applied,
            transaction_applied=transaction_applied,
        )
        return CompositionalFeatureAdapterUpdateResult(
            state=committed_state,
            predictions=jnp.where(
                transaction_applied,
                raw_result.predictions,
                jnp.full_like(raw_result.predictions, jnp.nan),
            ),
            errors=jnp.where(
                transaction_applied,
                raw_result.errors,
                jnp.full_like(raw_result.errors, jnp.nan),
            ),
            metrics=jnp.where(
                transaction_applied,
                raw_result.metrics,
                jnp.full_like(raw_result.metrics, jnp.nan),
            ),
            curation_trace=raw_result.curation_trace,
            diagnostics=diagnostics,
        )

    def _prepared_context(self, context_id: Array | int) -> Int[Array, ""]:
        if type(context_id) is int:
            context = _strict_int(
                context_id,
                name="context_id",
                minimum=0,
            )
            return jnp.asarray(context, dtype=jnp.int32)
        if not _array_contract(context_id, (), jnp.int32):
            raise ValueError("context_id must be an exact int32 scalar")
        return jnp.asarray(context_id, dtype=jnp.int32)

    def prepare_update(
        self,
        state: CompositionalFeatureAdapterState,
        observation: Array,
        targets: Array,
        context_id: Array | int = 0,
        curation_allowed: Array | bool = True,
    ) -> CompositionalFeatureAdapterPreparedUpdate:
        """Form but do not adopt one source-bound compositional update.

        The candidate can be used to prepare row re-encoding and consumer
        routing.  Adoption remains deferred until every required consumer is
        ready and :meth:`commit_prepared_update` authenticates the proposal.
        """

        context = self._prepared_context(context_id)
        allow_curation = _curation_permission(curation_allowed)
        result = self.update(
            state,
            observation,
            targets,
            context_id=context,
            curation_allowed=allow_curation,
        )
        return CompositionalFeatureAdapterPreparedUpdate(
            source_state=state,
            observation=observation,
            targets=targets,
            context_id=context,
            curation_allowed=allow_curation,
            candidate_state=result.state,
            predictions=result.predictions,
            errors=result.errors,
            metrics=result.metrics,
            curation_trace=result.curation_trace,
            diagnostics=result.diagnostics,
            preparation_learner_update_evaluations=jnp.asarray(1, dtype=jnp.int32),
        )

    def _prepared_update_contract(
        self,
        proposal: CompositionalFeatureAdapterPreparedUpdate,
    ) -> None:
        if type(proposal) is not CompositionalFeatureAdapterPreparedUpdate:
            raise TypeError(
                "proposal must be an exact CompositionalFeatureAdapterPreparedUpdate"
            )
        if not self._learner_state_contract(proposal.source_state.learner_state):
            raise ValueError("proposal source learner tree contract is unsafe")
        if not self._learner_state_contract(proposal.candidate_state.learner_state):
            raise ValueError("proposal candidate learner tree contract is unsafe")
        if not _array_contract(
            proposal.observation,
            (self._base_feature_dim,),
            jnp.float32,
        ):
            raise ValueError("proposal observation contract drifted")
        if not _array_contract(proposal.targets, (self._n_tasks,), jnp.float32):
            raise ValueError("proposal targets contract drifted")
        if not _array_contract(proposal.context_id, (), jnp.int32):
            raise ValueError("proposal context_id contract drifted")
        if not _array_contract(proposal.curation_allowed, (), jnp.bool_):
            raise ValueError("proposal curation_allowed contract drifted")
        if not _array_contract(proposal.predictions, (self._n_tasks,), jnp.float32):
            raise ValueError("proposal predictions contract drifted")
        if not _array_contract(proposal.errors, (self._n_tasks,), jnp.float32):
            raise ValueError("proposal errors contract drifted")
        if not _array_contract(proposal.metrics, (7,), jnp.float32):
            raise ValueError("proposal metrics contract drifted")
        if not _array_contract(
            proposal.preparation_learner_update_evaluations,
            (),
            jnp.int32,
        ):
            raise ValueError("proposal work accounting contract drifted")

    def commit_prepared_update(
        self,
        destination_state: CompositionalFeatureAdapterState,
        proposal: CompositionalFeatureAdapterPreparedUpdate,
        *,
        consumers_ready: Array,
    ) -> CompositionalFeatureAdapterCommitResult:
        """Recompute and atomically adopt an exact current prepared update.

        Commit intentionally performs one additional learner update evaluation
        from the captured source.  This fixed recomputation authenticates every
        JAX-authoritative proposal leaf without advancing persistent RNG or
        clocks twice.  Legacy host birth/uptime floats remain non-authoritative.
        Any unavailable consumer, stale source, corrupt proposal, invalid state,
        or exhausted lifetime leaves the destination unchanged.
        """

        if type(destination_state) is not CompositionalFeatureAdapterState or not (
            self._learner_state_contract(destination_state.learner_state)
        ):
            raise ValueError("destination adapter state tree contract is unsafe")
        self._prepared_update_contract(proposal)
        if not _array_contract(consumers_ready, (), jnp.bool_):
            raise ValueError("consumers_ready must be an exact bool scalar")

        expected = self.prepare_update(
            proposal.source_state,
            proposal.observation,
            proposal.targets,
            context_id=proposal.context_id,
            curation_allowed=proposal.curation_allowed,
        )
        proposal_integrity = _tree_bits_equal(proposal, expected)
        source_matches = _tree_bits_equal(destination_state, proposal.source_state)
        destination_state_valid = self.state_valid(destination_state)
        proposal_valid = expected.diagnostics.transaction_applied
        applied = (
            proposal_integrity
            & source_matches
            & destination_state_valid
            & proposal_valid
            & consumers_ready
        )
        next_state = cast(
            CompositionalFeatureAdapterState,
            jax.lax.cond(
                applied,
                lambda: expected.candidate_state,
                lambda: destination_state,
            ),
        )
        prepare_work = jnp.asarray(1, dtype=jnp.int32)
        commit_work = jnp.asarray(1, dtype=jnp.int32)
        diagnostics = CompositionalFeatureAdapterCommitDiagnostics(
            proposal_integrity=proposal_integrity,
            source_matches=source_matches,
            destination_state_valid=destination_state_valid,
            proposal_valid=proposal_valid,
            consumers_ready=consumers_ready,
            applied=applied,
            rejected=~applied,
            preparation_learner_update_evaluations=prepare_work,
            commit_recomputed_learner_update_evaluations=commit_work,
            total_learner_update_evaluations=prepare_work + commit_work,
        )
        return CompositionalFeatureAdapterCommitResult(
            state=next_state,
            diagnostics=diagnostics,
        )

    def _bindings_equal(
        self,
        left: CompositionalFeatureBinding,
        right: CompositionalFeatureBinding,
    ) -> Bool[Array, ""]:
        return jnp.all(
            jnp.asarray(
                (
                    _words_equal(
                        left.semantic_generation_words,
                        right.semantic_generation_words,
                    ),
                    left.semantic_generation == right.semantic_generation,
                    jnp.all(left.ops == right.ops),
                    jnp.all(left.parent_a == right.parent_a),
                    jnp.all(left.parent_b == right.parent_b),
                    jnp.all(left.theta_bits == right.theta_bits),
                    jnp.all(left.depth == right.depth),
                    jnp.all(left.slot_birth_words == right.slot_birth_words),
                    jnp.all(left.schema_digest == right.schema_digest),
                ),
                dtype=jnp.bool_,
            )
        )

    def _successor_binding_valid(
        self,
        source: CompositionalFeatureAdapterState,
        destination: CompositionalFeatureAdapterState,
    ) -> Bool[Array, ""]:
        expected_generation, capacity = _words_successor(
            source.binding.semantic_generation_words
        )
        generation_successor = capacity & _words_equal(
            destination.binding.semantic_generation_words,
            expected_generation,
        )
        expected_learner_step, learner_capacity = _words_successor(
            source.learner_state.step_words
        )
        learner_step_successor = learner_capacity & _words_equal(
            destination.learner_state.step_words,
            expected_learner_step,
        )
        birth_changed = jnp.any(
            source.binding.slot_birth_words != destination.binding.slot_birth_words,
            axis=1,
        )
        descriptor_changed = self._descriptor_change_mask(
            source.learner_state,
            destination.learner_state,
        )
        valid = generation_successor & learner_step_successor & jnp.any(
            birth_changed[self._base_feature_dim :]
        )
        valid &= ~jnp.any(birth_changed[: self._base_feature_dim])
        valid &= jnp.all((~descriptor_changed) | birth_changed)
        valid &= jnp.all(
            jnp.where(
                birth_changed[:, None],
                destination.binding.slot_birth_words
                == expected_generation[None, :],
                destination.binding.slot_birth_words
                == source.binding.slot_birth_words,
            )
        )
        valid &= self._dependency_closure_valid(
            destination.learner_state,
            birth_changed,
        )
        return valid

    def reencode_rows(
        self,
        source: CompositionalFeatureAdapterState,
        destination: CompositionalFeatureAdapterState,
        values: Array,
        valid_rows: Array,
    ) -> CompositionalFeatureReencodeResult:
        """Bit-authenticate source rows and atomically encode one successor bank."""

        values_shape = getattr(values, "shape", None)
        if type(values_shape) is not tuple or len(values_shape) != 2:
            raise ValueError("values must have rank two with static dimensions")
        rows = values_shape[0]
        if type(rows) is not int:
            raise ValueError("values must have one static leading row dimension")
        if not _array_contract(values, (rows, self._n_features), jnp.float32):
            raise ValueError("values must be exact float32 deployed-representation rows")
        if not _array_contract(valid_rows, (rows,), jnp.bool_):
            raise ValueError("valid_rows must be an exact bool row mask")
        if rows > _MAX_REENCODE_ROWS // max(2 * self._n_features, 1):
            raise ValueError("reencode row count exceeds exact int32 work accounting")
        if not self._learner_state_contract(source.learner_state) or not (
            self._learner_state_contract(destination.learner_state)
        ):
            raise ValueError("adapter state tree contract is not safe to re-encode")

        source_state_valid = self.state_valid(source)
        destination_state_valid = self.state_valid(destination)
        bindings_same = self._bindings_equal(source.binding, destination.binding)
        expected_generation, generation_capacity = _words_successor(
            source.binding.semantic_generation_words
        )
        generation_is_successor = generation_capacity & _words_equal(
            destination.binding.semantic_generation_words,
            expected_generation,
        )
        expected_learner_step, learner_capacity = _words_successor(
            source.learner_state.step_words
        )
        learner_step_is_successor = learner_capacity & _words_equal(
            destination.learner_state.step_words,
            expected_learner_step,
        )
        transition_consistent = self._successor_binding_valid(source, destination)
        source_reconstructed = jax.vmap(
            lambda row: self._representation_unchecked(
                source.learner_state,
                row[: self._base_feature_dim],
            )
        )(values)
        source_bits = jax.lax.bitcast_convert_type(values, jnp.uint32)
        reconstructed_bits = jax.lax.bitcast_convert_type(
            source_reconstructed,
            jnp.uint32,
        )
        row_matches = jnp.all(source_bits == reconstructed_bits, axis=1)
        source_rows_match = jnp.all((~valid_rows) | row_matches)
        source_rows_finite = jnp.all(
            jnp.isfinite(jnp.where(valid_rows[:, None], values, 0.0))
        )
        candidate_values = jax.vmap(
            lambda row: self._representation_unchecked(
                destination.learner_state,
                row[: self._base_feature_dim],
            )
        )(values)
        candidate_values_finite = jnp.all(
            jnp.isfinite(jnp.where(valid_rows[:, None], candidate_values, 0.0))
        )
        transaction_applied = (
            source_state_valid
            & destination_state_valid
            & transition_consistent
            & source_rows_match
            & source_rows_finite
            & candidate_values_finite
        )
        transaction_noop = (
            source_state_valid
            & destination_state_valid
            & bindings_same
            & source_rows_match
            & source_rows_finite
        )
        proposed = jnp.where(valid_rows[:, None], candidate_values, values)
        committed = jnp.where(transaction_applied, proposed, values)
        valid_count = jnp.sum(valid_rows.astype(jnp.int32))
        diagnostics = CompositionalFeatureReencodeDiagnostics(
            source_state_valid=source_state_valid,
            destination_state_valid=destination_state_valid,
            bindings_same=bindings_same,
            generation_is_successor=generation_is_successor,
            learner_step_is_successor=learner_step_is_successor,
            transition_consistent=transition_consistent,
            source_rows_match=source_rows_match,
            candidate_values_finite=candidate_values_finite,
            transaction_applied=transaction_applied,
            transaction_noop=transaction_noop,
            valid_rows_reencoded=jnp.where(transaction_applied, valid_count, 0),
            feature_slot_evaluations=jnp.asarray(
                2 * rows * self._n_features,
                dtype=jnp.int32,
            ),
        )
        return CompositionalFeatureReencodeResult(
            values=committed,
            diagnostics=diagnostics,
        )

    def measure_state_nbytes(self, state: CompositionalFeatureAdapterState) -> int:
        """Measure every persistent numeric leaf represented by ``state``."""

        if not bool(self.state_valid(state)):
            raise ValueError("cannot measure invalid adapter state")
        return _adapter_state_nbytes(state)

    def measure_prepared_update_nbytes(
        self,
        proposal: CompositionalFeatureAdapterPreparedUpdate,
    ) -> int:
        """Measure proposal bytes, including the one-byte curation permission."""

        self._prepared_update_contract(proposal)
        return (
            _tree_nbytes(proposal)
            + _host_float_metadata_adjustment(proposal.source_state.learner_state)
            + _host_float_metadata_adjustment(proposal.candidate_state.learner_state)
        )

    def resource_budget(
        self,
        state: CompositionalFeatureAdapterState,
        *,
        reencode_capacity: int,
    ) -> CompositionalFeatureAdapterResourceBudget:
        """Return exact state bytes and the fixed two-pass replay work bound."""

        capacity = _strict_int(
            reencode_capacity,
            name="reencode_capacity",
            minimum=0,
            maximum=min(
                _MAX_REENCODE_ROWS,
                _INT32_MAX // max(2 * self._n_features, 1),
            ),
        )
        if not bool(self.state_valid(state)):
            raise ValueError("cannot budget invalid adapter state")
        learner_bytes = _learner_state_nbytes(state.learner_state)
        binding_bytes = _tree_nbytes(state.binding)
        expected_binding_bytes = 32 * self._n_features + 44
        if binding_bytes != expected_binding_bytes:
            raise RuntimeError("binding byte formula drifted from the live state tree")
        return CompositionalFeatureAdapterResourceBudget(
            schema=COMPOSITIONAL_FEATURE_ADAPTER_STATE_SCHEMA,
            mechanism_status=COMPOSITIONAL_FEATURE_ADAPTER_MECHANISM_STATUS,
            scientific_promotion_allowed=False,
            base_feature_dim=self._base_feature_dim,
            n_features=self._n_features,
            dynamic_feature_slots=self.dynamic_feature_slots,
            candidate_slots=self._candidate_count,
            n_tasks=self._n_tasks,
            learner_persistent_nbytes=learner_bytes,
            binding_persistent_nbytes=binding_bytes,
            total_persistent_state_nbytes=learner_bytes + binding_bytes,
            max_reencode_rows=capacity,
            max_reencode_feature_slot_evaluations=(
                2 * capacity * self._n_features
            ),
        )


__all__ = [
    "COMPOSITIONAL_FEATURE_ADAPTER_CONFIG_SCHEMA",
    "COMPOSITIONAL_FEATURE_ADAPTER_MECHANISM_STATUS",
    "COMPOSITIONAL_FEATURE_ADAPTER_PREPARED_CURATION_PERMISSION_NBYTES",
    "COMPOSITIONAL_FEATURE_ADAPTER_SCIENTIFIC_PROMOTION_ALLOWED",
    "COMPOSITIONAL_FEATURE_ADAPTER_STATE_SCHEMA",
    "CompositionalFeatureAdapter",
    "CompositionalFeatureAdapterCommitDiagnostics",
    "CompositionalFeatureAdapterCommitResult",
    "CompositionalFeatureAdapterDiagnostics",
    "CompositionalFeatureAdapterPreparedUpdate",
    "CompositionalFeatureAdapterResourceBudget",
    "CompositionalFeatureAdapterState",
    "CompositionalFeatureAdapterUpdateResult",
    "CompositionalFeatureBinding",
    "CompositionalFeatureReencodeDiagnostics",
    "CompositionalFeatureReencodeResult",
]
