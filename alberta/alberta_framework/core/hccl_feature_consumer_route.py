# mypy: disable-error-code="call-arg,no-any-return"
"""Source-bound semantic births and a pure HCCL feature-route witness.

This module is the first additive unit toward the feature-consumer route in
``CONTINUAL_DYAD_BENCHMARK.md``.  It owns no learner or consumer parameters.
It records the complete 35-coordinate HCCL feature identity, constructs one
exact successor ledger, and emits the destination-to-source map that a later
outer transaction can use to route every consumer atomically.

The representation order is the one already pinned by the current HCCL outer
transaction: physical16, context3, fast4, pair12.  Pair identities may move
between the twelve pair slots.  Context identities cannot move: a changed
context-slot birth is a newborn and the retired source identity is scrubbed.
The same pair descriptor reappearing after retirement is also a newborn, and
an explicit admission bit can retire and readmit an immediately live descriptor
in one successor.  Descriptor equality is used only to locate a uniquely live
record whose complete birth identity is carried into the candidate ledger.
The emitted route map then matches the complete immutable birth record: kind,
descriptor, birth counter, birth source clock, parents, and birth event.

All integrity checks are deliberately host/eager-only SHA-256 bindings over
exact shapes, dtypes, and bytes.  They detect mutation or composition mismatch
within the supplied stateless values but do not authenticate a caller, retain
history, or detect replay of a previously valid ledger.  No consumer routing,
benchmark execution, artifact writing, evidence, promotion, or Alberta Plan
completion is claimed here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from enum import IntEnum
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Int, UInt

HCCL_FEATURE_CONSUMER_ROUTE_CONFIG_SCHEMA = (
    "alberta.hccl-feature-consumer-route.config.v2"
)
HCCL_FEATURE_BIRTH_LEDGER_SCHEMA = "alberta.hccl-feature-birth-ledger.v2"
HCCL_FEATURE_ROUTE_MAP_SCHEMA = "alberta.hccl-feature-route-map.v2"
HCCL_FEATURE_ROUTE_WITNESS_SCHEMA = "alberta.hccl-feature-route-witness.v2"
HCCL_FEATURE_CONSUMER_ROUTE_STATUS = (
    "l0-development-semantic-birth-route-map-only"
)
HCCL_FEATURE_CONSUMER_ROUTE_FULL_CONSUMER_ROUTING_CLAIMED = False
HCCL_FEATURE_CONSUMER_ROUTE_SCIENTIFIC_PROMOTION_ALLOWED = False

HCCL_FEATURE_PHYSICAL_DIM = 16
HCCL_FEATURE_CONTEXT_DIM = 3
HCCL_FEATURE_FAST_DIM = 4
HCCL_FEATURE_PAIR_SLOTS = 12
HCCL_FEATURE_CONTEXT_START = HCCL_FEATURE_PHYSICAL_DIM
HCCL_FEATURE_FAST_START = HCCL_FEATURE_CONTEXT_START + HCCL_FEATURE_CONTEXT_DIM
HCCL_FEATURE_PAIR_START = HCCL_FEATURE_FAST_START + HCCL_FEATURE_FAST_DIM
HCCL_FEATURE_TOTAL_DIM = HCCL_FEATURE_PAIR_START + HCCL_FEATURE_PAIR_SLOTS

_TOKEN_NBYTES = 32
_COUNTER_WORDS = 2
_UINT32_MAX = 2**32 - 1
_PAIR_DESCRIPTOR_COMPARISONS = HCCL_FEATURE_PAIR_SLOTS**2
_FULL_IDENTITY_COMPARISONS = HCCL_FEATURE_TOTAL_DIM**2
_INACTIVE_DESCRIPTOR = (-1, -1)
_NO_PARENTS = (-1, -1)


class HCCLFeatureKind(IntEnum):
    """Fixed type of one coordinate in the deployed HCCL representation."""

    PHYSICAL = 0
    CONTEXT = 1
    FAST = 2
    PAIR = 3


class HCCLFeatureBirthEvent(IntEnum):
    """Event that created the currently live identity in one slot."""

    INACTIVE = 0
    GENESIS = 1
    CONTEXT_ALLOCATION = 2
    PAIR_ADMISSION = 3


@chex.dataclass(frozen=True)
class HCCLFeatureBirthLedger:
    """One agent's complete current 35-coordinate semantic identity."""

    config_token: UInt[Array, " 32"]
    content_token: UInt[Array, " 32"]
    schema_digest: UInt[Array, " 32"]
    agent_index: Int[Array, ""]
    source_clock_words: UInt[Array, " 2"]
    semantic_generation_words: UInt[Array, " 2"]
    kind: Int[Array, " 35"]
    descriptor: Int[Array, "35 2"]
    birth_words: UInt[Array, "35 2"]
    birth_source_words: UInt[Array, "35 2"]
    parents: Int[Array, "35 2"]
    birth_event: Int[Array, " 35"]
    active: Bool[Array, " 35"]


@chex.dataclass(frozen=True)
class HCCLFeatureConsumerRouteMap:
    """Destination-to-source map under complete semantic-birth identity."""

    source_slots: Int[Array, " 35"]
    survivor_mask: Bool[Array, " 35"]
    newborn_mask: Bool[Array, " 35"]
    inactive_mask: Bool[Array, " 35"]
    retired_mask: Bool[Array, " 35"]
    survivor_count: Int[Array, ""]
    newborn_count: Int[Array, ""]
    inactive_count: Int[Array, ""]
    retired_count: Int[Array, ""]
    unique_full_identity_matches: Bool[Array, ""]
    unique_source_identity_use: Bool[Array, ""]
    classification_complete: Bool[Array, ""]


@chex.dataclass(frozen=True)
class HCCLFeatureConsumerRouteWork:
    """Exact logical work executed by :meth:`prepare_successor`."""

    source_ledger_validations: Int[Array, ""]
    successor_ledger_candidates: Int[Array, ""]
    pair_descriptor_identity_comparisons: Int[Array, ""]
    full_birth_identity_comparisons: Int[Array, ""]
    destination_slot_classifications: Int[Array, ""]
    source_slot_retirement_classifications: Int[Array, ""]
    ledger_content_digest_evaluations: Int[Array, ""]
    witness_content_digest_evaluations: Int[Array, ""]
    consumer_route_evaluations: Int[Array, ""]
    rng_draws: Int[Array, ""]


@chex.dataclass(frozen=True)
class HCCLFeatureConsumerRouteWitness:
    """Integrity-bound preparation result with no consumer mutation authority."""

    source_content_token: UInt[Array, " 32"]
    destination_content_token: UInt[Array, " 32"]
    source_clock_words: UInt[Array, " 2"]
    destination_source_clock_words: UInt[Array, " 2"]
    source_semantic_generation_words: UInt[Array, " 2"]
    destination_semantic_generation_words: UInt[Array, " 2"]
    requested_context_active: Bool[Array, " 3"]
    requested_context_birth_words: UInt[Array, "3 2"]
    requested_pair_descriptors: Int[Array, "12 2"]
    requested_pair_admission_mask: Bool[Array, " 12"]
    route_map: HCCLFeatureConsumerRouteMap
    work: HCCLFeatureConsumerRouteWork
    source_ledger_valid: Bool[Array, ""]
    destination_inputs_valid: Bool[Array, ""]
    source_clock_is_successor: Bool[Array, ""]
    semantic_generation_capacity_available: Bool[Array, ""]
    candidate_ledger_valid: Bool[Array, ""]
    route_map_valid: Bool[Array, ""]
    semantic_bank_changed: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]
    complete_source_returned: Bool[Array, ""]
    content_token: UInt[Array, " 32"]


@chex.dataclass(frozen=True)
class HCCLFeatureConsumerRouteResult:
    """Selected ledger, unselected candidate, and exact attempted witness."""

    ledger: HCCLFeatureBirthLedger
    candidate_ledger: HCCLFeatureBirthLedger
    witness: HCCLFeatureConsumerRouteWitness


_LEDGER_FIELD_NAMES = (
    "config_token",
    "content_token",
    "schema_digest",
    "agent_index",
    "source_clock_words",
    "semantic_generation_words",
    "kind",
    "descriptor",
    "birth_words",
    "birth_source_words",
    "parents",
    "birth_event",
    "active",
)
_WORK_FIELD_NAMES = (
    "source_ledger_validations",
    "successor_ledger_candidates",
    "pair_descriptor_identity_comparisons",
    "full_birth_identity_comparisons",
    "destination_slot_classifications",
    "source_slot_retirement_classifications",
    "ledger_content_digest_evaluations",
    "witness_content_digest_evaluations",
    "consumer_route_evaluations",
    "rng_draws",
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _contains_tracer(value: object) -> bool:
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(value))


def _digest_tree(schema: str, *values: object) -> UInt[Array, " 32"]:
    """Hash exact host material at this deliberately host-only boundary."""

    if _contains_tracer(values):
        raise TypeError("HCCL feature-consumer route integrity is host/eager-only")
    digest = hashlib.sha256()
    digest.update(schema.encode("ascii"))
    for value in values:
        digest.update(type(value).__module__.encode("utf-8"))
        digest.update(type(value).__qualname__.encode("utf-8"))
        leaves, structure = jax.tree.flatten(value)
        digest.update(repr(structure).encode("utf-8"))
        digest.update(len(leaves).to_bytes(8, "big"))
        for leaf in leaves:
            if hasattr(leaf, "dtype") and hasattr(leaf, "shape"):
                host = np.ascontiguousarray(
                    np.asarray(jax.device_get(jnp.asarray(leaf)))
                )
                digest.update(str(host.dtype).encode("ascii"))
                digest.update(np.asarray(host.shape, dtype=np.int64).tobytes())
                digest.update(host.tobytes(order="C"))
            else:
                digest.update(type(leaf).__module__.encode("utf-8"))
                digest.update(type(leaf).__qualname__.encode("utf-8"))
                digest.update(repr(leaf).encode("utf-8"))
    return jnp.asarray(tuple(digest.digest()), dtype=jnp.uint8)


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
    if jnp.dtype(array.dtype) != jnp.dtype(dtype):
        raise TypeError(f"{name} must have dtype {jnp.dtype(dtype)}; got {array.dtype}")
    return array


def _host(value: Array) -> np.ndarray[Any, Any]:
    return np.asarray(jax.device_get(value))


def _host_bool(value: object) -> bool:
    return bool(np.asarray(jax.device_get(value)))


def _words_tuple(value: Array) -> tuple[int, int]:
    host = _host(value)
    return int(host[0]), int(host[1])


def _words_equal(left: Array, right: Array) -> bool:
    return _words_tuple(left) == _words_tuple(right)


def _words_le(left: Array, right: Array) -> bool:
    return _words_tuple(left) <= _words_tuple(right)


def _words_successor(value: Array) -> tuple[Array, bool]:
    high, low = _words_tuple(value)
    if high == _UINT32_MAX and low == _UINT32_MAX:
        return jnp.asarray((high, low), dtype=jnp.uint32), False
    if low == _UINT32_MAX:
        return jnp.asarray((high + 1, 0), dtype=jnp.uint32), True
    return jnp.asarray((high, low + 1), dtype=jnp.uint32), True


def _array_exact_equal(left: Array, right: Array) -> bool:
    left_host = _host(left)
    right_host = _host(right)
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
        _array_exact_equal(jnp.asarray(left_leaf), jnp.asarray(right_leaf))
        for left_leaf, right_leaf in zip(
            left_leaves,
            right_leaves,
            strict=True,
        )
    )


def _zero_words() -> Array:
    return jnp.zeros((_COUNTER_WORDS,), dtype=jnp.uint32)


def _fixed_work() -> HCCLFeatureConsumerRouteWork:
    return HCCLFeatureConsumerRouteWork(
        source_ledger_validations=jnp.asarray(1, dtype=jnp.int32),
        successor_ledger_candidates=jnp.asarray(1, dtype=jnp.int32),
        pair_descriptor_identity_comparisons=jnp.asarray(
            _PAIR_DESCRIPTOR_COMPARISONS,
            dtype=jnp.int32,
        ),
        full_birth_identity_comparisons=jnp.asarray(
            _FULL_IDENTITY_COMPARISONS,
            dtype=jnp.int32,
        ),
        destination_slot_classifications=jnp.asarray(
            HCCL_FEATURE_TOTAL_DIM,
            dtype=jnp.int32,
        ),
        source_slot_retirement_classifications=jnp.asarray(
            HCCL_FEATURE_TOTAL_DIM,
            dtype=jnp.int32,
        ),
        ledger_content_digest_evaluations=jnp.asarray(3, dtype=jnp.int32),
        witness_content_digest_evaluations=jnp.asarray(1, dtype=jnp.int32),
        consumer_route_evaluations=jnp.asarray(0, dtype=jnp.int32),
        rng_draws=jnp.asarray(0, dtype=jnp.int32),
    )


class HCCLFeatureConsumerRoute:
    """Pure host/eager successor ledger and full-birth route-map owner."""

    def __init__(self, *, agent_index: int) -> None:
        if type(agent_index) is not int or agent_index not in (0, 1):
            raise ValueError("agent_index must be the exact dyad index 0 or 1")
        self._agent_index = agent_index
        self._config_payload = {
            "type": type(self).__name__,
            "schema": HCCL_FEATURE_CONSUMER_ROUTE_CONFIG_SCHEMA,
            "ledger_schema": HCCL_FEATURE_BIRTH_LEDGER_SCHEMA,
            "route_map_schema": HCCL_FEATURE_ROUTE_MAP_SCHEMA,
            "witness_schema": HCCL_FEATURE_ROUTE_WITNESS_SCHEMA,
            "mechanism_status": HCCL_FEATURE_CONSUMER_ROUTE_STATUS,
            "agent_index": agent_index,
            "representation_order": ["physical16", "context3", "fast4", "pair12"],
            "physical_dim": HCCL_FEATURE_PHYSICAL_DIM,
            "context_dim": HCCL_FEATURE_CONTEXT_DIM,
            "fast_dim": HCCL_FEATURE_FAST_DIM,
            "pair_slots": HCCL_FEATURE_PAIR_SLOTS,
            "total_dim": HCCL_FEATURE_TOTAL_DIM,
            "pair_parent_domain": "physical16-only",
            "identity_semantics": (
                "kind+descriptor+birth_words+birth_source_words+parents+birth_event"
            ),
            "context_identity_may_move": False,
            "pair_identity_may_move": True,
            "same_descriptor_reintroduction_is_newborn": True,
            "explicit_pair_admission_mask_required": True,
            "pure_pair_permutation_advances_generation": False,
            "host_eager_only": True,
            "integrity_authenticated": False,
            "history_owned": False,
            "replay_detection_claimed": False,
            "full_consumer_routing_claimed": (
                HCCL_FEATURE_CONSUMER_ROUTE_FULL_CONSUMER_ROUTING_CLAIMED
            ),
            "scientific_promotion_allowed": (
                HCCL_FEATURE_CONSUMER_ROUTE_SCIENTIFIC_PROMOTION_ALLOWED
            ),
        }
        self._config_token = jnp.asarray(
            tuple(hashlib.sha256(_canonical_json_bytes(self._config_payload)).digest()),
            dtype=jnp.uint8,
        )
        ledger_manifest = {
            "schema": HCCL_FEATURE_BIRTH_LEDGER_SCHEMA,
            "fields": list(_LEDGER_FIELD_NAMES),
            "identity_fields": [
                "kind",
                "descriptor",
                "birth_words",
                "birth_source_words",
                "parents",
                "birth_event",
            ],
        }
        self._schema_digest = jnp.asarray(
            tuple(hashlib.sha256(_canonical_json_bytes(ledger_manifest)).digest()),
            dtype=jnp.uint8,
        )

    @property
    def agent_index(self) -> int:
        return self._agent_index

    def to_config(self) -> dict[str, object]:
        return dict(self._config_payload)

    @classmethod
    def from_config(cls, payload: Mapping[str, object]) -> HCCLFeatureConsumerRoute:
        if type(payload) is not dict:
            raise TypeError("HCCL feature-consumer route config must be an exact dict")
        if type(payload.get("agent_index")) is not int:
            raise ValueError("serialized agent_index must be an exact int")
        restored = cls(agent_index=cast(int, payload["agent_index"]))
        if restored.to_config() != payload:
            raise ValueError("HCCL feature-consumer route config is noncanonical")
        return restored

    def _ledger_content_token(self, ledger: HCCLFeatureBirthLedger) -> Array:
        return _digest_tree(
            HCCL_FEATURE_BIRTH_LEDGER_SCHEMA,
            ledger.config_token,
            ledger.schema_digest,
            ledger.agent_index,
            ledger.source_clock_words,
            ledger.semantic_generation_words,
            ledger.kind,
            ledger.descriptor,
            ledger.birth_words,
            ledger.birth_source_words,
            ledger.parents,
            ledger.birth_event,
            ledger.active,
        )

    def _seal_ledger(self, ledger: HCCLFeatureBirthLedger) -> HCCLFeatureBirthLedger:
        return cast(
            HCCLFeatureBirthLedger,
            cast(Any, ledger).replace(
                content_token=self._ledger_content_token(ledger)
            ),
        )

    def _witness_content_token(
        self,
        witness: HCCLFeatureConsumerRouteWitness,
    ) -> Array:
        return _digest_tree(
            HCCL_FEATURE_ROUTE_WITNESS_SCHEMA,
            witness.source_content_token,
            witness.destination_content_token,
            witness.source_clock_words,
            witness.destination_source_clock_words,
            witness.source_semantic_generation_words,
            witness.destination_semantic_generation_words,
            witness.requested_context_active,
            witness.requested_context_birth_words,
            witness.requested_pair_descriptors,
            witness.requested_pair_admission_mask,
            witness.route_map,
            witness.work,
            witness.source_ledger_valid,
            witness.destination_inputs_valid,
            witness.source_clock_is_successor,
            witness.semantic_generation_capacity_available,
            witness.candidate_ledger_valid,
            witness.route_map_valid,
            witness.semantic_bank_changed,
            witness.transaction_applied,
            witness.complete_source_returned,
        )

    def _seal_witness(
        self,
        witness: HCCLFeatureConsumerRouteWitness,
    ) -> HCCLFeatureConsumerRouteWitness:
        return cast(
            HCCLFeatureConsumerRouteWitness,
            cast(Any, witness).replace(
                content_token=self._witness_content_token(witness)
            ),
        )

    def _require_ledger_contract(self, ledger: HCCLFeatureBirthLedger) -> None:
        if type(ledger) is not HCCLFeatureBirthLedger:
            raise TypeError("ledger must be an exact HCCLFeatureBirthLedger")
        for name in ("config_token", "content_token", "schema_digest"):
            _require_array(
                getattr(ledger, name),
                name=f"ledger.{name}",
                shape=(_TOKEN_NBYTES,),
                dtype=jnp.uint8,
            )
        _require_array(
            ledger.agent_index,
            name="ledger.agent_index",
            shape=(),
            dtype=jnp.int32,
        )
        for name in ("source_clock_words", "semantic_generation_words"):
            _require_array(
                getattr(ledger, name),
                name=f"ledger.{name}",
                shape=(_COUNTER_WORDS,),
                dtype=jnp.uint32,
            )
        for name in ("kind", "birth_event"):
            _require_array(
                getattr(ledger, name),
                name=f"ledger.{name}",
                shape=(HCCL_FEATURE_TOTAL_DIM,),
                dtype=jnp.int32,
            )
        for name, dtype in (
            ("descriptor", jnp.int32),
            ("birth_words", jnp.uint32),
            ("birth_source_words", jnp.uint32),
            ("parents", jnp.int32),
        ):
            _require_array(
                getattr(ledger, name),
                name=f"ledger.{name}",
                shape=(HCCL_FEATURE_TOTAL_DIM, 2),
                dtype=dtype,
            )
        _require_array(
            ledger.active,
            name="ledger.active",
            shape=(HCCL_FEATURE_TOTAL_DIM,),
            dtype=jnp.bool_,
        )

    def _require_destination_contract(
        self,
        destination_source_clock_words: Array,
        context_active: Array,
        context_birth_words: Array,
        pair_descriptors: Array,
        pair_admission_mask: Array,
    ) -> None:
        _require_array(
            destination_source_clock_words,
            name="destination_source_clock_words",
            shape=(_COUNTER_WORDS,),
            dtype=jnp.uint32,
        )
        _require_array(
            context_active,
            name="context_active",
            shape=(HCCL_FEATURE_CONTEXT_DIM,),
            dtype=jnp.bool_,
        )
        _require_array(
            context_birth_words,
            name="context_birth_words",
            shape=(HCCL_FEATURE_CONTEXT_DIM, _COUNTER_WORDS),
            dtype=jnp.uint32,
        )
        _require_array(
            pair_descriptors,
            name="pair_descriptors",
            shape=(HCCL_FEATURE_PAIR_SLOTS, 2),
            dtype=jnp.int32,
        )
        _require_array(
            pair_admission_mask,
            name="pair_admission_mask",
            shape=(HCCL_FEATURE_PAIR_SLOTS,),
            dtype=jnp.bool_,
        )

    def _require_route_map_contract(self, route_map: HCCLFeatureConsumerRouteMap) -> None:
        if type(route_map) is not HCCLFeatureConsumerRouteMap:
            raise TypeError("route_map must be an exact HCCLFeatureConsumerRouteMap")
        _require_array(
            route_map.source_slots,
            name="route_map.source_slots",
            shape=(HCCL_FEATURE_TOTAL_DIM,),
            dtype=jnp.int32,
        )
        for name in ("survivor_mask", "newborn_mask", "inactive_mask", "retired_mask"):
            _require_array(
                getattr(route_map, name),
                name=f"route_map.{name}",
                shape=(HCCL_FEATURE_TOTAL_DIM,),
                dtype=jnp.bool_,
            )
        for name in (
            "survivor_count",
            "newborn_count",
            "inactive_count",
            "retired_count",
        ):
            _require_array(
                getattr(route_map, name),
                name=f"route_map.{name}",
                shape=(),
                dtype=jnp.int32,
            )
        for name in (
            "unique_full_identity_matches",
            "unique_source_identity_use",
            "classification_complete",
        ):
            _require_array(
                getattr(route_map, name),
                name=f"route_map.{name}",
                shape=(),
                dtype=jnp.bool_,
            )

    def _require_work_contract(self, work: HCCLFeatureConsumerRouteWork) -> None:
        if type(work) is not HCCLFeatureConsumerRouteWork:
            raise TypeError("work must be exact HCCLFeatureConsumerRouteWork")
        for name in _WORK_FIELD_NAMES:
            _require_array(
                getattr(work, name),
                name=f"work.{name}",
                shape=(),
                dtype=jnp.int32,
            )

    def _require_witness_contract(
        self,
        witness: HCCLFeatureConsumerRouteWitness,
    ) -> None:
        if type(witness) is not HCCLFeatureConsumerRouteWitness:
            raise TypeError("witness must be an exact HCCLFeatureConsumerRouteWitness")
        for name in (
            "source_content_token",
            "destination_content_token",
            "content_token",
        ):
            _require_array(
                getattr(witness, name),
                name=f"witness.{name}",
                shape=(_TOKEN_NBYTES,),
                dtype=jnp.uint8,
            )
        for name in (
            "source_clock_words",
            "destination_source_clock_words",
            "source_semantic_generation_words",
            "destination_semantic_generation_words",
        ):
            _require_array(
                getattr(witness, name),
                name=f"witness.{name}",
                shape=(_COUNTER_WORDS,),
                dtype=jnp.uint32,
            )
        _require_array(
            witness.requested_context_active,
            name="witness.requested_context_active",
            shape=(HCCL_FEATURE_CONTEXT_DIM,),
            dtype=jnp.bool_,
        )
        _require_array(
            witness.requested_context_birth_words,
            name="witness.requested_context_birth_words",
            shape=(HCCL_FEATURE_CONTEXT_DIM, _COUNTER_WORDS),
            dtype=jnp.uint32,
        )
        _require_array(
            witness.requested_pair_descriptors,
            name="witness.requested_pair_descriptors",
            shape=(HCCL_FEATURE_PAIR_SLOTS, 2),
            dtype=jnp.int32,
        )
        _require_array(
            witness.requested_pair_admission_mask,
            name="witness.requested_pair_admission_mask",
            shape=(HCCL_FEATURE_PAIR_SLOTS,),
            dtype=jnp.bool_,
        )
        for name in (
            "source_ledger_valid",
            "destination_inputs_valid",
            "source_clock_is_successor",
            "semantic_generation_capacity_available",
            "candidate_ledger_valid",
            "route_map_valid",
            "semantic_bank_changed",
            "transaction_applied",
            "complete_source_returned",
        ):
            _require_array(
                getattr(witness, name),
                name=f"witness.{name}",
                shape=(),
                dtype=jnp.bool_,
            )
        self._require_route_map_contract(witness.route_map)
        self._require_work_contract(witness.work)

    def _require_result_contract(
        self,
        result: HCCLFeatureConsumerRouteResult,
    ) -> None:
        if type(result) is not HCCLFeatureConsumerRouteResult:
            raise TypeError("result must be an exact HCCLFeatureConsumerRouteResult")
        self._require_ledger_contract(result.ledger)
        self._require_ledger_contract(result.candidate_ledger)
        self._require_witness_contract(result.witness)

    @staticmethod
    def _canonical_kind() -> np.ndarray[Any, Any]:
        kinds = np.empty((HCCL_FEATURE_TOTAL_DIM,), dtype=np.int32)
        kinds[:HCCL_FEATURE_CONTEXT_START] = int(HCCLFeatureKind.PHYSICAL)
        kinds[HCCL_FEATURE_CONTEXT_START:HCCL_FEATURE_FAST_START] = int(
            HCCLFeatureKind.CONTEXT
        )
        kinds[HCCL_FEATURE_FAST_START:HCCL_FEATURE_PAIR_START] = int(
            HCCLFeatureKind.FAST
        )
        kinds[HCCL_FEATURE_PAIR_START:] = int(HCCLFeatureKind.PAIR)
        return kinds

    @staticmethod
    def _canonical_descriptors(pair_descriptors: Array) -> np.ndarray[Any, Any]:
        descriptors = np.full(
            (HCCL_FEATURE_TOTAL_DIM, 2),
            -1,
            dtype=np.int32,
        )
        for index in range(HCCL_FEATURE_PHYSICAL_DIM):
            descriptors[index] = (index, -1)
        for local in range(HCCL_FEATURE_CONTEXT_DIM):
            descriptors[HCCL_FEATURE_CONTEXT_START + local] = (local, -1)
        for local in range(HCCL_FEATURE_FAST_DIM):
            descriptors[HCCL_FEATURE_FAST_START + local] = (local, -1)
        descriptors[HCCL_FEATURE_PAIR_START:] = _host(pair_descriptors)
        return descriptors

    @staticmethod
    def _pair_input_valid(pair_descriptors: Array) -> bool:
        values = _host(pair_descriptors)
        active = np.all(values >= 0, axis=1)
        inactive = np.all(values == -1, axis=1)
        encoding_valid = bool(np.all(active | inactive))
        canonical = bool(
            np.all(
                (~active)
                | (
                    (values[:, 0] >= 0)
                    & (values[:, 0] < values[:, 1])
                    & (values[:, 1] < HCCL_FEATURE_PHYSICAL_DIM)
                )
            )
        )
        live = values[active]
        unique = len(live) == len({tuple(int(item) for item in row) for row in live})
        return encoding_valid and canonical and unique

    @staticmethod
    def _pair_descriptor_identity_matrix(
        source: HCCLFeatureBirthLedger,
        pair_descriptors: Array,
    ) -> np.ndarray[Any, Any]:
        destination = _host(pair_descriptors)
        destination_active = np.all(destination >= 0, axis=1)
        source_values = _host(source.descriptor)[HCCL_FEATURE_PAIR_START:]
        source_active = _host(source.active)[HCCL_FEATURE_PAIR_START:].astype(np.bool_)
        matches = np.all(
            destination[:, None, :] == source_values[None, :, :],
            axis=2,
        )
        matches &= destination_active[:, None] & source_active[None, :]
        return matches

    @classmethod
    def _pair_transition_input_valid(
        cls,
        pair_descriptors: Array,
        pair_admission_mask: Array,
        pair_descriptor_matches: np.ndarray[Any, Any],
    ) -> bool:
        """Require every non-admitted live destination to name one live birth."""

        if not cls._pair_input_valid(pair_descriptors):
            return False
        destination = _host(pair_descriptors)
        admissions = _host(pair_admission_mask).astype(np.bool_)
        destination_active = np.all(destination >= 0, axis=1)
        if np.any(admissions & ~destination_active):
            return False
        carried = destination_active & ~admissions
        return bool(
            np.all(np.sum(pair_descriptor_matches[carried], axis=1) == 1)
        )

    @staticmethod
    def _context_input_valid(
        source: HCCLFeatureBirthLedger,
        destination_clock: Array,
        context_active: Array,
        context_birth_words: Array,
    ) -> bool:
        active = _host(context_active).astype(np.bool_)
        births = _host(context_birth_words)
        source_active = _host(source.active)[
            HCCL_FEATURE_CONTEXT_START:HCCL_FEATURE_FAST_START
        ].astype(np.bool_)
        source_births = _host(source.birth_words)[
            HCCL_FEATURE_CONTEXT_START:HCCL_FEATURE_FAST_START
        ]
        zero = np.zeros((2,), dtype=np.uint32)
        for index in range(HCCL_FEATURE_CONTEXT_DIM):
            if not active[index]:
                if not np.array_equal(births[index], zero):
                    return False
                continue
            survivor = source_active[index] and np.array_equal(
                births[index], source_births[index]
            )
            if survivor:
                continue
            if not np.array_equal(births[index], _host(destination_clock)):
                return False
        return True

    def _ledger_semantics_valid(self, ledger: HCCLFeatureBirthLedger) -> bool:
        kind = _host(ledger.kind)
        descriptors = _host(ledger.descriptor)
        births = _host(ledger.birth_words)
        birth_sources = _host(ledger.birth_source_words)
        parents = _host(ledger.parents)
        events = _host(ledger.birth_event)
        active = _host(ledger.active).astype(np.bool_)
        zero_words = np.zeros((2,), dtype=np.uint32)
        valid = np.array_equal(kind, self._canonical_kind())
        valid &= int(_host(ledger.agent_index)) == self._agent_index
        valid &= _array_exact_equal(ledger.config_token, self._config_token)
        valid &= _array_exact_equal(ledger.schema_digest, self._schema_digest)
        valid &= _words_le(ledger.semantic_generation_words, ledger.source_clock_words)

        expected_descriptors = self._canonical_descriptors(
            ledger.descriptor[HCCL_FEATURE_PAIR_START:]
        )
        valid &= np.array_equal(
            descriptors[:HCCL_FEATURE_PAIR_START],
            expected_descriptors[:HCCL_FEATURE_PAIR_START],
        )
        valid &= bool(np.all(active[:HCCL_FEATURE_CONTEXT_START]))
        valid &= bool(
            np.all(active[HCCL_FEATURE_FAST_START:HCCL_FEATURE_PAIR_START])
        )

        for slot in range(HCCL_FEATURE_TOTAL_DIM):
            slot_kind = int(kind[slot])
            if slot_kind in (int(HCCLFeatureKind.PHYSICAL), int(HCCLFeatureKind.FAST)):
                valid &= bool(active[slot])
                valid &= np.array_equal(births[slot], zero_words)
                valid &= np.array_equal(birth_sources[slot], zero_words)
                valid &= np.array_equal(parents[slot], _NO_PARENTS)
                valid &= int(events[slot]) == int(HCCLFeatureBirthEvent.GENESIS)
                continue
            if not active[slot]:
                valid &= np.array_equal(births[slot], zero_words)
                valid &= np.array_equal(birth_sources[slot], zero_words)
                valid &= np.array_equal(parents[slot], _NO_PARENTS)
                valid &= int(events[slot]) == int(HCCLFeatureBirthEvent.INACTIVE)
                if slot_kind == int(HCCLFeatureKind.PAIR):
                    valid &= tuple(int(item) for item in descriptors[slot]) == (
                        _INACTIVE_DESCRIPTOR
                    )
                continue

            valid &= tuple(int(item) for item in parents[slot]) == (
                _NO_PARENTS
                if slot_kind == int(HCCLFeatureKind.CONTEXT)
                else tuple(int(item) for item in descriptors[slot])
            )
            valid &= _words_le(jnp.asarray(birth_sources[slot]), ledger.source_clock_words)
            genesis = np.array_equal(births[slot], zero_words) and np.array_equal(
                birth_sources[slot], zero_words
            )
            if genesis:
                valid &= int(events[slot]) == int(HCCLFeatureBirthEvent.GENESIS)
            elif slot_kind == int(HCCLFeatureKind.CONTEXT):
                valid &= int(events[slot]) == int(
                    HCCLFeatureBirthEvent.CONTEXT_ALLOCATION
                )
                valid &= np.array_equal(births[slot], birth_sources[slot])
            else:
                valid &= int(events[slot]) == int(HCCLFeatureBirthEvent.PAIR_ADMISSION)
                valid &= not np.array_equal(births[slot], zero_words)
                valid &= not np.array_equal(birth_sources[slot], zero_words)
                valid &= _words_le(
                    jnp.asarray(births[slot]),
                    ledger.semantic_generation_words,
                )

        pair_values = descriptors[HCCL_FEATURE_PAIR_START:]
        pair_active = active[HCCL_FEATURE_PAIR_START:]
        valid &= self._pair_input_valid(jnp.asarray(pair_values, dtype=jnp.int32))
        valid &= np.array_equal(
            parents[HCCL_FEATURE_PAIR_START:][pair_active],
            pair_values[pair_active],
        )
        return bool(valid)

    def ledger_valid(self, ledger: HCCLFeatureBirthLedger) -> Bool[Array, ""]:
        """Validate exact structure, fixed semantics, source bounds, and SHA-256."""

        self._require_ledger_contract(ledger)
        if _contains_tracer(ledger):
            raise TypeError("HCCL feature-consumer route validity is host/eager-only")
        content_valid = _array_exact_equal(
            ledger.content_token,
            self._ledger_content_token(ledger),
        )
        return jnp.asarray(
            content_valid and self._ledger_semantics_valid(ledger),
            dtype=jnp.bool_,
        )

    def init(
        self,
        *,
        context_active: Array,
        pair_descriptors: Array,
    ) -> HCCLFeatureBirthLedger:
        """Create a zero-clock genesis ledger for one exact fixed geometry."""

        context_birth_words = jnp.zeros(
            (HCCL_FEATURE_CONTEXT_DIM, _COUNTER_WORDS),
            dtype=jnp.uint32,
        )
        self._require_destination_contract(
            _zero_words(),
            context_active,
            context_birth_words,
            pair_descriptors,
            jnp.zeros((HCCL_FEATURE_PAIR_SLOTS,), dtype=jnp.bool_),
        )
        if _contains_tracer((context_active, pair_descriptors)):
            raise TypeError("HCCL feature-consumer route genesis is host/eager-only")
        if not self._pair_input_valid(pair_descriptors):
            raise ValueError("genesis pair descriptors are invalid")

        active = np.zeros((HCCL_FEATURE_TOTAL_DIM,), dtype=np.bool_)
        active[:HCCL_FEATURE_CONTEXT_START] = True
        active[HCCL_FEATURE_CONTEXT_START:HCCL_FEATURE_FAST_START] = _host(
            context_active
        )
        active[HCCL_FEATURE_FAST_START:HCCL_FEATURE_PAIR_START] = True
        pair_values = _host(pair_descriptors)
        active[HCCL_FEATURE_PAIR_START:] = np.all(pair_values >= 0, axis=1)
        descriptors = self._canonical_descriptors(pair_descriptors)
        parents = np.full((HCCL_FEATURE_TOTAL_DIM, 2), -1, dtype=np.int32)
        parents[HCCL_FEATURE_PAIR_START:][active[HCCL_FEATURE_PAIR_START:]] = (
            pair_values[active[HCCL_FEATURE_PAIR_START:]]
        )
        events = np.where(
            active,
            int(HCCLFeatureBirthEvent.GENESIS),
            int(HCCLFeatureBirthEvent.INACTIVE),
        ).astype(np.int32)
        unsigned = HCCLFeatureBirthLedger(
            config_token=self._config_token,
            content_token=jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8),
            schema_digest=self._schema_digest,
            agent_index=jnp.asarray(self._agent_index, dtype=jnp.int32),
            source_clock_words=_zero_words(),
            semantic_generation_words=_zero_words(),
            kind=jnp.asarray(self._canonical_kind(), dtype=jnp.int32),
            descriptor=jnp.asarray(descriptors, dtype=jnp.int32),
            birth_words=jnp.zeros(
                (HCCL_FEATURE_TOTAL_DIM, _COUNTER_WORDS),
                dtype=jnp.uint32,
            ),
            birth_source_words=jnp.zeros(
                (HCCL_FEATURE_TOTAL_DIM, _COUNTER_WORDS),
                dtype=jnp.uint32,
            ),
            parents=jnp.asarray(parents, dtype=jnp.int32),
            birth_event=jnp.asarray(events, dtype=jnp.int32),
            active=jnp.asarray(active, dtype=jnp.bool_),
        )
        ledger = self._seal_ledger(unsigned)
        if not _host_bool(self.ledger_valid(ledger)):
            raise RuntimeError("constructed HCCL feature-birth genesis is invalid")
        return ledger

    @staticmethod
    def _semantic_change_requested(
        source: HCCLFeatureBirthLedger,
        context_active: Array,
        context_birth_words: Array,
        pair_descriptors: Array,
        pair_admission_mask: Array,
        pair_descriptor_matches: np.ndarray[Any, Any],
    ) -> bool:
        """Compare context births by slot and pair birth membership without slots.

        Every non-admitted destination pair maps to one uniquely live source
        record and carries that record in full.  Tracking which source births
        are used therefore compares the complete carried birth set while
        treating a pure permutation as identity-preserving.
        """

        source_context_active = _host(source.active)[
            HCCL_FEATURE_CONTEXT_START:HCCL_FEATURE_FAST_START
        ]
        source_context_births = _host(source.birth_words)[
            HCCL_FEATURE_CONTEXT_START:HCCL_FEATURE_FAST_START
        ]
        context_changed = bool(
            not np.array_equal(source_context_active, _host(context_active))
            or not np.array_equal(source_context_births, _host(context_birth_words))
        )
        destination_pairs = _host(pair_descriptors)
        destination_active = np.all(destination_pairs >= 0, axis=1)
        admissions = _host(pair_admission_mask).astype(np.bool_)
        source_active = _host(source.active)[HCCL_FEATURE_PAIR_START:].astype(np.bool_)
        source_birth_used = np.zeros((HCCL_FEATURE_PAIR_SLOTS,), dtype=np.bool_)
        pair_changed = bool(np.any(admissions))
        for local in np.flatnonzero(destination_active & ~admissions):
            sources = np.flatnonzero(pair_descriptor_matches[local])
            if len(sources) != 1:
                pair_changed = True
                continue
            source_birth_used[int(sources[0])] = True
        pair_changed |= not np.array_equal(source_birth_used, source_active)
        return context_changed or pair_changed

    def _candidate_ledger(
        self,
        source: HCCLFeatureBirthLedger,
        *,
        destination_source_clock_words: Array,
        context_active: Array,
        context_birth_words: Array,
        pair_descriptors: Array,
        pair_admission_mask: Array,
        pair_descriptor_matches: np.ndarray[Any, Any],
        semantic_change_requested: bool,
        semantic_generation_capacity_available: bool,
    ) -> HCCLFeatureBirthLedger:
        next_generation, _ = _words_successor(source.semantic_generation_words)
        destination_generation = (
            next_generation
            if semantic_change_requested and semantic_generation_capacity_available
            else source.semantic_generation_words
        )
        descriptors = self._canonical_descriptors(pair_descriptors)
        kinds = self._canonical_kind()
        active = np.zeros((HCCL_FEATURE_TOTAL_DIM,), dtype=np.bool_)
        active[:HCCL_FEATURE_CONTEXT_START] = True
        active[HCCL_FEATURE_CONTEXT_START:HCCL_FEATURE_FAST_START] = _host(
            context_active
        )
        active[HCCL_FEATURE_FAST_START:HCCL_FEATURE_PAIR_START] = True
        pair_values = _host(pair_descriptors)
        pair_admissions = _host(pair_admission_mask).astype(np.bool_)
        active[HCCL_FEATURE_PAIR_START:] = np.all(pair_values >= 0, axis=1)

        births = np.zeros((HCCL_FEATURE_TOTAL_DIM, 2), dtype=np.uint32)
        birth_sources = np.zeros((HCCL_FEATURE_TOTAL_DIM, 2), dtype=np.uint32)
        parents = np.full((HCCL_FEATURE_TOTAL_DIM, 2), -1, dtype=np.int32)
        events = np.full(
            (HCCL_FEATURE_TOTAL_DIM,),
            int(HCCLFeatureBirthEvent.INACTIVE),
            dtype=np.int32,
        )
        source_births = _host(source.birth_words)
        source_birth_sources = _host(source.birth_source_words)
        source_parents = _host(source.parents)
        source_events = _host(source.birth_event)
        source_active = _host(source.active).astype(np.bool_)
        destination_clock = _host(destination_source_clock_words)
        destination_generation_host = _host(destination_generation)

        for slot in list(range(HCCL_FEATURE_CONTEXT_START)) + list(
            range(HCCL_FEATURE_FAST_START, HCCL_FEATURE_PAIR_START)
        ):
            births[slot] = source_births[slot]
            birth_sources[slot] = source_birth_sources[slot]
            parents[slot] = source_parents[slot]
            events[slot] = source_events[slot]

        context_births = _host(context_birth_words)
        for local in range(HCCL_FEATURE_CONTEXT_DIM):
            slot = HCCL_FEATURE_CONTEXT_START + local
            if not active[slot]:
                continue
            survivor = source_active[slot] and np.array_equal(
                context_births[local],
                source_births[slot],
            )
            if survivor:
                births[slot] = source_births[slot]
                birth_sources[slot] = source_birth_sources[slot]
                parents[slot] = source_parents[slot]
                events[slot] = source_events[slot]
            else:
                births[slot] = context_births[local]
                birth_sources[slot] = destination_clock
                events[slot] = int(HCCLFeatureBirthEvent.CONTEXT_ALLOCATION)

        for local in range(HCCL_FEATURE_PAIR_SLOTS):
            slot = HCCL_FEATURE_PAIR_START + local
            if not active[slot]:
                continue
            matches = np.flatnonzero(pair_descriptor_matches[local])
            if len(matches) == 1 and not pair_admissions[local]:
                source_slot = HCCL_FEATURE_PAIR_START + int(matches[0])
                births[slot] = source_births[source_slot]
                birth_sources[slot] = source_birth_sources[source_slot]
                parents[slot] = source_parents[source_slot]
                events[slot] = source_events[source_slot]
            else:
                births[slot] = destination_generation_host
                birth_sources[slot] = destination_clock
                parents[slot] = pair_values[local]
                events[slot] = int(HCCLFeatureBirthEvent.PAIR_ADMISSION)

        unsigned = HCCLFeatureBirthLedger(
            config_token=self._config_token,
            content_token=jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8),
            schema_digest=self._schema_digest,
            agent_index=jnp.asarray(self._agent_index, dtype=jnp.int32),
            source_clock_words=destination_source_clock_words,
            semantic_generation_words=destination_generation,
            kind=jnp.asarray(kinds, dtype=jnp.int32),
            descriptor=jnp.asarray(descriptors, dtype=jnp.int32),
            birth_words=jnp.asarray(births, dtype=jnp.uint32),
            birth_source_words=jnp.asarray(birth_sources, dtype=jnp.uint32),
            parents=jnp.asarray(parents, dtype=jnp.int32),
            birth_event=jnp.asarray(events, dtype=jnp.int32),
            active=jnp.asarray(active, dtype=jnp.bool_),
        )
        return self._seal_ledger(unsigned)

    @staticmethod
    def _full_identity_matrix(
        source: HCCLFeatureBirthLedger,
        destination: HCCLFeatureBirthLedger,
    ) -> np.ndarray[Any, Any]:
        source_active = _host(source.active).astype(np.bool_)
        destination_active = _host(destination.active).astype(np.bool_)
        identity = np.ones(
            (HCCL_FEATURE_TOTAL_DIM, HCCL_FEATURE_TOTAL_DIM),
            dtype=np.bool_,
        )
        for field_name in (
            "kind",
            "descriptor",
            "birth_words",
            "birth_source_words",
            "parents",
            "birth_event",
        ):
            source_values = _host(getattr(source, field_name))
            destination_values = _host(getattr(destination, field_name))
            if source_values.ndim == 1:
                identity &= destination_values[:, None] == source_values[None, :]
            else:
                identity &= np.all(
                    destination_values[:, None, :] == source_values[None, :, :],
                    axis=2,
                )
        identity &= destination_active[:, None] & source_active[None, :]
        return identity

    def _route_map(
        self,
        source: HCCLFeatureBirthLedger,
        destination: HCCLFeatureBirthLedger,
    ) -> HCCLFeatureConsumerRouteMap:
        identity = self._full_identity_matrix(source, destination)
        destination_active = _host(destination.active).astype(np.bool_)
        source_active = _host(source.active).astype(np.bool_)
        match_count = np.sum(identity, axis=1)
        survivor = destination_active & (match_count == 1)
        newborn = destination_active & (match_count == 0)
        inactive = ~destination_active
        source_slots = np.where(
            survivor,
            np.argmax(identity, axis=1),
            -1,
        ).astype(np.int32)
        retired = source_active & ~np.any(identity, axis=0)
        unique_matches = bool(np.all(match_count <= 1))
        source_use_count = np.sum(identity, axis=0)
        unique_source_use = bool(np.all(source_use_count <= 1))
        complete = bool(
            np.all(survivor | newborn | inactive)
            and not np.any(survivor & newborn)
            and not np.any(survivor & inactive)
            and not np.any(newborn & inactive)
        )
        return HCCLFeatureConsumerRouteMap(
            source_slots=jnp.asarray(source_slots, dtype=jnp.int32),
            survivor_mask=jnp.asarray(survivor, dtype=jnp.bool_),
            newborn_mask=jnp.asarray(newborn, dtype=jnp.bool_),
            inactive_mask=jnp.asarray(inactive, dtype=jnp.bool_),
            retired_mask=jnp.asarray(retired, dtype=jnp.bool_),
            survivor_count=jnp.asarray(np.sum(survivor), dtype=jnp.int32),
            newborn_count=jnp.asarray(np.sum(newborn), dtype=jnp.int32),
            inactive_count=jnp.asarray(np.sum(inactive), dtype=jnp.int32),
            retired_count=jnp.asarray(np.sum(retired), dtype=jnp.int32),
            unique_full_identity_matches=jnp.asarray(
                unique_matches,
                dtype=jnp.bool_,
            ),
            unique_source_identity_use=jnp.asarray(
                unique_source_use,
                dtype=jnp.bool_,
            ),
            classification_complete=jnp.asarray(complete, dtype=jnp.bool_),
        )

    @staticmethod
    def _route_map_semantics_valid(route_map: HCCLFeatureConsumerRouteMap) -> bool:
        source_slots = _host(route_map.source_slots)
        survivor = _host(route_map.survivor_mask).astype(np.bool_)
        newborn = _host(route_map.newborn_mask).astype(np.bool_)
        inactive = _host(route_map.inactive_mask).astype(np.bool_)
        retired = _host(route_map.retired_mask).astype(np.bool_)
        survivor_sources = source_slots[survivor]
        unique_source_slots = len(survivor_sources) == len(
            {int(slot) for slot in survivor_sources}
        )
        classification_complete = bool(
            np.all(survivor | newborn | inactive)
            and not np.any(survivor & newborn)
            and not np.any(survivor & inactive)
            and not np.any(newborn & inactive)
        )
        return bool(
            _host_bool(route_map.unique_full_identity_matches)
            and _host_bool(route_map.unique_source_identity_use)
            and unique_source_slots
            and _host_bool(route_map.classification_complete)
            and classification_complete
            and np.all((source_slots >= 0) == survivor)
            and np.all(source_slots[~survivor] == -1)
            and int(_host(route_map.survivor_count)) == int(np.sum(survivor))
            and int(_host(route_map.newborn_count)) == int(np.sum(newborn))
            and int(_host(route_map.inactive_count)) == int(np.sum(inactive))
            and int(_host(route_map.retired_count)) == int(np.sum(retired))
            and int(np.sum(survivor | newborn | inactive)) == HCCL_FEATURE_TOTAL_DIM
        )

    def prepare_successor(
        self,
        source: HCCLFeatureBirthLedger,
        *,
        destination_source_clock_words: Array,
        context_active: Array,
        context_birth_words: Array,
        pair_descriptors: Array,
        pair_admission_mask: Array,
    ) -> HCCLFeatureConsumerRouteResult:
        """Prepare one successor and exact route map, or return ``source``.

        Static shape or dtype mistakes raise.  Dynamic invalidity—including a
        content/semantic mutation in the supplied source, bad destination
        identity input, foreign agent binding, skipped source clock, or
        exhausted generation—returns the supplied source ledger bit-for-bit.
        This stateless check owns no history and makes no replay-detection
        claim.  The rejected candidate and sealed witness remain available for
        audit.
        """

        self._require_ledger_contract(source)
        self._require_destination_contract(
            destination_source_clock_words,
            context_active,
            context_birth_words,
            pair_descriptors,
            pair_admission_mask,
        )
        if _contains_tracer(
            (
                source,
                destination_source_clock_words,
                context_active,
                context_birth_words,
                pair_descriptors,
                pair_admission_mask,
            )
        ):
            raise TypeError("HCCL feature-consumer route preparation is host/eager-only")

        source_valid = _host_bool(self.ledger_valid(source))
        expected_clock, source_clock_capacity = _words_successor(source.source_clock_words)
        clock_is_successor = source_clock_capacity and _words_equal(
            expected_clock,
            destination_source_clock_words,
        )
        pair_descriptor_matches = self._pair_descriptor_identity_matrix(
            source,
            pair_descriptors,
        )
        semantic_change_requested = self._semantic_change_requested(
            source,
            context_active,
            context_birth_words,
            pair_descriptors,
            pair_admission_mask,
            pair_descriptor_matches,
        )
        _, semantic_capacity = _words_successor(source.semantic_generation_words)
        semantic_capacity_available = (not semantic_change_requested) or semantic_capacity
        pair_inputs_valid = self._pair_transition_input_valid(
            pair_descriptors,
            pair_admission_mask,
            pair_descriptor_matches,
        )
        context_inputs_valid = self._context_input_valid(
            source,
            destination_source_clock_words,
            context_active,
            context_birth_words,
        )
        destination_inputs_valid = pair_inputs_valid and context_inputs_valid

        candidate = self._candidate_ledger(
            source,
            destination_source_clock_words=destination_source_clock_words,
            context_active=context_active,
            context_birth_words=context_birth_words,
            pair_descriptors=pair_descriptors,
            pair_admission_mask=pair_admission_mask,
            pair_descriptor_matches=pair_descriptor_matches,
            semantic_change_requested=semantic_change_requested,
            semantic_generation_capacity_available=semantic_capacity_available,
        )
        candidate_valid = _host_bool(self.ledger_valid(candidate))
        route_map = self._route_map(source, candidate)
        route_map_valid = self._route_map_semantics_valid(route_map)
        transaction_applied = bool(
            source_valid
            and clock_is_successor
            and semantic_capacity_available
            and destination_inputs_valid
            and candidate_valid
            and route_map_valid
        )
        selected = candidate if transaction_applied else source
        bare_witness = HCCLFeatureConsumerRouteWitness(
            source_content_token=source.content_token,
            destination_content_token=candidate.content_token,
            source_clock_words=source.source_clock_words,
            destination_source_clock_words=candidate.source_clock_words,
            source_semantic_generation_words=source.semantic_generation_words,
            destination_semantic_generation_words=(
                candidate.semantic_generation_words
            ),
            requested_context_active=context_active,
            requested_context_birth_words=context_birth_words,
            requested_pair_descriptors=pair_descriptors,
            requested_pair_admission_mask=pair_admission_mask,
            route_map=route_map,
            work=_fixed_work(),
            source_ledger_valid=jnp.asarray(source_valid, dtype=jnp.bool_),
            destination_inputs_valid=jnp.asarray(
                destination_inputs_valid,
                dtype=jnp.bool_,
            ),
            source_clock_is_successor=jnp.asarray(
                clock_is_successor,
                dtype=jnp.bool_,
            ),
            semantic_generation_capacity_available=jnp.asarray(
                semantic_capacity_available,
                dtype=jnp.bool_,
            ),
            candidate_ledger_valid=jnp.asarray(candidate_valid, dtype=jnp.bool_),
            route_map_valid=jnp.asarray(route_map_valid, dtype=jnp.bool_),
            semantic_bank_changed=jnp.asarray(
                semantic_change_requested,
                dtype=jnp.bool_,
            ),
            transaction_applied=jnp.asarray(transaction_applied, dtype=jnp.bool_),
            complete_source_returned=jnp.asarray(
                not transaction_applied,
                dtype=jnp.bool_,
            ),
            content_token=jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8),
        )
        witness = self._seal_witness(bare_witness)
        return HCCLFeatureConsumerRouteResult(
            ledger=selected,
            candidate_ledger=candidate,
            witness=witness,
        )

    def witness_integrity_valid(
        self,
        source: HCCLFeatureBirthLedger,
        candidate: HCCLFeatureBirthLedger,
        witness: HCCLFeatureConsumerRouteWitness,
    ) -> Bool[Array, ""]:
        """Validate a successful or rejected witness against exact ledgers."""

        self._require_ledger_contract(source)
        self._require_ledger_contract(candidate)
        self._require_witness_contract(witness)
        if _contains_tracer((source, candidate, witness)):
            raise TypeError("HCCL feature-consumer route witness is host/eager-only")
        expected_map = self._route_map(source, candidate)
        source_valid = _host_bool(self.ledger_valid(source))
        candidate_valid = _host_bool(self.ledger_valid(candidate))
        expected_clock, clock_capacity = _words_successor(source.source_clock_words)
        clock_is_successor = clock_capacity and _words_equal(
            expected_clock,
            candidate.source_clock_words,
        )
        pair_descriptor_matches = self._pair_descriptor_identity_matrix(
            source,
            witness.requested_pair_descriptors,
        )
        semantic_changed = self._semantic_change_requested(
            source,
            witness.requested_context_active,
            witness.requested_context_birth_words,
            witness.requested_pair_descriptors,
            witness.requested_pair_admission_mask,
            pair_descriptor_matches,
        )
        expected_generation, semantic_capacity = _words_successor(
            source.semantic_generation_words
        )
        semantic_capacity_available = (not semantic_changed) or semantic_capacity
        expected_destination_generation = (
            expected_generation
            if semantic_changed and semantic_capacity_available
            else source.semantic_generation_words
        )
        semantic_generation_transition_valid = _words_equal(
            expected_destination_generation,
            candidate.semantic_generation_words,
        )
        destination_inputs_valid = self._pair_transition_input_valid(
            witness.requested_pair_descriptors,
            witness.requested_pair_admission_mask,
            pair_descriptor_matches,
        ) and self._context_input_valid(
            source,
            candidate.source_clock_words,
            witness.requested_context_active,
            witness.requested_context_birth_words,
        )
        reconstructed_candidate = self._candidate_ledger(
            source,
            destination_source_clock_words=witness.destination_source_clock_words,
            context_active=witness.requested_context_active,
            context_birth_words=witness.requested_context_birth_words,
            pair_descriptors=witness.requested_pair_descriptors,
            pair_admission_mask=witness.requested_pair_admission_mask,
            pair_descriptor_matches=pair_descriptor_matches,
            semantic_change_requested=semantic_changed,
            semantic_generation_capacity_available=semantic_capacity_available,
        )
        candidate_reconstruction_valid = _tree_exact_equal(
            candidate,
            reconstructed_candidate,
        )
        route_map_valid = self._route_map_semantics_valid(expected_map)
        expected_applied = bool(
            source_valid
            and candidate_valid
            and clock_is_successor
            and semantic_capacity_available
            and semantic_generation_transition_valid
            and candidate_reconstruction_valid
            and route_map_valid
            and destination_inputs_valid
        )
        fixed_fields_valid = all(
            (
                _array_exact_equal(
                    witness.source_content_token,
                    source.content_token,
                ),
                _array_exact_equal(
                    witness.destination_content_token,
                    candidate.content_token,
                ),
                _array_exact_equal(witness.source_clock_words, source.source_clock_words),
                _array_exact_equal(
                    witness.destination_source_clock_words,
                    candidate.source_clock_words,
                ),
                _array_exact_equal(
                    witness.source_semantic_generation_words,
                    source.semantic_generation_words,
                ),
                _array_exact_equal(
                    witness.destination_semantic_generation_words,
                    candidate.semantic_generation_words,
                ),
                _tree_exact_equal(witness.route_map, expected_map),
                _tree_exact_equal(witness.work, _fixed_work()),
                _host_bool(witness.source_ledger_valid) == source_valid,
                _host_bool(witness.destination_inputs_valid)
                == destination_inputs_valid,
                _host_bool(witness.source_clock_is_successor) == clock_is_successor,
                _host_bool(witness.semantic_generation_capacity_available)
                == semantic_capacity_available,
                _host_bool(witness.candidate_ledger_valid) == candidate_valid,
                _host_bool(witness.route_map_valid) == route_map_valid,
                _host_bool(witness.semantic_bank_changed) == semantic_changed,
                semantic_generation_transition_valid,
                candidate_reconstruction_valid,
                _host_bool(witness.transaction_applied) == expected_applied,
                _host_bool(witness.complete_source_returned) == (not expected_applied),
                _array_exact_equal(
                    witness.content_token,
                    self._witness_content_token(witness),
                ),
            )
        )
        return jnp.asarray(fixed_fields_valid, dtype=jnp.bool_)

    def result_integrity_valid(
        self,
        source: HCCLFeatureBirthLedger,
        result: HCCLFeatureConsumerRouteResult,
    ) -> Bool[Array, ""]:
        """Validate the witness and the ledger actually selected by the result."""

        self._require_ledger_contract(source)
        self._require_result_contract(result)
        if _contains_tracer((source, result)):
            raise TypeError("HCCL feature-consumer route result is host/eager-only")
        witness_valid = _host_bool(
            self.witness_integrity_valid(
                source,
                result.candidate_ledger,
                result.witness,
            )
        )
        applied = _host_bool(result.witness.transaction_applied)
        expected_selected = result.candidate_ledger if applied else source
        selected_is_expected = _tree_exact_equal(result.ledger, expected_selected)
        source_was_returned = _tree_exact_equal(result.ledger, source)
        source_return_claim_valid = (
            _host_bool(result.witness.complete_source_returned)
            == source_was_returned
        )
        return jnp.asarray(
            witness_valid and selected_is_expected and source_return_claim_valid,
            dtype=jnp.bool_,
        )


__all__ = [
    "HCCL_FEATURE_BIRTH_LEDGER_SCHEMA",
    "HCCL_FEATURE_CONSUMER_ROUTE_CONFIG_SCHEMA",
    "HCCL_FEATURE_CONSUMER_ROUTE_FULL_CONSUMER_ROUTING_CLAIMED",
    "HCCL_FEATURE_CONSUMER_ROUTE_SCIENTIFIC_PROMOTION_ALLOWED",
    "HCCL_FEATURE_CONSUMER_ROUTE_STATUS",
    "HCCL_FEATURE_CONTEXT_DIM",
    "HCCL_FEATURE_CONTEXT_START",
    "HCCL_FEATURE_FAST_DIM",
    "HCCL_FEATURE_FAST_START",
    "HCCL_FEATURE_PAIR_SLOTS",
    "HCCL_FEATURE_PAIR_START",
    "HCCL_FEATURE_PHYSICAL_DIM",
    "HCCL_FEATURE_ROUTE_MAP_SCHEMA",
    "HCCL_FEATURE_ROUTE_WITNESS_SCHEMA",
    "HCCL_FEATURE_TOTAL_DIM",
    "HCCLFeatureBirthEvent",
    "HCCLFeatureBirthLedger",
    "HCCLFeatureConsumerRoute",
    "HCCLFeatureConsumerRouteMap",
    "HCCLFeatureConsumerRouteResult",
    "HCCLFeatureConsumerRouteWitness",
    "HCCLFeatureConsumerRouteWork",
    "HCCLFeatureKind",
]
