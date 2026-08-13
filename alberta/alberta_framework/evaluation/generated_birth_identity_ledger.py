"""Development-only sidecar identities for generated compositional features.

The compositional learner exposes a public fixed-shape curation trace, but it
does not expose stable birth identifiers.  This module keeps a host-only ledger
beside the learner state.  It does not add a field to
``CompositionalFeatureState`` and it never enters a learner update.  A separate
adapter is required to bind the public trace to this ledger.

Every new lifetime identity is a SHA-256 digest of a target- and outcome-blind domain:
the evaluator namespace, paired development life seed, learner step, named
event channel, destination slot, and within-channel ordinal.  SHA-256 provides
cryptographic collision resistance, not a mathematical collision-impossibility
guarantee.  The ledger rejects collisions within the current transition scope:
pre-transition live identities, stable raw-source tokens, and every new birth,
including an intermediate birth overwritten later in the same transition.  It
does not retain dead identity history and therefore makes no all-history
collision-observation claim.

One transition follows the production ordering exactly.  Promotion first
transfers the source candidate identity and proposal-time parent snapshots to
the active destination.  The promoted candidate is then refreshed against the
post-root/pre-cascade active bank.  Active descendants are cascade-refilled.
Finally, any candidate whose proposal-time parent identity changed receives a
new *structural-parent rebound* identity when the rebound remains inside the
depth budget.  If the rebound would be too deep, the candidate instead receives
a separately ordered over-depth-regeneration birth and a fresh local
descriptor.  A rebound or regeneration wins even for a candidate refreshed
earlier in the same transition.  A plain rebound preserves the current local
op, parent slots, and generator provenance while recomputing depth from the
final active parents.  It is not a fresh generated local descriptor and makes
no erasure or learning-outcome claim.

Active and candidate structural-lifetime records retain exact op, parent slot
indices, ordered parent identities, derived depth, and generator-policy
provenance.  Raw rows bind a stable raw-source token instead of an all-zero
snapshot.  An unchanged active feature whose parent identity changed is
rejected as a missing cascade refill, while an unrefreshed local parent or op
change is rejected rather than mislabeled a rebound.  Theta is deliberately
excluded because it may adapt continuously within one structural lifetime;
the ledger therefore does not claim exact functional-expression identity.

All arrays have fixed shapes, exact byte accounting, immutable-bytes-backed host copies,
integrity hashes, and a strict validator that rebuilds the transaction from
separately supplied inputs.

Crucially, these checks authenticate the sidecar transition only.  Root,
cascade, refresh, descriptor, and parent events remain caller supplied.  A
public core curation trace now exists, but this module does not consume or bind
it, so the caller events are not authenticated as learner outputs.
This development module grants no execution, runner, artifact-write, evidence,
or scientific-promotion authority.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from alberta_framework.core.compositional_features import (
    DEFAULT_GENERATOR_META_POLICY_NAMES as _CORE_GENERATOR_META_POLICY_NAMES,
)
from alberta_framework.core.compositional_features import (
    FIXED_GENERATOR_POLICY_PLACEHOLDER,
    NUM_OPS,
    OP_RAW,
)

GENERATED_BIRTH_IDENTITY_LEDGER_SCHEMA = (
    "alberta.generated-structural-lifetime-identity-ledger.development.v3"
)
GENERATED_BIRTH_IDENTITY_LEDGER_STATUS = (
    "DEVELOPMENT_RUNNER_SIDE_CALLER_EVENTS_UNAUTHENTICATED_NO_AUTHORITY"
)
GENERATED_BIRTH_IDENTITY_LEDGER_V4_SCHEMA = (
    "alberta.generated-structural-lifetime-identity-ledger.development.v4"
)
GENERATED_BIRTH_IDENTITY_LEDGER_V4_STATUS = (
    "DEVELOPMENT_UINT64_WORD_COUNTER_CALLER_EVENTS_UNAUTHENTICATED_NO_AUTHORITY"
)

INITIAL_ACTIVE_CHANNEL = "initial_active"
INITIAL_CANDIDATE_CHANNEL = "initial_candidate"
RAW_SOURCE_IDENTITY_CHANNEL = "raw_observation_source_identity"
PROMOTION_TRANSFER_CHANNEL = "promotion_transfer_existing_candidate_identity"
DIRECT_ACTIVE_REPLACEMENT_CHANNEL = "direct_active_replacement"
CASCADE_ACTIVE_REFILL_CHANNEL = "cascade_active_refill"
ORDINARY_CANDIDATE_REFRESH_CHANNEL = "ordinary_candidate_refresh"
POST_PROMOTION_CANDIDATE_REFRESH_CHANNEL = "post_promotion_candidate_refresh"
CANDIDATE_PARENT_REBOUND_CHANNEL = "candidate_active_parent_identity_rebound"
CANDIDATE_OVERDEPTH_REGENERATION_CHANNEL = "candidate_overdepth_regeneration"

GENERATED_BIRTH_IDENTITY_GENERATOR_POLICY_MANIFEST = (
    "random_product_safe",
    "mutation_product_nominal",
    "residual_tanh",
    "residual_gated_aggressive",
)
if _CORE_GENERATOR_META_POLICY_NAMES != GENERATED_BIRTH_IDENTITY_GENERATOR_POLICY_MANIFEST:
    raise RuntimeError("public core generator-policy manifest drifted from ledger schema v3")

GENERATED_BIRTH_IDENTITY_CHANNELS = (
    INITIAL_ACTIVE_CHANNEL,
    INITIAL_CANDIDATE_CHANNEL,
    RAW_SOURCE_IDENTITY_CHANNEL,
    PROMOTION_TRANSFER_CHANNEL,
    DIRECT_ACTIVE_REPLACEMENT_CHANNEL,
    CASCADE_ACTIVE_REFILL_CHANNEL,
    ORDINARY_CANDIDATE_REFRESH_CHANNEL,
    POST_PROMOTION_CANDIDATE_REFRESH_CHANNEL,
    CANDIDATE_PARENT_REBOUND_CHANNEL,
    CANDIDATE_OVERDEPTH_REGENERATION_CHANNEL,
)

_NEW_ID_CHANNELS = frozenset(GENERATED_BIRTH_IDENTITY_CHANNELS) - {PROMOTION_TRANSFER_CHANNEL}
_IDENTITY_BYTES = 32
_INT32_MAX = 2**31 - 1
_INT64_MAX = 2**63 - 1
_STATE_NBYTES_FORMULA = "117 * (active_slots + candidate_slots)"
_ASSIGNMENT_NBYTES_FORMULA = "32 * (3 * active_slots + 4 * candidate_slots)"
_EVENT_NBYTES_FORMULA = "23 * active_slots + 45 * candidate_slots"
_NAMESPACE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}\Z")

_EVENT_SCALAR_FIELD_NAMES = (
    "schema",
    "status",
    "namespace",
    "paired_development_life_seed",
    "learner_step",
    "channel_manifest",
    "generator_policy_sampled",
    "generator_policy_id",
    "promotion_active_slot",
    "promotion_candidate_slot",
    "direct_active_replacement_slot",
    "ordinary_candidate_refresh_slot",
    "post_promotion_candidate_refresh_slot",
)
_EVENT_ARRAY_FIELD_NAMES = (
    "promotion_active_mask",
    "promotion_candidate_mask",
    "direct_active_replacement_mask",
    "cascade_refill_mask",
    "ordinary_candidate_refresh_mask",
    "post_promotion_candidate_refresh_mask",
    "candidate_rebound_mask",
    "candidate_overdepth_regeneration_mask",
    "active_parent_a",
    "active_parent_b",
    "active_ops",
    "active_depth",
    "active_generator_policy",
    "candidate_staged_parent_a",
    "candidate_staged_parent_b",
    "candidate_staged_ops",
    "candidate_staged_depth",
    "candidate_staged_generator_policy",
    "candidate_parent_a",
    "candidate_parent_b",
    "candidate_ops",
    "candidate_depth",
    "candidate_generator_policy",
)

UInt8Array = NDArray[np.uint8]
Int32Array = NDArray[np.int32]
UInt32Array = NDArray[np.uint32]
BoolArray = NDArray[np.bool_]


class GeneratedBirthIdentityLedgerConstructionError(ValueError):
    """A fixed-shape ledger input or integrity contract failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GeneratedBirthIdentityLedgerConstructionError(message)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _array_record(value: np.ndarray[Any, Any]) -> dict[str, object]:
    contiguous = np.ascontiguousarray(value)
    return {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "bytes_hex": contiguous.tobytes(order="C").hex(),
    }


def _readonly_copy(value: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    contiguous = np.ascontiguousarray(value)
    immutable_bytes = contiguous.tobytes(order="C")
    result = np.frombuffer(immutable_bytes, dtype=contiguous.dtype).reshape(contiguous.shape)
    return result


def _has_immutable_bytes_backing(value: np.ndarray[Any, Any]) -> bool:
    base: object = value
    while type(base) is np.ndarray:
        array = base
        if array.flags.owndata:
            return False
        base = array.base
    if type(base) is bytes:
        return True
    return type(base) is memoryview and cast(memoryview, base).readonly


def _exact_array(
    value: object,
    *,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
    name: str,
    require_immutable: bool = False,
) -> np.ndarray[Any, Any]:
    _require(type(value) is np.ndarray, f"{name} must be an exact numpy array")
    array = cast(np.ndarray[Any, Any], value)
    _require(array.dtype == dtype, f"{name} must have dtype {dtype.name}")
    _require(array.shape == shape, f"{name} must have shape {shape}")
    _require(array.flags.c_contiguous, f"{name} must be C-contiguous")
    if require_immutable:
        _require(not array.flags.writeable, f"{name} must be an immutable read-only array")
        _require(
            _has_immutable_bytes_backing(array),
            f"{name} must have immutable bytes backing",
        )
    return _readonly_copy(array)


def _exact_int(value: object, *, name: str, lower: int, upper: int) -> int:
    _require(type(value) is int, f"{name} must be an exact Python integer")
    integer = cast(int, value)
    _require(lower <= integer <= upper, f"{name} is outside [{lower}, {upper}]")
    return integer


def _exact_bool(value: object, *, name: str) -> bool:
    _require(type(value) is bool, f"{name} must be an exact Python boolean")
    return cast(bool, value)


def _valid_sha256(value: object) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedBirthIdentityLedgerConfig:
    """Static sidecar shape and immutable non-authority disclosures."""

    namespace: str
    active_slots: int
    candidate_slots: int
    raw_feature_slots: int
    max_depth: int
    learn_generator_resources: bool
    generator_policy_count: int = len(GENERATED_BIRTH_IDENTITY_GENERATOR_POLICY_MANIFEST)
    generator_policy_manifest: tuple[str, ...] = GENERATED_BIRTH_IDENTITY_GENERATOR_POLICY_MANIFEST
    channel_manifest: tuple[str, ...] = GENERATED_BIRTH_IDENTITY_CHANNELS
    schema: str = GENERATED_BIRTH_IDENTITY_LEDGER_SCHEMA
    status: str = GENERATED_BIRTH_IDENTITY_LEDGER_STATUS
    development_only: bool = True
    host_only_not_jittable: bool = True
    fixed_shape: bool = True
    runner_side_state_only: bool = True
    compositional_feature_state_fields_added: bool = False
    target_inputs_accepted: bool = False
    outcome_inputs_accepted: bool = False
    caller_supplied_events_authenticated: bool = False
    public_core_event_trace_required_for_authentication: bool = True
    public_core_event_trace_available: bool = True
    public_core_event_trace_consumed: bool = False
    structural_lifetime_descriptor_complete: bool = True
    active_parent_snapshot_complete: bool = True
    candidate_parent_snapshot_complete: bool = True
    raw_source_identity_bound: bool = True
    theta_bound_to_structural_lifetime_identity: bool = False
    theta_may_adapt_within_structural_lifetime: bool = True
    exact_functional_expression_identity_claimed: bool = False
    cryptographic_collision_resistance_claimed: bool = True
    cryptographic_collision_impossibility_claimed: bool = False
    dead_identity_history_retained: bool = False
    historical_global_uniqueness_claimed: bool = False
    lifecycle_prerequisite_complete: bool = False
    execution_authorized: bool = False
    runner_authorized: bool = False
    artifact_writes_authorized: bool = False
    evidence_authorized: bool = False
    scientific_promotion_allowed: bool = False

    def __post_init__(self) -> None:
        if type(self.namespace) is not str or _NAMESPACE_PATTERN.fullmatch(self.namespace) is None:
            raise ValueError("namespace must be a nonempty canonical ASCII identifier")
        for name in (
            "active_slots",
            "candidate_slots",
            "raw_feature_slots",
            "max_depth",
            "generator_policy_count",
        ):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an exact Python integer")
        if not 2 <= self.active_slots <= _INT32_MAX:
            raise ValueError("active_slots must be inside [2, int32_max]")
        if not 0 <= self.candidate_slots <= _INT32_MAX:
            raise ValueError("candidate_slots must be inside [0, int32_max]")
        if self.active_slots + self.candidate_slots > _INT32_MAX:
            raise ValueError("combined bank size must fit the core int32 event-count domain")
        if not 1 <= self.raw_feature_slots < self.active_slots:
            raise ValueError("raw_feature_slots must be inside the active bank")
        if not 1 <= self.max_depth <= _INT32_MAX:
            raise ValueError("max_depth must be inside [1, int32_max]")
        if self.generator_policy_count != len(GENERATED_BIRTH_IDENTITY_GENERATOR_POLICY_MANIFEST):
            raise ValueError("generator_policy_count must bind the public core policy count")
        if self.generator_policy_manifest != GENERATED_BIRTH_IDENTITY_GENERATOR_POLICY_MANIFEST:
            raise ValueError("generator_policy_manifest must bind the public core manifest")
        if _CORE_GENERATOR_META_POLICY_NAMES != self.generator_policy_manifest:
            raise ValueError("live public core generator-policy manifest drifted")
        if self.channel_manifest != GENERATED_BIRTH_IDENTITY_CHANNELS:
            raise ValueError("channel_manifest must be the canonical channel manifest")
        if self.schema != GENERATED_BIRTH_IDENTITY_LEDGER_SCHEMA:
            raise ValueError("schema must be the canonical schema")
        if self.status != GENERATED_BIRTH_IDENTITY_LEDGER_STATUS:
            raise ValueError("status must be the canonical status")
        bool_fields = (
            "learn_generator_resources",
            "development_only",
            "host_only_not_jittable",
            "fixed_shape",
            "runner_side_state_only",
            "compositional_feature_state_fields_added",
            "target_inputs_accepted",
            "outcome_inputs_accepted",
            "caller_supplied_events_authenticated",
            "public_core_event_trace_required_for_authentication",
            "public_core_event_trace_available",
            "public_core_event_trace_consumed",
            "structural_lifetime_descriptor_complete",
            "active_parent_snapshot_complete",
            "candidate_parent_snapshot_complete",
            "raw_source_identity_bound",
            "theta_bound_to_structural_lifetime_identity",
            "theta_may_adapt_within_structural_lifetime",
            "exact_functional_expression_identity_claimed",
            "cryptographic_collision_resistance_claimed",
            "cryptographic_collision_impossibility_claimed",
            "dead_identity_history_retained",
            "historical_global_uniqueness_claimed",
            "lifecycle_prerequisite_complete",
            "execution_authorized",
            "runner_authorized",
            "artifact_writes_authorized",
            "evidence_authorized",
            "scientific_promotion_allowed",
        )
        for name in bool_fields:
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact Python boolean")
        required_true = (
            self.development_only,
            self.host_only_not_jittable,
            self.fixed_shape,
            self.runner_side_state_only,
            self.public_core_event_trace_required_for_authentication,
            self.public_core_event_trace_available,
            self.structural_lifetime_descriptor_complete,
            self.active_parent_snapshot_complete,
            self.candidate_parent_snapshot_complete,
            self.raw_source_identity_bound,
            self.theta_may_adapt_within_structural_lifetime,
            self.cryptographic_collision_resistance_claimed,
        )
        if not all(required_true):
            raise ValueError("development sidecar integrity requirements must remain enabled")
        if self.cryptographic_collision_impossibility_claimed:
            raise ValueError("SHA-256 identities cannot claim collision impossibility")
        forbidden_claims = (
            self.compositional_feature_state_fields_added,
            self.target_inputs_accepted,
            self.outcome_inputs_accepted,
            self.caller_supplied_events_authenticated,
            self.public_core_event_trace_consumed,
            self.theta_bound_to_structural_lifetime_identity,
            self.exact_functional_expression_identity_claimed,
            self.dead_identity_history_retained,
            self.historical_global_uniqueness_claimed,
            self.lifecycle_prerequisite_complete,
            self.execution_authorized,
            self.runner_authorized,
            self.artifact_writes_authorized,
            self.evidence_authorized,
            self.scientific_promotion_allowed,
        )
        if any(forbidden_claims):
            raise ValueError("development identity ledger cannot grant authority or causal claims")


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedBirthIdentityLedgerState:
    """Fixed-shape runner-side identity state; never learner state."""

    schema: str
    status: str
    config_sha256: str
    namespace: str
    paired_development_life_seed: int
    learner_step: int
    active_identity: UInt8Array
    active_parent_a: Int32Array
    active_parent_b: Int32Array
    active_parent_identity_snapshot: UInt8Array
    active_ops: Int32Array
    active_depth: Int32Array
    active_generator_policy: Int32Array
    active_generator_policy_sampled: BoolArray
    candidate_identity: UInt8Array
    candidate_parent_a: Int32Array
    candidate_parent_b: Int32Array
    candidate_parent_identity_snapshot: UInt8Array
    candidate_ops: Int32Array
    candidate_depth: Int32Array
    candidate_generator_policy: Int32Array
    candidate_generator_policy_sampled: BoolArray
    persistent_array_nbytes: int
    expected_persistent_array_nbytes_formula: str
    integrity_sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedBirthIdentityEvent:
    """Caller-supplied fixed-shape event declaration for one learner step."""

    schema: str
    status: str
    namespace: str
    paired_development_life_seed: int
    learner_step: int
    channel_manifest: tuple[str, ...]
    generator_policy_sampled: bool
    generator_policy_id: int
    promotion_active_slot: int
    promotion_candidate_slot: int
    direct_active_replacement_slot: int
    ordinary_candidate_refresh_slot: int
    post_promotion_candidate_refresh_slot: int
    promotion_active_mask: BoolArray
    promotion_candidate_mask: BoolArray
    direct_active_replacement_mask: BoolArray
    cascade_refill_mask: BoolArray
    ordinary_candidate_refresh_mask: BoolArray
    post_promotion_candidate_refresh_mask: BoolArray
    candidate_rebound_mask: BoolArray
    candidate_overdepth_regeneration_mask: BoolArray
    active_parent_a: Int32Array
    active_parent_b: Int32Array
    active_ops: Int32Array
    active_depth: Int32Array
    active_generator_policy: Int32Array
    candidate_staged_parent_a: Int32Array
    candidate_staged_parent_b: Int32Array
    candidate_staged_ops: Int32Array
    candidate_staged_depth: Int32Array
    candidate_staged_generator_policy: Int32Array
    candidate_parent_a: Int32Array
    candidate_parent_b: Int32Array
    candidate_ops: Int32Array
    candidate_depth: Int32Array
    candidate_generator_policy: Int32Array
    integrity_sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedBirthIdentityAssignments:
    """All fixed-shape identities assigned during one transition."""

    promotion_transfer_active_identity: UInt8Array
    direct_active_birth_identity: UInt8Array
    cascade_active_birth_identity: UInt8Array
    ordinary_candidate_birth_identity: UInt8Array
    post_promotion_candidate_birth_identity: UInt8Array
    candidate_rebound_identity: UInt8Array
    candidate_overdepth_regeneration_identity: UInt8Array
    persistent_array_nbytes: int
    expected_persistent_array_nbytes_formula: str
    integrity_sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedBirthIdentityLedgerAudit:
    """Integrity, order, resource, and non-authority record."""

    schema: str
    status: str
    config_sha256: str
    pre_state_sha256: str
    event_sha256: str
    assignments_sha256: str
    post_state_sha256: str
    transaction_sha256: str
    channel_manifest: tuple[str, ...]
    promotion_transfer_count: int
    direct_active_birth_count: int
    cascade_active_birth_count: int
    ordinary_candidate_birth_count: int
    post_promotion_candidate_birth_count: int
    candidate_rebound_count: int
    candidate_overdepth_regeneration_count: int
    just_refreshed_candidate_rebound_count: int
    just_refreshed_candidate_overdepth_regeneration_count: int
    applied_identity_event_count: int
    state_persistent_array_nbytes: int
    expected_state_persistent_array_nbytes: int
    expected_state_persistent_array_nbytes_formula: str
    assignment_persistent_array_nbytes: int
    expected_assignment_persistent_array_nbytes: int
    expected_assignment_persistent_array_nbytes_formula: str
    event_fixed_array_nbytes: int
    expected_event_fixed_array_nbytes: int
    expected_event_fixed_array_nbytes_formula: str
    identity_array_noop_with_monotone_step_advance: bool
    post_promotion_refresh_then_cascade_then_candidate_resolution: bool
    structural_lifetime_descriptor_complete: bool
    active_parent_snapshot_complete: bool
    candidate_parent_snapshot_complete: bool
    raw_source_identity_bound: bool
    active_lineage_consistent: bool
    theta_bound_to_structural_lifetime_identity: bool
    theta_may_adapt_within_structural_lifetime: bool
    exact_functional_expression_identity_claimed: bool
    transition_collision_observed: bool
    dead_identity_history_retained: bool
    historical_global_uniqueness_claimed: bool
    collision_resistance_primitive: str
    cryptographic_collision_impossibility_claimed: bool
    runner_side_state_only: bool
    compositional_feature_state_fields_added: bool
    caller_supplied_events_authenticated: bool
    public_core_event_trace_required_for_authentication: bool
    public_core_event_trace_available: bool
    public_core_event_trace_consumed: bool
    lifecycle_prerequisite_complete: bool
    execution_authorized: bool
    runner_authorized: bool
    artifact_writes_authorized: bool
    evidence_authorized: bool
    scientific_promotion_allowed: bool


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedBirthIdentityTransaction:
    """One sidecar transition and its complete fixed-shape assignments."""

    assignments: GeneratedBirthIdentityAssignments
    post_state: GeneratedBirthIdentityLedgerState
    audit: GeneratedBirthIdentityLedgerAudit


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedBirthIdentityValidation:
    """Successful result of strict independent canonical reconstruction."""

    valid: bool
    canonical_transaction_sha256: str
    supplied_transaction_sha256: str
    caller_supplied_events_authenticated: bool
    lifecycle_prerequisite_complete: bool
    execution_authorized: bool
    runner_authorized: bool
    artifact_writes_authorized: bool
    evidence_authorized: bool
    scientific_promotion_allowed: bool

    def __post_init__(self) -> None:
        for name in (
            "valid",
            "caller_supplied_events_authenticated",
            "lifecycle_prerequisite_complete",
            "execution_authorized",
            "runner_authorized",
            "artifact_writes_authorized",
            "evidence_authorized",
            "scientific_promotion_allowed",
        ):
            _exact_bool(getattr(self, name), name=f"validation {name}")
        _require(self.valid, "successful validation must remain true")
        _require(
            not any(
                (
                    self.caller_supplied_events_authenticated,
                    self.lifecycle_prerequisite_complete,
                    self.execution_authorized,
                    self.runner_authorized,
                    self.artifact_writes_authorized,
                    self.evidence_authorized,
                    self.scientific_promotion_allowed,
                )
            ),
            "development validation cannot grant authority",
        )
        _require(
            _valid_sha256(self.canonical_transaction_sha256),
            "canonical validation hash is malformed",
        )
        _require(
            _valid_sha256(self.supplied_transaction_sha256),
            "supplied validation hash is malformed",
        )


def _config_payload(config: GeneratedBirthIdentityLedgerConfig) -> dict[str, object]:
    return {field.name: getattr(config, field.name) for field in dataclasses.fields(config)}


def _config_sha256(config: GeneratedBirthIdentityLedgerConfig) -> str:
    return _sha256(_config_payload(config))


def _state_payload(state: GeneratedBirthIdentityLedgerState) -> dict[str, object]:
    return {
        "schema": state.schema,
        "status": state.status,
        "config_sha256": state.config_sha256,
        "namespace": state.namespace,
        "paired_development_life_seed": state.paired_development_life_seed,
        "learner_step": state.learner_step,
        "active_identity": _array_record(state.active_identity),
        "active_parent_a": _array_record(state.active_parent_a),
        "active_parent_b": _array_record(state.active_parent_b),
        "active_parent_identity_snapshot": _array_record(state.active_parent_identity_snapshot),
        "active_ops": _array_record(state.active_ops),
        "active_depth": _array_record(state.active_depth),
        "active_generator_policy": _array_record(state.active_generator_policy),
        "active_generator_policy_sampled": _array_record(state.active_generator_policy_sampled),
        "candidate_identity": _array_record(state.candidate_identity),
        "candidate_parent_a": _array_record(state.candidate_parent_a),
        "candidate_parent_b": _array_record(state.candidate_parent_b),
        "candidate_parent_identity_snapshot": _array_record(
            state.candidate_parent_identity_snapshot
        ),
        "candidate_ops": _array_record(state.candidate_ops),
        "candidate_depth": _array_record(state.candidate_depth),
        "candidate_generator_policy": _array_record(state.candidate_generator_policy),
        "candidate_generator_policy_sampled": _array_record(
            state.candidate_generator_policy_sampled
        ),
        "persistent_array_nbytes": state.persistent_array_nbytes,
        "expected_persistent_array_nbytes_formula": (
            state.expected_persistent_array_nbytes_formula
        ),
    }


def generated_birth_identity_ledger_state_sha256(
    state: GeneratedBirthIdentityLedgerState,
) -> str:
    """Hash every semantic state field except the self-hash."""

    return _sha256(_state_payload(state))


def _event_payload(event: GeneratedBirthIdentityEvent) -> dict[str, object]:
    payload = {name: getattr(event, name) for name in _EVENT_SCALAR_FIELD_NAMES}
    payload.update({name: _array_record(getattr(event, name)) for name in _EVENT_ARRAY_FIELD_NAMES})
    return payload


def generated_birth_identity_event_sha256(event: GeneratedBirthIdentityEvent) -> str:
    """Hash every event field except the self-hash."""

    return _sha256(_event_payload(event))


def _assignments_payload(
    assignments: GeneratedBirthIdentityAssignments,
) -> dict[str, object]:
    return {
        "promotion_transfer_active_identity": _array_record(
            assignments.promotion_transfer_active_identity
        ),
        "direct_active_birth_identity": _array_record(assignments.direct_active_birth_identity),
        "cascade_active_birth_identity": _array_record(assignments.cascade_active_birth_identity),
        "ordinary_candidate_birth_identity": _array_record(
            assignments.ordinary_candidate_birth_identity
        ),
        "post_promotion_candidate_birth_identity": _array_record(
            assignments.post_promotion_candidate_birth_identity
        ),
        "candidate_rebound_identity": _array_record(assignments.candidate_rebound_identity),
        "candidate_overdepth_regeneration_identity": _array_record(
            assignments.candidate_overdepth_regeneration_identity
        ),
        "persistent_array_nbytes": assignments.persistent_array_nbytes,
        "expected_persistent_array_nbytes_formula": (
            assignments.expected_persistent_array_nbytes_formula
        ),
    }


def _assignments_sha256(assignments: GeneratedBirthIdentityAssignments) -> str:
    return _sha256(_assignments_payload(assignments))


def _validate_assignments(
    config: GeneratedBirthIdentityLedgerConfig,
    assignments: GeneratedBirthIdentityAssignments,
) -> None:
    _require(
        type(assignments) is GeneratedBirthIdentityAssignments,
        "assignment type is invalid",
    )
    specifications = (
        (
            "promotion_transfer_active_identity",
            (config.active_slots, _IDENTITY_BYTES),
        ),
        ("direct_active_birth_identity", (config.active_slots, _IDENTITY_BYTES)),
        ("cascade_active_birth_identity", (config.active_slots, _IDENTITY_BYTES)),
        (
            "ordinary_candidate_birth_identity",
            (config.candidate_slots, _IDENTITY_BYTES),
        ),
        (
            "post_promotion_candidate_birth_identity",
            (config.candidate_slots, _IDENTITY_BYTES),
        ),
        ("candidate_rebound_identity", (config.candidate_slots, _IDENTITY_BYTES)),
        (
            "candidate_overdepth_regeneration_identity",
            (config.candidate_slots, _IDENTITY_BYTES),
        ),
    )
    actual_nbytes = 0
    for name, shape in specifications:
        array = getattr(assignments, name)
        _exact_array(
            array,
            dtype=np.dtype(np.uint8),
            shape=shape,
            name=name,
            require_immutable=True,
        )
        actual_nbytes += array.nbytes
    expected_nbytes = _IDENTITY_BYTES * (3 * config.active_slots + 4 * config.candidate_slots)
    _require(actual_nbytes == expected_nbytes, "assignment array byte count is invalid")
    _exact_int(
        assignments.persistent_array_nbytes,
        name="assignment persistent_array_nbytes",
        lower=0,
        upper=_INT64_MAX,
    )
    _require(
        assignments.persistent_array_nbytes == expected_nbytes,
        "assignment persistent_array_nbytes is stale",
    )
    _require(
        assignments.expected_persistent_array_nbytes_formula == _ASSIGNMENT_NBYTES_FORMULA,
        "assignment byte formula is stale",
    )
    _require(
        _valid_sha256(assignments.integrity_sha256),
        "assignment integrity hash is malformed",
    )
    _require(
        assignments.integrity_sha256 == _assignments_sha256(assignments),
        "assignment integrity hash does not reconstruct",
    )


def _audit_payload(
    audit: GeneratedBirthIdentityLedgerAudit,
    *,
    include_transaction_sha256: bool,
) -> dict[str, object]:
    result = {field.name: getattr(audit, field.name) for field in dataclasses.fields(audit)}
    if not include_transaction_sha256:
        result.pop("transaction_sha256")
    return result


def _transaction_payload(
    transaction: GeneratedBirthIdentityTransaction,
    *,
    include_transaction_sha256: bool,
) -> dict[str, object]:
    return {
        "assignments": {
            **_assignments_payload(transaction.assignments),
            "integrity_sha256": transaction.assignments.integrity_sha256,
        },
        "post_state": {
            **_state_payload(transaction.post_state),
            "integrity_sha256": transaction.post_state.integrity_sha256,
        },
        "audit": _audit_payload(
            transaction.audit,
            include_transaction_sha256=include_transaction_sha256,
        ),
    }


def generated_birth_identity_transaction_sha256(
    transaction: GeneratedBirthIdentityTransaction,
) -> str:
    """Hash a complete transaction excluding its own transaction hash."""

    return _sha256(_transaction_payload(transaction, include_transaction_sha256=False))


def _validate_audit(
    config: GeneratedBirthIdentityLedgerConfig,
    audit: GeneratedBirthIdentityLedgerAudit,
) -> None:
    _require(type(audit) is GeneratedBirthIdentityLedgerAudit, "audit type is invalid")
    _require(type(audit.schema) is str, "audit schema must be an exact string")
    _require(type(audit.status) is str, "audit status must be an exact string")
    _require(audit.schema == config.schema, "audit schema is stale")
    _require(audit.status == config.status, "audit status is stale")
    _require(audit.config_sha256 == _config_sha256(config), "audit config hash is stale")
    for name in (
        "config_sha256",
        "pre_state_sha256",
        "event_sha256",
        "assignments_sha256",
        "post_state_sha256",
        "transaction_sha256",
    ):
        _require(_valid_sha256(getattr(audit, name)), f"audit {name} is malformed")
    _require(
        type(audit.channel_manifest) is tuple
        and audit.channel_manifest == GENERATED_BIRTH_IDENTITY_CHANNELS,
        "audit channel manifest is stale",
    )
    count_bounds = {
        "promotion_transfer_count": 1,
        "direct_active_birth_count": 1,
        "cascade_active_birth_count": config.active_slots,
        "ordinary_candidate_birth_count": 1,
        "post_promotion_candidate_birth_count": 1,
        "candidate_rebound_count": config.candidate_slots,
        "candidate_overdepth_regeneration_count": config.candidate_slots,
        "just_refreshed_candidate_rebound_count": 1,
        "just_refreshed_candidate_overdepth_regeneration_count": 1,
        "applied_identity_event_count": _INT32_MAX,
    }
    for name, upper in count_bounds.items():
        _exact_int(getattr(audit, name), name=f"audit {name}", lower=0, upper=upper)
    for name in (
        "state_persistent_array_nbytes",
        "expected_state_persistent_array_nbytes",
        "assignment_persistent_array_nbytes",
        "expected_assignment_persistent_array_nbytes",
        "event_fixed_array_nbytes",
        "expected_event_fixed_array_nbytes",
    ):
        _exact_int(getattr(audit, name), name=f"audit {name}", lower=0, upper=_INT64_MAX)
    bool_fields = (
        "identity_array_noop_with_monotone_step_advance",
        "post_promotion_refresh_then_cascade_then_candidate_resolution",
        "structural_lifetime_descriptor_complete",
        "active_parent_snapshot_complete",
        "candidate_parent_snapshot_complete",
        "raw_source_identity_bound",
        "active_lineage_consistent",
        "theta_bound_to_structural_lifetime_identity",
        "theta_may_adapt_within_structural_lifetime",
        "exact_functional_expression_identity_claimed",
        "transition_collision_observed",
        "dead_identity_history_retained",
        "historical_global_uniqueness_claimed",
        "cryptographic_collision_impossibility_claimed",
        "runner_side_state_only",
        "compositional_feature_state_fields_added",
        "caller_supplied_events_authenticated",
        "public_core_event_trace_required_for_authentication",
        "public_core_event_trace_available",
        "public_core_event_trace_consumed",
        "lifecycle_prerequisite_complete",
        "execution_authorized",
        "runner_authorized",
        "artifact_writes_authorized",
        "evidence_authorized",
        "scientific_promotion_allowed",
    )
    for name in bool_fields:
        _exact_bool(getattr(audit, name), name=f"audit {name}")
    required_true = (
        audit.post_promotion_refresh_then_cascade_then_candidate_resolution,
        audit.structural_lifetime_descriptor_complete,
        audit.active_parent_snapshot_complete,
        audit.candidate_parent_snapshot_complete,
        audit.raw_source_identity_bound,
        audit.active_lineage_consistent,
        audit.theta_may_adapt_within_structural_lifetime,
        audit.runner_side_state_only,
        audit.public_core_event_trace_required_for_authentication,
        audit.public_core_event_trace_available,
    )
    _require(all(required_true), "audit integrity disclosures must remain enabled")
    forbidden = (
        audit.theta_bound_to_structural_lifetime_identity,
        audit.exact_functional_expression_identity_claimed,
        audit.transition_collision_observed,
        audit.dead_identity_history_retained,
        audit.historical_global_uniqueness_claimed,
        audit.cryptographic_collision_impossibility_claimed,
        audit.compositional_feature_state_fields_added,
        audit.caller_supplied_events_authenticated,
        audit.public_core_event_trace_consumed,
        audit.lifecycle_prerequisite_complete,
        audit.execution_authorized,
        audit.runner_authorized,
        audit.artifact_writes_authorized,
        audit.evidence_authorized,
        audit.scientific_promotion_allowed,
    )
    _require(not any(forbidden), "audit cannot grant authority or historical claims")
    _require(
        type(audit.collision_resistance_primitive) is str
        and audit.collision_resistance_primitive == "SHA-256",
        "audit collision primitive is stale",
    )
    _require(
        audit.expected_state_persistent_array_nbytes_formula == _STATE_NBYTES_FORMULA,
        "audit state byte formula is stale",
    )
    _require(
        audit.expected_assignment_persistent_array_nbytes_formula == _ASSIGNMENT_NBYTES_FORMULA,
        "audit assignment byte formula is stale",
    )
    _require(
        audit.expected_event_fixed_array_nbytes_formula == _EVENT_NBYTES_FORMULA,
        "audit event byte formula is stale",
    )


def derive_generated_birth_identity(
    *,
    namespace: str,
    paired_development_life_seed: int,
    learner_step: int,
    event_channel: str,
    slot: int,
    ordinal: int,
) -> bytes:
    """Derive one target- and outcome-blind, domain-separated SHA-256 ID."""

    _require(
        type(namespace) is str and _NAMESPACE_PATTERN.fullmatch(namespace) is not None,
        "namespace must be a canonical ASCII identifier",
    )
    _exact_int(
        paired_development_life_seed,
        name="paired_development_life_seed",
        lower=0,
        upper=_INT64_MAX,
    )
    _exact_int(learner_step, name="learner_step", lower=0, upper=_INT32_MAX)
    _require(
        type(event_channel) is str and event_channel in _NEW_ID_CHANNELS,
        "event_channel must name a canonical new-identity channel",
    )
    _exact_int(slot, name="slot", lower=0, upper=_INT32_MAX)
    _exact_int(ordinal, name="ordinal", lower=0, upper=_INT32_MAX)
    payload = {
        "domain": "alberta-generated-structural-lifetime-identity-sha256-v3",
        "schema": GENERATED_BIRTH_IDENTITY_LEDGER_SCHEMA,
        "namespace": namespace,
        "paired_development_life_seed": paired_development_life_seed,
        "learner_step": learner_step,
        "event_channel": event_channel,
        "slot": slot,
        "ordinal": ordinal,
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).digest()


def derive_generated_birth_identity_v4(
    *,
    namespace: str,
    paired_development_life_seed: int,
    learner_step_words: UInt32Array,
    event_channel: str,
    slot: int,
    ordinal: int,
) -> bytes:
    """Derive one schema-v4 identity from an exact big-endian uint32 pair."""

    _require(
        type(namespace) is str and _NAMESPACE_PATTERN.fullmatch(namespace) is not None,
        "namespace must be a canonical ASCII identifier",
    )
    _exact_int(
        paired_development_life_seed,
        name="paired_development_life_seed",
        lower=0,
        upper=_INT64_MAX,
    )
    words = _exact_array(
        learner_step_words,
        dtype=np.dtype(np.uint32),
        shape=(2,),
        name="learner_step_words",
    )
    _require(
        type(event_channel) is str and event_channel in _NEW_ID_CHANNELS,
        "event_channel must name a canonical new-identity channel",
    )
    _exact_int(slot, name="slot", lower=0, upper=_INT32_MAX)
    _exact_int(ordinal, name="ordinal", lower=0, upper=_INT32_MAX)
    payload = {
        "domain": "alberta-generated-structural-lifetime-identity-sha256-v4",
        "schema": GENERATED_BIRTH_IDENTITY_LEDGER_V4_SCHEMA,
        "namespace": namespace,
        "paired_development_life_seed": paired_development_life_seed,
        "learner_step_words_big_endian_uint32": [int(words[0]), int(words[1])],
        "event_channel": event_channel,
        "slot": slot,
        "ordinal": ordinal,
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).digest()


def _identity_array(
    *,
    namespace: str,
    seed: int,
    step: int,
    channel: str,
    slot: int,
    ordinal: int,
    step_words: UInt32Array | None = None,
) -> UInt8Array:
    if step_words is not None:
        return np.frombuffer(
            derive_generated_birth_identity_v4(
                namespace=namespace,
                paired_development_life_seed=seed,
                learner_step_words=step_words,
                event_channel=channel,
                slot=slot,
                ordinal=ordinal,
            ),
            dtype=np.uint8,
        ).copy()
    return np.frombuffer(
        derive_generated_birth_identity(
            namespace=namespace,
            paired_development_life_seed=seed,
            learner_step=step,
            event_channel=channel,
            slot=slot,
            ordinal=ordinal,
        ),
        dtype=np.uint8,
    ).copy()


def _singleton_mask(slot: int, size: int) -> BoolArray:
    mask = np.zeros((size,), dtype=np.bool_)
    if slot >= 0:
        mask[slot] = True
    return mask


def _raw_source_identity(
    config: GeneratedBirthIdentityLedgerConfig,
    *,
    seed: int,
    raw_source_index: int,
) -> UInt8Array:
    """Return the stable per-life identity of one raw observation coordinate."""

    return _identity_array(
        namespace=config.namespace,
        seed=seed,
        step=0,
        channel=RAW_SOURCE_IDENTITY_CHANNEL,
        slot=raw_source_index,
        ordinal=0,
    )


def _validate_structural_arrays(
    *,
    config: GeneratedBirthIdentityLedgerConfig,
    parent_a: Int32Array,
    parent_b: Int32Array,
    ops: Int32Array,
    depth: Int32Array,
    generator_policy: Int32Array,
    active_bank: bool,
    active_depth: Int32Array | None = None,
) -> None:
    size = config.active_slots if active_bank else config.candidate_slots
    _require(parent_a.shape == (size,), "parent_a shape is invalid")
    _require(parent_b.shape == (size,), "parent_b shape is invalid")
    _require(ops.shape == (size,), "ops shape is invalid")
    _require(depth.shape == (size,), "depth shape is invalid")
    _require(generator_policy.shape == (size,), "generator_policy shape is invalid")
    if not active_bank:
        _require(active_depth is not None, "candidate depth validation needs active depth")
    for slot in range(size):
        op = int(ops[slot])
        _require(0 <= op < NUM_OPS, "op is outside the public compositional operation set")
        _require(
            0 <= int(generator_policy[slot]) < config.generator_policy_count,
            "generator provenance is outside the bound public policy manifest",
        )
        _require(
            0 <= int(depth[slot]) <= config.max_depth,
            "depth is outside the bound compositional depth budget",
        )
        if op != OP_RAW:
            upper = slot if active_bank else config.active_slots
            _require(upper > 0, "composed slot has no legal parent prefix")
            _require(
                0 <= int(parent_a[slot]) < upper,
                "composed parent_a violates the active topological range",
            )
            _require(
                0 <= int(parent_b[slot]) < upper,
                "composed parent_b violates the active topological range",
            )
            depth_bank = depth if active_bank else active_depth
            assert depth_bank is not None
            expected_depth = (
                max(
                    int(depth_bank[int(parent_a[slot])]),
                    int(depth_bank[int(parent_b[slot])]),
                )
                + 1
            )
            _require(
                int(depth[slot]) == expected_depth,
                "depth is not exactly derived from the ordered parent slots",
            )
        else:
            _require(
                0 <= int(parent_a[slot]) < config.raw_feature_slots,
                "raw parent_a must name a raw observation source",
            )
            _require(int(parent_b[slot]) == -1, "raw parent_b must be -1")
            _require(int(depth[slot]) == 0, "raw depth must be exactly zero")
            if active_bank and slot < config.raw_feature_slots:
                _require(
                    int(parent_a[slot]) == slot,
                    "raw-prefix active slot must bind its own raw source index",
                )


def _validate_sampled_policy_codes(
    config: GeneratedBirthIdentityLedgerConfig,
    *,
    generator_policy: Int32Array,
    sampled: BoolArray,
    name: str,
) -> None:
    _require(generator_policy.shape == sampled.shape, f"{name} sampled mask shape is invalid")
    _require(
        bool(np.all(generator_policy[~sampled] == np.int32(FIXED_GENERATOR_POLICY_PLACEHOLDER))),
        f"{name} unsampled codes must be the fixed policy placeholder",
    )
    _require(
        bool(
            np.all(
                (generator_policy[sampled] >= 0)
                & (generator_policy[sampled] < config.generator_policy_count)
            )
        ),
        f"{name} sampled codes are outside the bound public policy manifest",
    )


def _parent_identity_snapshot(
    config: GeneratedBirthIdentityLedgerConfig,
    *,
    seed: int,
    active_identity: UInt8Array,
    parent_a: Int32Array,
    parent_b: Int32Array,
    ops: Int32Array,
) -> UInt8Array:
    result = np.zeros((parent_a.shape[0], 2, _IDENTITY_BYTES), dtype=np.uint8)
    for slot in range(parent_a.shape[0]):
        if int(ops[slot]) != OP_RAW:
            result[slot, 0] = active_identity[int(parent_a[slot])]
            result[slot, 1] = active_identity[int(parent_b[slot])]
        else:
            result[slot, 0] = _raw_source_identity(
                config,
                seed=seed,
                raw_source_index=int(parent_a[slot]),
            )
    return result


def _identity_rows_unique(*banks: UInt8Array) -> bool:
    rows = np.concatenate(banks, axis=0)
    identities = [row.tobytes() for row in rows]
    return len(identities) == len(set(identities)) and all(
        identity != bytes(_IDENTITY_BYTES) for identity in identities
    )


def _raw_source_identity_bank(
    config: GeneratedBirthIdentityLedgerConfig,
    *,
    seed: int,
) -> UInt8Array:
    return np.stack(
        [
            _raw_source_identity(config, seed=seed, raw_source_index=slot)
            for slot in range(config.raw_feature_slots)
        ],
        axis=0,
    )


def _validate_state(
    config: GeneratedBirthIdentityLedgerConfig,
    state: GeneratedBirthIdentityLedgerState,
) -> None:
    _require(type(state) is GeneratedBirthIdentityLedgerState, "ledger state type is invalid")
    _require(type(state.schema) is str, "ledger state schema must be an exact string")
    _require(type(state.status) is str, "ledger state status must be an exact string")
    _require(type(state.namespace) is str, "ledger state namespace must be an exact string")
    _require(state.schema == config.schema, "ledger state schema is stale")
    _require(state.status == config.status, "ledger state status is stale")
    _require(state.config_sha256 == _config_sha256(config), "ledger state config hash is stale")
    _require(state.namespace == config.namespace, "ledger state namespace is stale")
    _exact_int(
        state.paired_development_life_seed,
        name="paired development life seed",
        lower=0,
        upper=_INT64_MAX,
    )
    _exact_int(state.learner_step, name="learner_step", lower=0, upper=_INT32_MAX)
    arrays = (
        (
            state.active_identity,
            np.dtype(np.uint8),
            (config.active_slots, _IDENTITY_BYTES),
            "active_identity",
        ),
        (state.active_parent_a, np.dtype(np.int32), (config.active_slots,), "active_parent_a"),
        (state.active_parent_b, np.dtype(np.int32), (config.active_slots,), "active_parent_b"),
        (
            state.active_parent_identity_snapshot,
            np.dtype(np.uint8),
            (config.active_slots, 2, _IDENTITY_BYTES),
            "active_parent_identity_snapshot",
        ),
        (state.active_ops, np.dtype(np.int32), (config.active_slots,), "active_ops"),
        (state.active_depth, np.dtype(np.int32), (config.active_slots,), "active_depth"),
        (
            state.active_generator_policy,
            np.dtype(np.int32),
            (config.active_slots,),
            "active_generator_policy",
        ),
        (
            state.active_generator_policy_sampled,
            np.dtype(np.bool_),
            (config.active_slots,),
            "active_generator_policy_sampled",
        ),
        (
            state.candidate_identity,
            np.dtype(np.uint8),
            (config.candidate_slots, _IDENTITY_BYTES),
            "candidate_identity",
        ),
        (
            state.candidate_parent_a,
            np.dtype(np.int32),
            (config.candidate_slots,),
            "candidate_parent_a",
        ),
        (
            state.candidate_parent_b,
            np.dtype(np.int32),
            (config.candidate_slots,),
            "candidate_parent_b",
        ),
        (
            state.candidate_parent_identity_snapshot,
            np.dtype(np.uint8),
            (config.candidate_slots, 2, _IDENTITY_BYTES),
            "candidate_parent_identity_snapshot",
        ),
        (state.candidate_ops, np.dtype(np.int32), (config.candidate_slots,), "candidate_ops"),
        (
            state.candidate_depth,
            np.dtype(np.int32),
            (config.candidate_slots,),
            "candidate_depth",
        ),
        (
            state.candidate_generator_policy,
            np.dtype(np.int32),
            (config.candidate_slots,),
            "candidate_generator_policy",
        ),
        (
            state.candidate_generator_policy_sampled,
            np.dtype(np.bool_),
            (config.candidate_slots,),
            "candidate_generator_policy_sampled",
        ),
    )
    actual_nbytes = 0
    for array, dtype, shape, name in arrays:
        _exact_array(
            array,
            dtype=dtype,
            shape=shape,
            name=name,
            require_immutable=True,
        )
        actual_nbytes += array.nbytes
    expected_nbytes = 117 * (config.active_slots + config.candidate_slots)
    _validate_structural_arrays(
        config=config,
        parent_a=state.active_parent_a,
        parent_b=state.active_parent_b,
        ops=state.active_ops,
        depth=state.active_depth,
        generator_policy=state.active_generator_policy,
        active_bank=True,
    )
    _validate_structural_arrays(
        config=config,
        parent_a=state.candidate_parent_a,
        parent_b=state.candidate_parent_b,
        ops=state.candidate_ops,
        depth=state.candidate_depth,
        generator_policy=state.candidate_generator_policy,
        active_bank=False,
        active_depth=state.active_depth,
    )
    _validate_sampled_policy_codes(
        config,
        generator_policy=state.active_generator_policy,
        sampled=state.active_generator_policy_sampled,
        name="active generator policy",
    )
    _validate_sampled_policy_codes(
        config,
        generator_policy=state.candidate_generator_policy,
        sampled=state.candidate_generator_policy_sampled,
        name="candidate generator policy",
    )
    expected_active_snapshot = _parent_identity_snapshot(
        config,
        seed=state.paired_development_life_seed,
        active_identity=state.active_identity,
        parent_a=state.active_parent_a,
        parent_b=state.active_parent_b,
        ops=state.active_ops,
    )
    expected_candidate_snapshot = _parent_identity_snapshot(
        config,
        seed=state.paired_development_life_seed,
        active_identity=state.active_identity,
        parent_a=state.candidate_parent_a,
        parent_b=state.candidate_parent_b,
        ops=state.candidate_ops,
    )
    _require(
        np.array_equal(state.active_parent_identity_snapshot, expected_active_snapshot),
        "active parent identity snapshot is stale",
    )
    _require(
        np.array_equal(
            state.candidate_parent_identity_snapshot,
            expected_candidate_snapshot,
        ),
        "candidate parent identity snapshot is stale",
    )
    _require(actual_nbytes == expected_nbytes, "ledger state array byte count is invalid")
    _exact_int(
        state.persistent_array_nbytes,
        name="ledger state persistent_array_nbytes",
        lower=0,
        upper=_INT64_MAX,
    )
    _require(
        state.persistent_array_nbytes == expected_nbytes,
        "ledger state persistent_array_nbytes is stale",
    )
    _require(
        state.expected_persistent_array_nbytes_formula == _STATE_NBYTES_FORMULA,
        "ledger state byte formula is stale",
    )
    _require(_valid_sha256(state.integrity_sha256), "ledger state integrity hash is malformed")
    _require(
        state.integrity_sha256 == generated_birth_identity_ledger_state_sha256(state),
        "ledger state integrity hash does not reconstruct",
    )
    _require(
        _identity_rows_unique(
            state.active_identity,
            state.candidate_identity,
            _raw_source_identity_bank(
                config,
                seed=state.paired_development_life_seed,
            ),
        ),
        "duplicate or zero live/raw identity observed",
    )


def _make_state(
    config: GeneratedBirthIdentityLedgerConfig,
    *,
    seed: int,
    step: int,
    active_identity: UInt8Array,
    active_parent_a: Int32Array,
    active_parent_b: Int32Array,
    active_ops: Int32Array,
    active_depth: Int32Array,
    active_generator_policy: Int32Array,
    active_generator_policy_sampled: BoolArray,
    candidate_identity: UInt8Array,
    candidate_parent_a: Int32Array,
    candidate_parent_b: Int32Array,
    candidate_ops: Int32Array,
    candidate_depth: Int32Array,
    candidate_generator_policy: Int32Array,
    candidate_generator_policy_sampled: BoolArray,
) -> GeneratedBirthIdentityLedgerState:
    active_snapshot = _parent_identity_snapshot(
        config,
        seed=seed,
        active_identity=active_identity,
        parent_a=active_parent_a,
        parent_b=active_parent_b,
        ops=active_ops,
    )
    candidate_snapshot = _parent_identity_snapshot(
        config,
        seed=seed,
        active_identity=active_identity,
        parent_a=candidate_parent_a,
        parent_b=candidate_parent_b,
        ops=candidate_ops,
    )
    expected_nbytes = 117 * (config.active_slots + config.candidate_slots)
    state = GeneratedBirthIdentityLedgerState(
        schema=config.schema,
        status=config.status,
        config_sha256=_config_sha256(config),
        namespace=config.namespace,
        paired_development_life_seed=seed,
        learner_step=step,
        active_identity=_readonly_copy(active_identity),
        active_parent_a=_readonly_copy(active_parent_a),
        active_parent_b=_readonly_copy(active_parent_b),
        active_parent_identity_snapshot=_readonly_copy(active_snapshot),
        active_ops=_readonly_copy(active_ops),
        active_depth=_readonly_copy(active_depth),
        active_generator_policy=_readonly_copy(active_generator_policy),
        active_generator_policy_sampled=_readonly_copy(active_generator_policy_sampled),
        candidate_identity=_readonly_copy(candidate_identity),
        candidate_parent_a=_readonly_copy(candidate_parent_a),
        candidate_parent_b=_readonly_copy(candidate_parent_b),
        candidate_parent_identity_snapshot=_readonly_copy(candidate_snapshot),
        candidate_ops=_readonly_copy(candidate_ops),
        candidate_depth=_readonly_copy(candidate_depth),
        candidate_generator_policy=_readonly_copy(candidate_generator_policy),
        candidate_generator_policy_sampled=_readonly_copy(candidate_generator_policy_sampled),
        persistent_array_nbytes=expected_nbytes,
        expected_persistent_array_nbytes_formula=_STATE_NBYTES_FORMULA,
        integrity_sha256="0" * 64,
    )
    state = dataclasses.replace(
        state,
        integrity_sha256=generated_birth_identity_ledger_state_sha256(state),
    )
    _validate_state(config, state)
    return state


def initialize_generated_birth_identity_ledger(
    config: GeneratedBirthIdentityLedgerConfig,
    *,
    paired_development_life_seed: int,
    learner_step: int,
    active_parent_a: Int32Array,
    active_parent_b: Int32Array,
    active_ops: Int32Array,
    active_depth: Int32Array,
    active_generator_policy: Int32Array,
    candidate_parent_a: Int32Array,
    candidate_parent_b: Int32Array,
    candidate_ops: Int32Array,
    candidate_depth: Int32Array,
    candidate_generator_policy: Int32Array,
) -> GeneratedBirthIdentityLedgerState:
    """Create one deterministic, target-blind sidecar genesis."""

    _exact_int(
        paired_development_life_seed,
        name="paired_development_life_seed",
        lower=0,
        upper=_INT64_MAX,
    )
    _exact_int(learner_step, name="learner_step", lower=0, upper=_INT32_MAX)
    active_pa = _exact_array(
        active_parent_a,
        dtype=np.dtype(np.int32),
        shape=(config.active_slots,),
        name="active_parent_a",
    )
    active_pb = _exact_array(
        active_parent_b,
        dtype=np.dtype(np.int32),
        shape=(config.active_slots,),
        name="active_parent_b",
    )
    active_ops_array = _exact_array(
        active_ops,
        dtype=np.dtype(np.int32),
        shape=(config.active_slots,),
        name="active_ops",
    )
    active_depth_array = _exact_array(
        active_depth,
        dtype=np.dtype(np.int32),
        shape=(config.active_slots,),
        name="active_depth",
    )
    active_provenance = _exact_array(
        active_generator_policy,
        dtype=np.dtype(np.int32),
        shape=(config.active_slots,),
        name="active_generator_policy",
    )
    candidate_pa = _exact_array(
        candidate_parent_a,
        dtype=np.dtype(np.int32),
        shape=(config.candidate_slots,),
        name="candidate_parent_a",
    )
    candidate_pb = _exact_array(
        candidate_parent_b,
        dtype=np.dtype(np.int32),
        shape=(config.candidate_slots,),
        name="candidate_parent_b",
    )
    candidate_ops_array = _exact_array(
        candidate_ops,
        dtype=np.dtype(np.int32),
        shape=(config.candidate_slots,),
        name="candidate_ops",
    )
    candidate_depth_array = _exact_array(
        candidate_depth,
        dtype=np.dtype(np.int32),
        shape=(config.candidate_slots,),
        name="candidate_depth",
    )
    candidate_provenance = _exact_array(
        candidate_generator_policy,
        dtype=np.dtype(np.int32),
        shape=(config.candidate_slots,),
        name="candidate_generator_policy",
    )
    _validate_structural_arrays(
        config=config,
        parent_a=active_pa,
        parent_b=active_pb,
        ops=active_ops_array,
        depth=active_depth_array,
        generator_policy=active_provenance,
        active_bank=True,
    )
    _validate_structural_arrays(
        config=config,
        parent_a=candidate_pa,
        parent_b=candidate_pb,
        ops=candidate_ops_array,
        depth=candidate_depth_array,
        generator_policy=candidate_provenance,
        active_bank=False,
        active_depth=active_depth_array,
    )
    _require(
        bool(np.all(active_provenance == np.int32(FIXED_GENERATOR_POLICY_PLACEHOLDER))),
        "genesis active generator policies must be fixed placeholders",
    )
    _require(
        bool(np.all(candidate_provenance == np.int32(FIXED_GENERATOR_POLICY_PLACEHOLDER))),
        "genesis candidate generator policies must be fixed placeholders",
    )
    _require(
        np.array_equal(
            active_ops_array[: config.raw_feature_slots],
            np.full((config.raw_feature_slots,), OP_RAW, dtype=np.int32),
        ),
        "raw-prefix active slots must remain raw",
    )
    active = np.stack(
        [
            _identity_array(
                namespace=config.namespace,
                seed=paired_development_life_seed,
                step=learner_step,
                channel=INITIAL_ACTIVE_CHANNEL,
                slot=slot,
                ordinal=slot,
            )
            for slot in range(config.active_slots)
        ],
        axis=0,
    )
    candidate = (
        np.stack(
            [
                _identity_array(
                    namespace=config.namespace,
                    seed=paired_development_life_seed,
                    step=learner_step,
                    channel=INITIAL_CANDIDATE_CHANNEL,
                    slot=slot,
                    ordinal=slot,
                )
                for slot in range(config.candidate_slots)
            ],
            axis=0,
        )
        if config.candidate_slots
        else np.zeros((0, _IDENTITY_BYTES), dtype=np.uint8)
    )
    return _make_state(
        config,
        seed=paired_development_life_seed,
        step=learner_step,
        active_identity=active,
        active_parent_a=active_pa,
        active_parent_b=active_pb,
        active_ops=active_ops_array,
        active_depth=active_depth_array,
        active_generator_policy=active_provenance,
        active_generator_policy_sampled=np.zeros((config.active_slots,), dtype=np.bool_),
        candidate_identity=candidate,
        candidate_parent_a=candidate_pa,
        candidate_parent_b=candidate_pb,
        candidate_ops=candidate_ops_array,
        candidate_depth=candidate_depth_array,
        candidate_generator_policy=candidate_provenance,
        candidate_generator_policy_sampled=np.zeros((config.candidate_slots,), dtype=np.bool_),
    )


def _validate_slot(slot: int, *, size: int, name: str) -> None:
    _require(type(slot) is int, f"{name} must be an exact Python integer")
    _require(slot == -1 or 0 <= slot < size, f"{name} is outside the fixed bank")


def _require_transition_available(pre: GeneratedBirthIdentityLedgerState) -> None:
    _require(
        pre.learner_step < _INT32_MAX,
        "terminal learner_step is exhausted at the core int32 maximum",
    )


def _validate_event_arrays(
    config: GeneratedBirthIdentityLedgerConfig,
    event: GeneratedBirthIdentityEvent,
) -> None:
    specifications = (
        ("promotion_active_mask", np.bool_, (config.active_slots,)),
        ("promotion_candidate_mask", np.bool_, (config.candidate_slots,)),
        ("direct_active_replacement_mask", np.bool_, (config.active_slots,)),
        ("cascade_refill_mask", np.bool_, (config.active_slots,)),
        ("ordinary_candidate_refresh_mask", np.bool_, (config.candidate_slots,)),
        (
            "post_promotion_candidate_refresh_mask",
            np.bool_,
            (config.candidate_slots,),
        ),
        ("candidate_rebound_mask", np.bool_, (config.candidate_slots,)),
        (
            "candidate_overdepth_regeneration_mask",
            np.bool_,
            (config.candidate_slots,),
        ),
        ("active_parent_a", np.int32, (config.active_slots,)),
        ("active_parent_b", np.int32, (config.active_slots,)),
        ("active_ops", np.int32, (config.active_slots,)),
        ("active_depth", np.int32, (config.active_slots,)),
        ("active_generator_policy", np.int32, (config.active_slots,)),
        ("candidate_staged_parent_a", np.int32, (config.candidate_slots,)),
        ("candidate_staged_parent_b", np.int32, (config.candidate_slots,)),
        ("candidate_staged_ops", np.int32, (config.candidate_slots,)),
        ("candidate_staged_depth", np.int32, (config.candidate_slots,)),
        (
            "candidate_staged_generator_policy",
            np.int32,
            (config.candidate_slots,),
        ),
        ("candidate_parent_a", np.int32, (config.candidate_slots,)),
        ("candidate_parent_b", np.int32, (config.candidate_slots,)),
        ("candidate_ops", np.int32, (config.candidate_slots,)),
        ("candidate_depth", np.int32, (config.candidate_slots,)),
        (
            "candidate_generator_policy",
            np.int32,
            (config.candidate_slots,),
        ),
    )
    for name, dtype, shape in specifications:
        _exact_array(
            getattr(event, name),
            dtype=np.dtype(dtype),
            shape=shape,
            name=name,
            require_immutable=True,
        )


def _exact_pre_graph_descendant_closure(
    pre: GeneratedBirthIdentityLedgerState,
    root_slot: int,
) -> BoolArray:
    """Return production's topological descendant closure from the pre graph."""

    closure = np.zeros((pre.active_identity.shape[0],), dtype=np.bool_)
    if root_slot < 0:
        return closure
    changed = np.array(closure, copy=True)
    changed[root_slot] = True
    for slot in range(root_slot + 1, pre.active_identity.shape[0]):
        if int(pre.active_ops[slot]) == OP_RAW:
            continue
        if bool(changed[int(pre.active_parent_a[slot])]) or bool(
            changed[int(pre.active_parent_b[slot])]
        ):
            closure[slot] = True
            changed[slot] = True
    return closure


def _validate_event_structure(
    config: GeneratedBirthIdentityLedgerConfig,
    pre: GeneratedBirthIdentityLedgerState,
    event: GeneratedBirthIdentityEvent,
    *,
    check_integrity: bool,
) -> None:
    _require(type(event) is GeneratedBirthIdentityEvent, "event type is invalid")
    _require(type(event.schema) is str, "event schema must be an exact string")
    _require(type(event.status) is str, "event status must be an exact string")
    _require(type(event.namespace) is str, "event namespace must be an exact string")
    _require(event.schema == config.schema, "event schema is stale")
    _require(event.status == config.status, "event status is stale")
    _require(event.namespace == config.namespace, "event namespace is stale")
    _exact_int(
        event.paired_development_life_seed,
        name="event paired development life seed",
        lower=0,
        upper=_INT64_MAX,
    )
    _require(
        event.paired_development_life_seed == pre.paired_development_life_seed,
        "event paired development life seed is stale",
    )
    _require_transition_available(pre)
    _exact_int(event.learner_step, name="event learner_step", lower=0, upper=_INT32_MAX)
    _require(event.learner_step == pre.learner_step + 1, "event learner_step is stale")
    _require(
        event.channel_manifest == GENERATED_BIRTH_IDENTITY_CHANNELS,
        "event channel manifest is stale",
    )
    _exact_bool(event.generator_policy_sampled, name="event generator_policy_sampled")
    _exact_int(
        event.generator_policy_id,
        name="event generator_policy_id",
        lower=0,
        upper=config.generator_policy_count - 1,
    )
    _require(
        event.generator_policy_sampled == config.learn_generator_resources,
        "event sampled-policy flag does not match the bound learner configuration",
    )
    if not event.generator_policy_sampled:
        _require(
            event.generator_policy_id == FIXED_GENERATOR_POLICY_PLACEHOLDER,
            "unsampled event policy id must be the fixed policy placeholder",
        )
    _validate_event_arrays(config, event)
    _validate_slot(
        event.promotion_active_slot,
        size=config.active_slots,
        name="promotion_active_slot",
    )
    _validate_slot(
        event.promotion_candidate_slot,
        size=config.candidate_slots,
        name="promotion_candidate_slot",
    )
    _validate_slot(
        event.direct_active_replacement_slot,
        size=config.active_slots,
        name="direct_active_replacement_slot",
    )
    _validate_slot(
        event.ordinary_candidate_refresh_slot,
        size=config.candidate_slots,
        name="ordinary_candidate_refresh_slot",
    )
    _validate_slot(
        event.post_promotion_candidate_refresh_slot,
        size=config.candidate_slots,
        name="post_promotion_candidate_refresh_slot",
    )
    _require(
        np.array_equal(
            event.promotion_active_mask,
            _singleton_mask(event.promotion_active_slot, config.active_slots),
        ),
        "promotion active mask does not match its index",
    )
    _require(
        np.array_equal(
            event.promotion_candidate_mask,
            _singleton_mask(event.promotion_candidate_slot, config.candidate_slots),
        ),
        "promotion candidate mask does not match its index",
    )
    _require(
        np.array_equal(
            event.direct_active_replacement_mask,
            _singleton_mask(event.direct_active_replacement_slot, config.active_slots),
        ),
        "direct active replacement mask does not match its index",
    )
    _require(
        np.array_equal(
            event.ordinary_candidate_refresh_mask,
            _singleton_mask(event.ordinary_candidate_refresh_slot, config.candidate_slots),
        ),
        "ordinary candidate refresh mask does not match its index",
    )
    _require(
        np.array_equal(
            event.post_promotion_candidate_refresh_mask,
            _singleton_mask(
                event.post_promotion_candidate_refresh_slot,
                config.candidate_slots,
            ),
        ),
        "post-promotion candidate refresh mask does not match its index",
    )
    promotion = event.promotion_active_slot >= 0
    _require(
        promotion == (event.promotion_candidate_slot >= 0),
        "promotion active and candidate indices must be paired",
    )
    _require(
        promotion == (event.post_promotion_candidate_refresh_slot >= 0),
        "promotion must have exactly one source-slot post-promotion refresh",
    )
    if promotion:
        _require(
            event.post_promotion_candidate_refresh_slot == event.promotion_candidate_slot,
            "post-promotion refresh index must equal the promotion source",
        )
        _require(
            event.ordinary_candidate_refresh_slot == -1,
            "ordinary and post-promotion refresh indices are incompatible",
        )
    _require(
        not (promotion and event.direct_active_replacement_slot >= 0),
        "promotion and direct active replacement are incompatible",
    )
    _require(
        event.direct_active_replacement_slot < 0 or config.candidate_slots == 0,
        "direct active replacement requires candidate_slots == 0",
    )
    _require(
        not (
            event.direct_active_replacement_slot >= 0
            and (
                event.ordinary_candidate_refresh_slot >= 0
                or event.post_promotion_candidate_refresh_slot >= 0
            )
        ),
        "direct active replacement and candidate refresh are incompatible",
    )
    root_slot = event.promotion_active_slot if promotion else event.direct_active_replacement_slot
    for slot_name in (
        "promotion_active_slot",
        "direct_active_replacement_slot",
    ):
        slot = getattr(event, slot_name)
        _require(
            slot < 0 or slot >= config.raw_feature_slots,
            f"{slot_name} cannot write the raw-prefix",
        )
    if np.any(event.cascade_refill_mask):
        _require(root_slot >= 0, "cascade refill requires an applied active root")
    active_event_sum = (
        event.promotion_active_mask.astype(np.int8)
        + event.direct_active_replacement_mask.astype(np.int8)
        + event.cascade_refill_mask.astype(np.int8)
    )
    _require(
        bool(np.all(active_event_sum <= 1)),
        "active event masks must be disjoint",
    )
    exact_cascade = _exact_pre_graph_descendant_closure(pre, root_slot)
    _require(
        np.array_equal(event.cascade_refill_mask, exact_cascade),
        "cascade_refill_mask does not equal the exact pre-graph descendant closure",
    )
    _require(
        not np.any(event.cascade_refill_mask[: config.raw_feature_slots]),
        "cascade refill cannot write the raw-prefix",
    )
    for slot in np.flatnonzero(event.cascade_refill_mask):
        _require(
            int(slot) > root_slot,
            "cascade refill slot must be a strict active descendant index",
        )
    _require(
        not np.any(
            event.ordinary_candidate_refresh_mask & event.post_promotion_candidate_refresh_mask
        ),
        "ordinary and post-promotion candidate refresh masks must be disjoint",
    )
    _require(
        not np.any(event.candidate_rebound_mask & event.candidate_overdepth_regeneration_mask),
        "candidate rebound and overdepth-regeneration masks must be disjoint",
    )
    _require(
        np.array_equal(
            event.active_ops[: config.raw_feature_slots],
            np.full((config.raw_feature_slots,), OP_RAW, dtype=np.int32),
        ),
        "raw-prefix active slots must remain raw",
    )
    _validate_structural_arrays(
        config=config,
        parent_a=event.active_parent_a,
        parent_b=event.active_parent_b,
        ops=event.active_ops,
        depth=event.active_depth,
        generator_policy=event.active_generator_policy,
        active_bank=True,
    )
    _validate_structural_arrays(
        config=config,
        parent_a=event.candidate_parent_a,
        parent_b=event.candidate_parent_b,
        ops=event.candidate_ops,
        depth=event.candidate_depth,
        generator_policy=event.candidate_generator_policy,
        active_bank=False,
        active_depth=event.active_depth,
    )
    for slot in range(config.candidate_slots):
        staged_op = int(event.candidate_staged_ops[slot])
        _require(
            0 <= staged_op < NUM_OPS,
            "staged candidate op is outside the public compositional operation set",
        )
        _require(
            0 <= int(event.candidate_staged_generator_policy[slot]) < config.generator_policy_count,
            "staged candidate provenance is outside the bound public policy manifest",
        )
        _require(
            0 <= int(event.candidate_staged_depth[slot]) <= config.max_depth,
            "staged candidate depth is outside the bound compositional depth budget",
        )
        if staged_op == OP_RAW:
            _require(
                0 <= int(event.candidate_staged_parent_a[slot]) < config.raw_feature_slots,
                "staged raw candidate parent_a must name a raw source",
            )
            _require(
                int(event.candidate_staged_parent_b[slot]) == -1,
                "staged raw candidate parent_b must be -1",
            )
            _require(
                int(event.candidate_staged_depth[slot]) == 0,
                "staged raw candidate depth must be zero",
            )
        else:
            _require(
                0 <= int(event.candidate_staged_parent_a[slot]) < config.active_slots,
                "staged candidate parent_a is outside the active bank",
            )
            _require(
                0 <= int(event.candidate_staged_parent_b[slot]) < config.active_slots,
                "staged candidate parent_b is outside the active bank",
            )
    if check_integrity:
        _require(_valid_sha256(event.integrity_sha256), "event integrity hash is malformed")
        _require(
            event.integrity_sha256 == generated_birth_identity_event_sha256(event),
            "event integrity hash does not reconstruct",
        )


@dataclasses.dataclass(frozen=True, slots=True)
class _TransitionArrays:
    assignments: GeneratedBirthIdentityAssignments
    post_active: UInt8Array
    post_active_snapshot: UInt8Array
    post_active_generator_policy_sampled: BoolArray
    post_candidate: UInt8Array
    post_candidate_snapshot: UInt8Array
    post_candidate_generator_policy_sampled: BoolArray
    canonical_rebound_mask: BoolArray
    canonical_overdepth_regeneration_mask: BoolArray
    active_lineage_consistent: bool


def _assignment_bank(active_slots: int) -> UInt8Array:
    return np.zeros((active_slots, _IDENTITY_BYTES), dtype=np.uint8)


def _nonzero_identity_rows(bank: UInt8Array) -> UInt8Array:
    nonzero = np.any(bank != np.uint8(0), axis=1)
    return bank[nonzero]


def _validate_transition_collision_scope(
    config: GeneratedBirthIdentityLedgerConfig,
    pre: GeneratedBirthIdentityLedgerState,
    assignments: GeneratedBirthIdentityAssignments,
) -> None:
    base_banks = (
        pre.active_identity,
        pre.candidate_identity,
        _raw_source_identity_bank(
            config,
            seed=pre.paired_development_life_seed,
        ),
    )
    new_banks = tuple(
        _nonzero_identity_rows(bank)
        for bank in (
            assignments.direct_active_birth_identity,
            assignments.cascade_active_birth_identity,
            assignments.ordinary_candidate_birth_identity,
            assignments.post_promotion_candidate_birth_identity,
            assignments.candidate_rebound_identity,
            assignments.candidate_overdepth_regeneration_identity,
        )
    )
    _require(
        _identity_rows_unique(*base_banks, *new_banks),
        "identity collision observed in the pre-live/raw/new-transition scope",
    )


def _compute_transition_arrays(
    config: GeneratedBirthIdentityLedgerConfig,
    pre: GeneratedBirthIdentityLedgerState,
    event: GeneratedBirthIdentityEvent,
    *,
    identity_step_words: UInt32Array | None = None,
) -> _TransitionArrays:
    seed = pre.paired_development_life_seed
    step = event.learner_step
    active = np.array(pre.active_identity, copy=True)
    candidate = np.array(pre.candidate_identity, copy=True)
    active_sampled = np.array(pre.active_generator_policy_sampled, copy=True)
    candidate_staged_sampled = np.array(
        pre.candidate_generator_policy_sampled,
        copy=True,
    )

    # Candidate refresh is evaluated after the root write but before cascade.
    # These stage arrays retain old cascade-slot descriptors until that point.
    staged_active_depth = np.array(pre.active_depth, copy=True)

    promotion_assign = _assignment_bank(config.active_slots)
    direct_assign = _assignment_bank(config.active_slots)
    cascade_assign = _assignment_bank(config.active_slots)
    ordinary_assign = _assignment_bank(config.candidate_slots)
    post_promotion_assign = _assignment_bank(config.candidate_slots)
    rebound_assign = _assignment_bank(config.candidate_slots)
    overdepth_assign = _assignment_bank(config.candidate_slots)

    active_change_mask = (
        event.promotion_active_mask
        | event.direct_active_replacement_mask
        | event.cascade_refill_mask
    )
    unchanged_active = ~active_change_mask
    for supplied, prior, label in (
        (event.active_parent_a, pre.active_parent_a, "parent_a"),
        (event.active_parent_b, pre.active_parent_b, "parent_b"),
        (event.active_ops, pre.active_ops, "op"),
        (event.active_depth, pre.active_depth, "depth"),
        (
            event.active_generator_policy,
            pre.active_generator_policy,
            "generator provenance",
        ),
    ):
        _require(
            np.array_equal(supplied[unchanged_active], prior[unchanged_active]),
            f"unchanged active structural descriptor changed at {label}",
        )

    if event.promotion_active_slot >= 0:
        destination = event.promotion_active_slot
        source = event.promotion_candidate_slot
        active[destination] = candidate[source]
        promotion_assign[destination] = candidate[source]
        for supplied, prior, label in (
            (event.active_parent_a, pre.candidate_parent_a, "parent_a"),
            (event.active_parent_b, pre.candidate_parent_b, "parent_b"),
            (event.active_ops, pre.candidate_ops, "op"),
            (event.active_depth, pre.candidate_depth, "depth"),
            (
                event.active_generator_policy,
                pre.candidate_generator_policy,
                "generator provenance",
            ),
        ):
            _require(
                int(supplied[destination]) == int(prior[source]),
                f"promoted active structural descriptor did not transfer {label}",
            )
        staged_active_depth[destination] = event.active_depth[destination]
        active_sampled[destination] = pre.candidate_generator_policy_sampled[source]

    if event.direct_active_replacement_slot >= 0:
        slot = event.direct_active_replacement_slot
        identity = _identity_array(
            namespace=config.namespace,
            seed=seed,
            step=step,
            channel=DIRECT_ACTIVE_REPLACEMENT_CHANNEL,
            slot=slot,
            ordinal=0,
            step_words=identity_step_words,
        )
        active[slot] = identity
        direct_assign[slot] = identity
        staged_active_depth[slot] = event.active_depth[slot]
        _require(
            int(event.active_generator_policy[slot]) == event.generator_policy_id,
            "direct active replacement did not use the current generator policy",
        )
        active_sampled[slot] = event.generator_policy_sampled

    for raw_slot in np.flatnonzero(event.cascade_refill_mask):
        slot = int(raw_slot)
        _require(
            int(event.active_generator_policy[slot]) == event.generator_policy_id,
            "cascade refill did not use the current generator policy",
        )
        active_sampled[slot] = event.generator_policy_sampled

    _validate_sampled_policy_codes(
        config,
        generator_policy=event.active_generator_policy,
        sampled=active_sampled,
        name="post-transition active generator policy",
    )

    # Candidate refresh occurs after promotion transfer but before cascade in
    # the current learner.  Its proposal-time snapshot must use this stage.
    refresh_mask = (
        event.ordinary_candidate_refresh_mask | event.post_promotion_candidate_refresh_mask
    )
    unrefreshed = ~refresh_mask
    for supplied, prior, label in (
        (event.candidate_staged_parent_a, pre.candidate_parent_a, "parent_a"),
        (event.candidate_staged_parent_b, pre.candidate_parent_b, "parent_b"),
        (event.candidate_staged_ops, pre.candidate_ops, "op"),
        (event.candidate_staged_depth, pre.candidate_depth, "depth"),
        (
            event.candidate_staged_generator_policy,
            pre.candidate_generator_policy,
            "generator provenance",
        ),
    ):
        _require(
            np.array_equal(supplied[unrefreshed], prior[unrefreshed]),
            f"unrefreshed candidate local structural descriptor changed at {label}",
        )

    # A refreshed candidate is generated against the post-root/pre-cascade
    # active bank and uses the update's current sampled/placeholder policy.
    for raw_slot in np.flatnonzero(refresh_mask):
        slot = int(raw_slot)
        _require(
            int(event.candidate_staged_generator_policy[slot]) == event.generator_policy_id,
            "candidate refresh did not use the current generator policy",
        )
        candidate_staged_sampled[slot] = event.generator_policy_sampled
        if int(event.candidate_staged_ops[slot]) == OP_RAW:
            expected_staged_depth = 0
        else:
            expected_staged_depth = (
                max(
                    int(staged_active_depth[int(event.candidate_staged_parent_a[slot])]),
                    int(staged_active_depth[int(event.candidate_staged_parent_b[slot])]),
                )
                + 1
            )
        _require(
            int(event.candidate_staged_depth[slot]) == expected_staged_depth,
            "refreshed candidate staged depth is not derived from the post-root bank",
        )
    _validate_sampled_policy_codes(
        config,
        generator_policy=event.candidate_staged_generator_policy,
        sampled=candidate_staged_sampled,
        name="staged candidate generator policy",
    )
    if event.ordinary_candidate_refresh_slot >= 0:
        slot = event.ordinary_candidate_refresh_slot
        identity = _identity_array(
            namespace=config.namespace,
            seed=seed,
            step=step,
            channel=ORDINARY_CANDIDATE_REFRESH_CHANNEL,
            slot=slot,
            ordinal=0,
            step_words=identity_step_words,
        )
        candidate[slot] = identity
        ordinary_assign[slot] = identity
    if event.post_promotion_candidate_refresh_slot >= 0:
        slot = event.post_promotion_candidate_refresh_slot
        identity = _identity_array(
            namespace=config.namespace,
            seed=seed,
            step=step,
            channel=POST_PROMOTION_CANDIDATE_REFRESH_CHANNEL,
            slot=slot,
            ordinal=0,
            step_words=identity_step_words,
        )
        candidate[slot] = identity
        post_promotion_assign[slot] = identity

    # Cascade refills use ascending fixed-bank order.  Their ordinal is the
    # rank among applied cascade writes, not the slot index.
    for ordinal, raw_slot in enumerate(np.flatnonzero(event.cascade_refill_mask)):
        slot = int(raw_slot)
        identity = _identity_array(
            namespace=config.namespace,
            seed=seed,
            step=step,
            channel=CASCADE_ACTIVE_REFILL_CHANNEL,
            slot=slot,
            ordinal=ordinal,
            step_words=identity_step_words,
        )
        active[slot] = identity
        cascade_assign[slot] = identity

    final_active_snapshot = _parent_identity_snapshot(
        config,
        seed=seed,
        active_identity=active,
        parent_a=event.active_parent_a,
        parent_b=event.active_parent_b,
        ops=event.active_ops,
    )
    # Cascade generation only samples parents outside the cascade mask.
    for raw_slot in np.flatnonzero(event.cascade_refill_mask):
        slot = int(raw_slot)
        if int(event.active_ops[slot]) != OP_RAW:
            _require(
                not bool(event.cascade_refill_mask[int(event.active_parent_a[slot])])
                and not bool(event.cascade_refill_mask[int(event.active_parent_b[slot])]),
                "cascade refill parent cannot be another cascade-refilled slot",
            )
    if event.promotion_active_slot >= 0:
        destination = event.promotion_active_slot
        _require(
            np.array_equal(
                final_active_snapshot[destination],
                pre.candidate_parent_identity_snapshot[event.promotion_candidate_slot],
            ),
            "promotion active parent identities do not match the source candidate snapshot",
        )
    active_lineage_consistent = bool(
        np.array_equal(
            final_active_snapshot[unchanged_active],
            pre.active_parent_identity_snapshot[unchanged_active],
        )
    )
    _require(
        active_lineage_consistent,
        "unchanged active parent identity changed without a cascade refill",
    )
    active_change_mask = np.asarray(active_change_mask, dtype=np.bool_)
    structural_parent_change = np.zeros((config.candidate_slots,), dtype=np.bool_)
    rebound_depth = np.asarray(event.candidate_staged_depth, dtype=np.int64).copy()
    for slot in range(config.candidate_slots):
        if int(event.candidate_staged_ops[slot]) == OP_RAW:
            continue
        parent_a = int(event.candidate_staged_parent_a[slot])
        parent_b = int(event.candidate_staged_parent_b[slot])
        changed_bank = (
            event.cascade_refill_mask
            if bool(event.post_promotion_candidate_refresh_mask[slot])
            else active_change_mask
        )
        structural_parent_change[slot] = bool(changed_bank[parent_a] or changed_bank[parent_b])
        rebound_depth[slot] = (
            max(int(event.active_depth[parent_a]), int(event.active_depth[parent_b])) + 1
        )
    canonical_overdepth = structural_parent_change & (rebound_depth > config.max_depth)
    canonical_rebound = structural_parent_change & ~canonical_overdepth
    candidate_final_sampled = np.array(candidate_staged_sampled, copy=True)
    candidate_final_sampled[canonical_overdepth] = event.generator_policy_sampled
    for slot in range(config.candidate_slots):
        if bool(canonical_overdepth[slot]):
            _require(
                int(event.candidate_generator_policy[slot]) == event.generator_policy_id,
                "overdepth regeneration did not use the current generator policy",
            )
            continue
        for supplied, staged, label in (
            (event.candidate_parent_a, event.candidate_staged_parent_a, "parent_a"),
            (event.candidate_parent_b, event.candidate_staged_parent_b, "parent_b"),
            (event.candidate_ops, event.candidate_staged_ops, "op"),
            (
                event.candidate_generator_policy,
                event.candidate_staged_generator_policy,
                "generator provenance",
            ),
        ):
            changed_prefix = (
                "refreshed candidate changed staged"
                if bool(refresh_mask[slot])
                else "unrefreshed candidate local structural descriptor changed at"
            )
            _require(
                int(supplied[slot]) == int(staged[slot]),
                f"{changed_prefix} {label}",
            )
        expected_depth = (
            int(rebound_depth[slot])
            if bool(canonical_rebound[slot])
            else int(event.candidate_staged_depth[slot])
        )
        _require(
            int(event.candidate_depth[slot]) == expected_depth,
            "candidate depth does not match its structural parent-change class",
        )
    _validate_sampled_policy_codes(
        config,
        generator_policy=event.candidate_generator_policy,
        sampled=candidate_final_sampled,
        name="post-transition candidate generator policy",
    )

    for ordinal, raw_slot in enumerate(np.flatnonzero(canonical_rebound)):
        slot = int(raw_slot)
        identity = _identity_array(
            namespace=config.namespace,
            seed=seed,
            step=step,
            channel=CANDIDATE_PARENT_REBOUND_CHANNEL,
            slot=slot,
            ordinal=ordinal,
            step_words=identity_step_words,
        )
        candidate[slot] = identity
        rebound_assign[slot] = identity

    for ordinal, raw_slot in enumerate(np.flatnonzero(canonical_overdepth)):
        slot = int(raw_slot)
        identity = _identity_array(
            namespace=config.namespace,
            seed=seed,
            step=step,
            channel=CANDIDATE_OVERDEPTH_REGENERATION_CHANNEL,
            slot=slot,
            ordinal=ordinal,
            step_words=identity_step_words,
        )
        candidate[slot] = identity
        overdepth_assign[slot] = identity

    final_candidate_snapshot = _parent_identity_snapshot(
        config,
        seed=seed,
        active_identity=active,
        parent_a=event.candidate_parent_a,
        parent_b=event.candidate_parent_b,
        ops=event.candidate_ops,
    )

    expected_assignment_nbytes = _IDENTITY_BYTES * (
        3 * config.active_slots + 4 * config.candidate_slots
    )
    assignments = GeneratedBirthIdentityAssignments(
        promotion_transfer_active_identity=_readonly_copy(promotion_assign),
        direct_active_birth_identity=_readonly_copy(direct_assign),
        cascade_active_birth_identity=_readonly_copy(cascade_assign),
        ordinary_candidate_birth_identity=_readonly_copy(ordinary_assign),
        post_promotion_candidate_birth_identity=_readonly_copy(post_promotion_assign),
        candidate_rebound_identity=_readonly_copy(rebound_assign),
        candidate_overdepth_regeneration_identity=_readonly_copy(overdepth_assign),
        persistent_array_nbytes=expected_assignment_nbytes,
        expected_persistent_array_nbytes_formula=_ASSIGNMENT_NBYTES_FORMULA,
        integrity_sha256="0" * 64,
    )
    assignments = dataclasses.replace(
        assignments,
        integrity_sha256=_assignments_sha256(assignments),
    )
    _validate_assignments(config, assignments)
    _validate_transition_collision_scope(config, pre, assignments)
    return _TransitionArrays(
        assignments=assignments,
        post_active=active,
        post_active_snapshot=final_active_snapshot,
        post_active_generator_policy_sampled=active_sampled,
        post_candidate=candidate,
        post_candidate_snapshot=final_candidate_snapshot,
        post_candidate_generator_policy_sampled=candidate_final_sampled,
        canonical_rebound_mask=canonical_rebound,
        canonical_overdepth_regeneration_mask=canonical_overdepth,
        active_lineage_consistent=active_lineage_consistent,
    )


def build_generated_birth_identity_event(
    config: GeneratedBirthIdentityLedgerConfig,
    pre_state: GeneratedBirthIdentityLedgerState,
    *,
    learner_step: int,
    generator_policy_sampled: bool,
    generator_policy_id: int,
    active_parent_a: Int32Array,
    active_parent_b: Int32Array,
    active_ops: Int32Array,
    active_depth: Int32Array,
    active_generator_policy: Int32Array,
    candidate_staged_parent_a: Int32Array,
    candidate_staged_parent_b: Int32Array,
    candidate_staged_ops: Int32Array,
    candidate_staged_depth: Int32Array,
    candidate_staged_generator_policy: Int32Array,
    candidate_parent_a: Int32Array,
    candidate_parent_b: Int32Array,
    candidate_ops: Int32Array,
    candidate_depth: Int32Array,
    candidate_generator_policy: Int32Array,
    promotion_active_slot: int = -1,
    promotion_candidate_slot: int = -1,
    direct_active_replacement_slot: int = -1,
    cascade_refill_mask: BoolArray | None = None,
    ordinary_candidate_refresh_slot: int = -1,
    post_promotion_candidate_refresh_slot: int | None = None,
) -> GeneratedBirthIdentityEvent:
    """Build exact redundant masks for one staged/final structural transition."""

    _validate_state(config, pre_state)
    _require_transition_available(pre_state)
    _validate_slot(promotion_active_slot, size=config.active_slots, name="promotion_active_slot")
    _validate_slot(
        promotion_candidate_slot,
        size=config.candidate_slots,
        name="promotion_candidate_slot",
    )
    _validate_slot(
        direct_active_replacement_slot,
        size=config.active_slots,
        name="direct_active_replacement_slot",
    )
    _validate_slot(
        ordinary_candidate_refresh_slot,
        size=config.candidate_slots,
        name="ordinary_candidate_refresh_slot",
    )
    if post_promotion_candidate_refresh_slot is None:
        post_promotion_candidate_refresh_slot = promotion_candidate_slot
    _validate_slot(
        post_promotion_candidate_refresh_slot,
        size=config.candidate_slots,
        name="post_promotion_candidate_refresh_slot",
    )
    cascade = (
        np.zeros((config.active_slots,), dtype=np.bool_)
        if cascade_refill_mask is None
        else _exact_array(
            cascade_refill_mask,
            dtype=np.dtype(np.bool_),
            shape=(config.active_slots,),
            name="cascade_refill_mask",
        )
    )
    event = GeneratedBirthIdentityEvent(
        schema=config.schema,
        status=config.status,
        namespace=config.namespace,
        paired_development_life_seed=pre_state.paired_development_life_seed,
        learner_step=learner_step,
        channel_manifest=config.channel_manifest,
        generator_policy_sampled=generator_policy_sampled,
        generator_policy_id=generator_policy_id,
        promotion_active_slot=promotion_active_slot,
        promotion_candidate_slot=promotion_candidate_slot,
        direct_active_replacement_slot=direct_active_replacement_slot,
        ordinary_candidate_refresh_slot=ordinary_candidate_refresh_slot,
        post_promotion_candidate_refresh_slot=post_promotion_candidate_refresh_slot,
        promotion_active_mask=_readonly_copy(
            _singleton_mask(promotion_active_slot, config.active_slots)
        ),
        promotion_candidate_mask=_readonly_copy(
            _singleton_mask(promotion_candidate_slot, config.candidate_slots)
        ),
        direct_active_replacement_mask=_readonly_copy(
            _singleton_mask(direct_active_replacement_slot, config.active_slots)
        ),
        cascade_refill_mask=_readonly_copy(cascade),
        ordinary_candidate_refresh_mask=_readonly_copy(
            _singleton_mask(ordinary_candidate_refresh_slot, config.candidate_slots)
        ),
        post_promotion_candidate_refresh_mask=_readonly_copy(
            _singleton_mask(
                post_promotion_candidate_refresh_slot,
                config.candidate_slots,
            )
        ),
        candidate_rebound_mask=_readonly_copy(np.zeros((config.candidate_slots,), dtype=np.bool_)),
        candidate_overdepth_regeneration_mask=_readonly_copy(
            np.zeros((config.candidate_slots,), dtype=np.bool_)
        ),
        active_parent_a=_exact_array(
            active_parent_a,
            dtype=np.dtype(np.int32),
            shape=(config.active_slots,),
            name="active_parent_a",
        ),
        active_parent_b=_exact_array(
            active_parent_b,
            dtype=np.dtype(np.int32),
            shape=(config.active_slots,),
            name="active_parent_b",
        ),
        active_ops=_exact_array(
            active_ops,
            dtype=np.dtype(np.int32),
            shape=(config.active_slots,),
            name="active_ops",
        ),
        active_depth=_exact_array(
            active_depth,
            dtype=np.dtype(np.int32),
            shape=(config.active_slots,),
            name="active_depth",
        ),
        active_generator_policy=_exact_array(
            active_generator_policy,
            dtype=np.dtype(np.int32),
            shape=(config.active_slots,),
            name="active_generator_policy",
        ),
        candidate_staged_parent_a=_exact_array(
            candidate_staged_parent_a,
            dtype=np.dtype(np.int32),
            shape=(config.candidate_slots,),
            name="candidate_staged_parent_a",
        ),
        candidate_staged_parent_b=_exact_array(
            candidate_staged_parent_b,
            dtype=np.dtype(np.int32),
            shape=(config.candidate_slots,),
            name="candidate_staged_parent_b",
        ),
        candidate_staged_ops=_exact_array(
            candidate_staged_ops,
            dtype=np.dtype(np.int32),
            shape=(config.candidate_slots,),
            name="candidate_staged_ops",
        ),
        candidate_staged_depth=_exact_array(
            candidate_staged_depth,
            dtype=np.dtype(np.int32),
            shape=(config.candidate_slots,),
            name="candidate_staged_depth",
        ),
        candidate_staged_generator_policy=_exact_array(
            candidate_staged_generator_policy,
            dtype=np.dtype(np.int32),
            shape=(config.candidate_slots,),
            name="candidate_staged_generator_policy",
        ),
        candidate_parent_a=_exact_array(
            candidate_parent_a,
            dtype=np.dtype(np.int32),
            shape=(config.candidate_slots,),
            name="candidate_parent_a",
        ),
        candidate_parent_b=_exact_array(
            candidate_parent_b,
            dtype=np.dtype(np.int32),
            shape=(config.candidate_slots,),
            name="candidate_parent_b",
        ),
        candidate_ops=_exact_array(
            candidate_ops,
            dtype=np.dtype(np.int32),
            shape=(config.candidate_slots,),
            name="candidate_ops",
        ),
        candidate_depth=_exact_array(
            candidate_depth,
            dtype=np.dtype(np.int32),
            shape=(config.candidate_slots,),
            name="candidate_depth",
        ),
        candidate_generator_policy=_exact_array(
            candidate_generator_policy,
            dtype=np.dtype(np.int32),
            shape=(config.candidate_slots,),
            name="candidate_generator_policy",
        ),
        integrity_sha256="0" * 64,
    )
    _validate_event_structure(config, pre_state, event, check_integrity=False)
    transition = _compute_transition_arrays(config, pre_state, event)
    event = dataclasses.replace(
        event,
        candidate_rebound_mask=_readonly_copy(transition.canonical_rebound_mask),
        candidate_overdepth_regeneration_mask=_readonly_copy(
            transition.canonical_overdepth_regeneration_mask
        ),
    )
    event = dataclasses.replace(
        event,
        integrity_sha256=generated_birth_identity_event_sha256(event),
    )
    _validate_event_structure(config, pre_state, event, check_integrity=True)
    return event


def _event_fixed_array_nbytes(event: GeneratedBirthIdentityEvent) -> int:
    return sum(getattr(event, name).nbytes for name in _EVENT_ARRAY_FIELD_NAMES)


def _rebuild_transaction(
    config: GeneratedBirthIdentityLedgerConfig,
    pre_state: GeneratedBirthIdentityLedgerState,
    event: GeneratedBirthIdentityEvent,
) -> GeneratedBirthIdentityTransaction:
    transition = _compute_transition_arrays(config, pre_state, event)
    _require(
        np.array_equal(event.candidate_rebound_mask, transition.canonical_rebound_mask),
        "candidate rebound mask is stale or not the exact structural classification",
    )
    _require(
        np.array_equal(
            event.candidate_overdepth_regeneration_mask,
            transition.canonical_overdepth_regeneration_mask,
        ),
        "candidate overdepth-regeneration mask is stale or not the exact depth class",
    )
    post_state = _make_state(
        config,
        seed=pre_state.paired_development_life_seed,
        step=event.learner_step,
        active_identity=transition.post_active,
        active_parent_a=event.active_parent_a,
        active_parent_b=event.active_parent_b,
        active_ops=event.active_ops,
        active_depth=event.active_depth,
        active_generator_policy=event.active_generator_policy,
        active_generator_policy_sampled=(transition.post_active_generator_policy_sampled),
        candidate_identity=transition.post_candidate,
        candidate_parent_a=event.candidate_parent_a,
        candidate_parent_b=event.candidate_parent_b,
        candidate_ops=event.candidate_ops,
        candidate_depth=event.candidate_depth,
        candidate_generator_policy=event.candidate_generator_policy,
        candidate_generator_policy_sampled=(transition.post_candidate_generator_policy_sampled),
    )
    expected_state_nbytes = 117 * (config.active_slots + config.candidate_slots)
    expected_assignment_nbytes = _IDENTITY_BYTES * (
        3 * config.active_slots + 4 * config.candidate_slots
    )
    expected_event_nbytes = 23 * config.active_slots + 45 * config.candidate_slots
    event_nbytes = _event_fixed_array_nbytes(event)
    _require(event_nbytes == expected_event_nbytes, "event fixed-array byte count is invalid")
    promotion_count = int(np.count_nonzero(event.promotion_active_mask))
    direct_count = int(np.count_nonzero(event.direct_active_replacement_mask))
    cascade_count = int(np.count_nonzero(event.cascade_refill_mask))
    ordinary_count = int(np.count_nonzero(event.ordinary_candidate_refresh_mask))
    post_count = int(np.count_nonzero(event.post_promotion_candidate_refresh_mask))
    rebound_count = int(np.count_nonzero(event.candidate_rebound_mask))
    overdepth_count = int(np.count_nonzero(event.candidate_overdepth_regeneration_mask))
    just_refreshed_rebound = int(
        np.count_nonzero(
            event.candidate_rebound_mask
            & (event.ordinary_candidate_refresh_mask | event.post_promotion_candidate_refresh_mask)
        )
    )
    just_refreshed_overdepth = int(
        np.count_nonzero(
            event.candidate_overdepth_regeneration_mask
            & (event.ordinary_candidate_refresh_mask | event.post_promotion_candidate_refresh_mask)
        )
    )
    applied_count = (
        direct_count + cascade_count + ordinary_count + post_count + rebound_count + overdepth_count
    )
    audit = GeneratedBirthIdentityLedgerAudit(
        schema=config.schema,
        status=config.status,
        config_sha256=_config_sha256(config),
        pre_state_sha256=pre_state.integrity_sha256,
        event_sha256=event.integrity_sha256,
        assignments_sha256=transition.assignments.integrity_sha256,
        post_state_sha256=post_state.integrity_sha256,
        transaction_sha256="0" * 64,
        channel_manifest=config.channel_manifest,
        promotion_transfer_count=promotion_count,
        direct_active_birth_count=direct_count,
        cascade_active_birth_count=cascade_count,
        ordinary_candidate_birth_count=ordinary_count,
        post_promotion_candidate_birth_count=post_count,
        candidate_rebound_count=rebound_count,
        candidate_overdepth_regeneration_count=overdepth_count,
        just_refreshed_candidate_rebound_count=just_refreshed_rebound,
        just_refreshed_candidate_overdepth_regeneration_count=(just_refreshed_overdepth),
        applied_identity_event_count=applied_count,
        state_persistent_array_nbytes=post_state.persistent_array_nbytes,
        expected_state_persistent_array_nbytes=expected_state_nbytes,
        expected_state_persistent_array_nbytes_formula=_STATE_NBYTES_FORMULA,
        assignment_persistent_array_nbytes=(transition.assignments.persistent_array_nbytes),
        expected_assignment_persistent_array_nbytes=expected_assignment_nbytes,
        expected_assignment_persistent_array_nbytes_formula=(_ASSIGNMENT_NBYTES_FORMULA),
        event_fixed_array_nbytes=event_nbytes,
        expected_event_fixed_array_nbytes=expected_event_nbytes,
        expected_event_fixed_array_nbytes_formula=_EVENT_NBYTES_FORMULA,
        identity_array_noop_with_monotone_step_advance=applied_count == 0,
        post_promotion_refresh_then_cascade_then_candidate_resolution=True,
        structural_lifetime_descriptor_complete=True,
        active_parent_snapshot_complete=True,
        candidate_parent_snapshot_complete=True,
        raw_source_identity_bound=True,
        active_lineage_consistent=transition.active_lineage_consistent,
        theta_bound_to_structural_lifetime_identity=False,
        theta_may_adapt_within_structural_lifetime=True,
        exact_functional_expression_identity_claimed=False,
        transition_collision_observed=False,
        dead_identity_history_retained=False,
        historical_global_uniqueness_claimed=False,
        collision_resistance_primitive="SHA-256",
        cryptographic_collision_impossibility_claimed=False,
        runner_side_state_only=True,
        compositional_feature_state_fields_added=False,
        caller_supplied_events_authenticated=False,
        public_core_event_trace_required_for_authentication=True,
        public_core_event_trace_available=True,
        public_core_event_trace_consumed=False,
        lifecycle_prerequisite_complete=False,
        execution_authorized=False,
        runner_authorized=False,
        artifact_writes_authorized=False,
        evidence_authorized=False,
        scientific_promotion_allowed=False,
    )
    transaction = GeneratedBirthIdentityTransaction(
        assignments=transition.assignments,
        post_state=post_state,
        audit=audit,
    )
    transaction = dataclasses.replace(
        transaction,
        audit=dataclasses.replace(
            audit,
            transaction_sha256=generated_birth_identity_transaction_sha256(transaction),
        ),
    )
    _validate_audit(config, transaction.audit)
    return transaction


def build_generated_birth_identity_transaction(
    config: GeneratedBirthIdentityLedgerConfig,
    pre_state: GeneratedBirthIdentityLedgerState,
    event: GeneratedBirthIdentityEvent,
) -> GeneratedBirthIdentityTransaction:
    """Apply one strictly validated caller-supplied sidecar event."""

    _validate_state(config, pre_state)
    _validate_event_structure(config, pre_state, event, check_integrity=True)
    return _rebuild_transaction(config, pre_state, event)


def _transaction_canonical_bytes(transaction: GeneratedBirthIdentityTransaction) -> bytes:
    return _canonical_json_bytes(_transaction_payload(transaction, include_transaction_sha256=True))


def validate_generated_birth_identity_transaction(
    transaction: GeneratedBirthIdentityTransaction,
    *,
    config: GeneratedBirthIdentityLedgerConfig,
    pre_state: GeneratedBirthIdentityLedgerState,
    event: GeneratedBirthIdentityEvent,
) -> GeneratedBirthIdentityValidation:
    """Independently rebuild and byte-compare one complete transaction.

    The expected config, pre-state, and event are separate arguments on
    purpose.  A transaction whose contents and self-hashes were all edited
    together therefore cannot redefine the validator's expected inputs.
    """

    _require(
        type(transaction) is GeneratedBirthIdentityTransaction,
        "transaction type is invalid",
    )
    _validate_state(config, pre_state)
    _validate_event_structure(config, pre_state, event, check_integrity=True)
    _validate_assignments(config, transaction.assignments)
    _validate_transition_collision_scope(config, pre_state, transaction.assignments)
    _validate_state(config, transaction.post_state)
    _validate_audit(config, transaction.audit)
    _require(
        transaction.audit.pre_state_sha256 == pre_state.integrity_sha256,
        "audit pre-state hash is stale",
    )
    _require(
        transaction.audit.event_sha256 == event.integrity_sha256,
        "audit event hash is stale",
    )
    _require(
        transaction.audit.assignments_sha256 == transaction.assignments.integrity_sha256,
        "audit assignment hash is stale",
    )
    _require(
        transaction.audit.post_state_sha256 == transaction.post_state.integrity_sha256,
        "audit post-state hash is stale",
    )
    _require(
        _valid_sha256(transaction.audit.transaction_sha256),
        "transaction integrity hash is malformed",
    )
    _require(
        transaction.audit.transaction_sha256
        == generated_birth_identity_transaction_sha256(transaction),
        "transaction integrity hash does not reconstruct",
    )
    canonical = _rebuild_transaction(config, pre_state, event)
    _require(
        _transaction_canonical_bytes(transaction) == _transaction_canonical_bytes(canonical),
        "transaction differs from the strict independent canonical rebuild",
    )
    return GeneratedBirthIdentityValidation(
        valid=True,
        canonical_transaction_sha256=canonical.audit.transaction_sha256,
        supplied_transaction_sha256=transaction.audit.transaction_sha256,
        caller_supplied_events_authenticated=False,
        lifecycle_prerequisite_complete=False,
        execution_authorized=False,
        runner_authorized=False,
        artifact_writes_authorized=False,
        evidence_authorized=False,
        scientific_promotion_allowed=False,
    )


# ---------------------------------------------------------------------------
# Additive schema v4: canonical uint32[2] lifetime counters.
#
# Schema v3 above remains a complete historical API.  V4 deliberately wraps
# its exhaustively checked fixed-shape structural engine while replacing the
# transition identity domain and state/event integrity domain with exact
# big-endian counter words.  The inner v3 state always uses surrogate step zero
# and therefore carries no identity authority in v4.


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedBirthIdentityLedgerV4Config:
    """Static schema-v4 ledger shape and non-authority disclosures."""

    namespace: str
    active_slots: int
    candidate_slots: int
    raw_feature_slots: int
    max_depth: int
    learn_generator_resources: bool
    generator_policy_count: int = len(GENERATED_BIRTH_IDENTITY_GENERATOR_POLICY_MANIFEST)
    generator_policy_manifest: tuple[str, ...] = GENERATED_BIRTH_IDENTITY_GENERATOR_POLICY_MANIFEST
    channel_manifest: tuple[str, ...] = GENERATED_BIRTH_IDENTITY_CHANNELS
    schema: str = GENERATED_BIRTH_IDENTITY_LEDGER_V4_SCHEMA
    status: str = GENERATED_BIRTH_IDENTITY_LEDGER_V4_STATUS
    development_only: bool = True
    canonical_step_words_bound: bool = True
    scalar_step_is_telemetry_only: bool = True
    execution_authorized: bool = False
    runner_authorized: bool = False
    artifact_writes_authorized: bool = False
    evidence_authorized: bool = False
    scientific_promotion_allowed: bool = False

    def __post_init__(self) -> None:
        _v4_to_v3_config(self)
        _require(self.schema == GENERATED_BIRTH_IDENTITY_LEDGER_V4_SCHEMA, "v4 schema is stale")
        _require(self.status == GENERATED_BIRTH_IDENTITY_LEDGER_V4_STATUS, "v4 status is stale")
        for name in (
            "learn_generator_resources",
            "development_only",
            "canonical_step_words_bound",
            "scalar_step_is_telemetry_only",
            "execution_authorized",
            "runner_authorized",
            "artifact_writes_authorized",
            "evidence_authorized",
            "scientific_promotion_allowed",
        ):
            _exact_bool(getattr(self, name), name=f"v4 config {name}")
        _require(
            self.development_only
            and self.canonical_step_words_bound
            and self.scalar_step_is_telemetry_only,
            "v4 counter disclosures must remain enabled",
        )
        _require(
            not any(
                (
                    self.execution_authorized,
                    self.runner_authorized,
                    self.artifact_writes_authorized,
                    self.evidence_authorized,
                    self.scientific_promotion_allowed,
                )
            ),
            "v4 config cannot grant authority",
        )


def _v4_to_v3_config(
    config: GeneratedBirthIdentityLedgerV4Config,
) -> GeneratedBirthIdentityLedgerConfig:
    return GeneratedBirthIdentityLedgerConfig(
        namespace=config.namespace,
        active_slots=config.active_slots,
        candidate_slots=config.candidate_slots,
        raw_feature_slots=config.raw_feature_slots,
        max_depth=config.max_depth,
        learn_generator_resources=config.learn_generator_resources,
        generator_policy_count=config.generator_policy_count,
        generator_policy_manifest=config.generator_policy_manifest,
        channel_manifest=config.channel_manifest,
    )


def _v4_config_payload(config: GeneratedBirthIdentityLedgerV4Config) -> dict[str, object]:
    return {field.name: getattr(config, field.name) for field in dataclasses.fields(config)}


def _v4_config_sha256(config: GeneratedBirthIdentityLedgerV4Config) -> str:
    return _sha256(_v4_config_payload(config))


def _v4_words(
    value: object,
    *,
    name: str,
    require_immutable: bool = False,
) -> UInt32Array:
    return cast(
        UInt32Array,
        _exact_array(
            value,
            dtype=np.dtype(np.uint32),
            shape=(2,),
            name=name,
            require_immutable=require_immutable,
        ),
    )


def _v4_increment_words(words: UInt32Array) -> UInt32Array:
    high = int(words[0])
    low = int(words[1])
    _require(
        not (high == 2**32 - 1 and low == 2**32 - 1),
        "schema-v4 lifetime counter is exhausted",
    )
    low = (low + 1) & (2**32 - 1)
    if low == 0:
        high = (high + 1) & (2**32 - 1)
    return np.asarray((high, low), dtype=np.uint32)


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedBirthIdentityLedgerV4State:
    """V4 canonical counter plus the fixed-shape structural state."""

    schema: str
    status: str
    config_sha256: str
    step_words: UInt32Array
    structural_state: GeneratedBirthIdentityLedgerState
    persistent_array_nbytes: int
    expected_persistent_array_nbytes_formula: str
    integrity_sha256: str

    @property
    def namespace(self) -> str:
        return self.structural_state.namespace

    @property
    def paired_development_life_seed(self) -> int:
        return self.structural_state.paired_development_life_seed

    @property
    def active_identity(self) -> UInt8Array:
        return self.structural_state.active_identity

    @property
    def candidate_identity(self) -> UInt8Array:
        return self.structural_state.candidate_identity

    @property
    def active_parent_a(self) -> Int32Array:
        return self.structural_state.active_parent_a

    @property
    def active_parent_b(self) -> Int32Array:
        return self.structural_state.active_parent_b

    @property
    def active_ops(self) -> Int32Array:
        return self.structural_state.active_ops

    @property
    def active_depth(self) -> Int32Array:
        return self.structural_state.active_depth

    @property
    def active_generator_policy(self) -> Int32Array:
        return self.structural_state.active_generator_policy

    @property
    def candidate_parent_a(self) -> Int32Array:
        return self.structural_state.candidate_parent_a

    @property
    def candidate_parent_b(self) -> Int32Array:
        return self.structural_state.candidate_parent_b

    @property
    def candidate_ops(self) -> Int32Array:
        return self.structural_state.candidate_ops

    @property
    def candidate_depth(self) -> Int32Array:
        return self.structural_state.candidate_depth

    @property
    def candidate_generator_policy(self) -> Int32Array:
        return self.structural_state.candidate_generator_policy


def _v4_state_payload(state: GeneratedBirthIdentityLedgerV4State) -> dict[str, object]:
    return {
        "schema": state.schema,
        "status": state.status,
        "config_sha256": state.config_sha256,
        "step_words": _array_record(state.step_words),
        "structural_state": {
            **_state_payload(state.structural_state),
            "integrity_sha256": state.structural_state.integrity_sha256,
        },
        "persistent_array_nbytes": state.persistent_array_nbytes,
        "expected_persistent_array_nbytes_formula": (
            state.expected_persistent_array_nbytes_formula
        ),
    }


def generated_birth_identity_ledger_v4_state_sha256(
    state: GeneratedBirthIdentityLedgerV4State,
) -> str:
    return _sha256(_v4_state_payload(state))


def _validate_v4_state(
    config: GeneratedBirthIdentityLedgerV4Config,
    state: GeneratedBirthIdentityLedgerV4State,
) -> None:
    _require(type(state) is GeneratedBirthIdentityLedgerV4State, "v4 state type is invalid")
    _require(state.schema == config.schema, "v4 state schema is stale")
    _require(state.status == config.status, "v4 state status is stale")
    _require(state.config_sha256 == _v4_config_sha256(config), "v4 config hash is stale")
    _v4_words(state.step_words, name="v4 state step_words", require_immutable=True)
    v3_config = _v4_to_v3_config(config)
    _validate_state(v3_config, state.structural_state)
    _require(
        state.structural_state.learner_step == 0,
        "v4 structural engine step must remain the unauthoritative zero surrogate",
    )
    expected_nbytes = 117 * (config.active_slots + config.candidate_slots) + 8
    _require(state.persistent_array_nbytes == expected_nbytes, "v4 state byte count is stale")
    _require(
        state.expected_persistent_array_nbytes_formula
        == "117 * (active_slots + candidate_slots) + 8",
        "v4 state byte formula is stale",
    )
    _require(
        state.integrity_sha256 == generated_birth_identity_ledger_v4_state_sha256(state),
        "v4 state integrity hash does not reconstruct",
    )


def _make_v4_state(
    config: GeneratedBirthIdentityLedgerV4Config,
    *,
    step_words: UInt32Array,
    structural_state: GeneratedBirthIdentityLedgerState,
) -> GeneratedBirthIdentityLedgerV4State:
    expected_nbytes = 117 * (config.active_slots + config.candidate_slots) + 8
    state = GeneratedBirthIdentityLedgerV4State(
        schema=config.schema,
        status=config.status,
        config_sha256=_v4_config_sha256(config),
        step_words=_readonly_copy(step_words),
        structural_state=structural_state,
        persistent_array_nbytes=expected_nbytes,
        expected_persistent_array_nbytes_formula=("117 * (active_slots + candidate_slots) + 8"),
        integrity_sha256="0" * 64,
    )
    state = dataclasses.replace(
        state,
        integrity_sha256=generated_birth_identity_ledger_v4_state_sha256(state),
    )
    _validate_v4_state(config, state)
    return state


def initialize_generated_birth_identity_ledger_v4(
    config: GeneratedBirthIdentityLedgerV4Config,
    *,
    paired_development_life_seed: int,
    active_parent_a: Int32Array,
    active_parent_b: Int32Array,
    active_ops: Int32Array,
    active_depth: Int32Array,
    active_generator_policy: Int32Array,
    candidate_parent_a: Int32Array,
    candidate_parent_b: Int32Array,
    candidate_ops: Int32Array,
    candidate_depth: Int32Array,
    candidate_generator_policy: Int32Array,
) -> GeneratedBirthIdentityLedgerV4State:
    """Create a schema-v4 genesis at exact words ``[0, 0]``."""

    _require(type(config) is GeneratedBirthIdentityLedgerV4Config, "v4 config type is invalid")
    v3_config = _v4_to_v3_config(config)
    genesis = initialize_generated_birth_identity_ledger(
        v3_config,
        paired_development_life_seed=paired_development_life_seed,
        learner_step=0,
        active_parent_a=active_parent_a,
        active_parent_b=active_parent_b,
        active_ops=active_ops,
        active_depth=active_depth,
        active_generator_policy=active_generator_policy,
        candidate_parent_a=candidate_parent_a,
        candidate_parent_b=candidate_parent_b,
        candidate_ops=candidate_ops,
        candidate_depth=candidate_depth,
        candidate_generator_policy=candidate_generator_policy,
    )
    zero_words = np.zeros((2,), dtype=np.uint32)
    active = np.stack(
        [
            _identity_array(
                namespace=config.namespace,
                seed=paired_development_life_seed,
                step=0,
                step_words=zero_words,
                channel=INITIAL_ACTIVE_CHANNEL,
                slot=slot,
                ordinal=slot,
            )
            for slot in range(config.active_slots)
        ],
        axis=0,
    )
    candidate = (
        np.stack(
            [
                _identity_array(
                    namespace=config.namespace,
                    seed=paired_development_life_seed,
                    step=0,
                    step_words=zero_words,
                    channel=INITIAL_CANDIDATE_CHANNEL,
                    slot=slot,
                    ordinal=slot,
                )
                for slot in range(config.candidate_slots)
            ],
            axis=0,
        )
        if config.candidate_slots > 0
        else np.zeros((0, _IDENTITY_BYTES), dtype=np.uint8)
    )
    structural = _make_state(
        v3_config,
        seed=paired_development_life_seed,
        step=0,
        active_identity=active,
        active_parent_a=genesis.active_parent_a,
        active_parent_b=genesis.active_parent_b,
        active_ops=genesis.active_ops,
        active_depth=genesis.active_depth,
        active_generator_policy=genesis.active_generator_policy,
        active_generator_policy_sampled=genesis.active_generator_policy_sampled,
        candidate_identity=candidate,
        candidate_parent_a=genesis.candidate_parent_a,
        candidate_parent_b=genesis.candidate_parent_b,
        candidate_ops=genesis.candidate_ops,
        candidate_depth=genesis.candidate_depth,
        candidate_generator_policy=genesis.candidate_generator_policy,
        candidate_generator_policy_sampled=(genesis.candidate_generator_policy_sampled),
    )
    return _make_v4_state(config, step_words=zero_words, structural_state=structural)


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedBirthIdentityLedgerV4Event:
    """Canonical counter words bound to one exact v3 structural declaration."""

    schema: str
    status: str
    config_sha256: str
    pre_step_words: UInt32Array
    post_step_words: UInt32Array
    structural_event: GeneratedBirthIdentityEvent
    integrity_sha256: str

    @property
    def cascade_refill_mask(self) -> BoolArray:
        return self.structural_event.cascade_refill_mask

    @property
    def candidate_rebound_mask(self) -> BoolArray:
        return self.structural_event.candidate_rebound_mask

    @property
    def candidate_overdepth_regeneration_mask(self) -> BoolArray:
        return self.structural_event.candidate_overdepth_regeneration_mask


def _v4_event_payload(event: GeneratedBirthIdentityLedgerV4Event) -> dict[str, object]:
    return {
        "schema": event.schema,
        "status": event.status,
        "config_sha256": event.config_sha256,
        "pre_step_words": _array_record(event.pre_step_words),
        "post_step_words": _array_record(event.post_step_words),
        "structural_event": {
            **_event_payload(event.structural_event),
            "integrity_sha256": event.structural_event.integrity_sha256,
        },
    }


def generated_birth_identity_ledger_v4_event_sha256(
    event: GeneratedBirthIdentityLedgerV4Event,
) -> str:
    return _sha256(_v4_event_payload(event))


def _validate_v4_event(
    config: GeneratedBirthIdentityLedgerV4Config,
    pre_state: GeneratedBirthIdentityLedgerV4State,
    event: GeneratedBirthIdentityLedgerV4Event,
) -> None:
    _require(type(event) is GeneratedBirthIdentityLedgerV4Event, "v4 event type is invalid")
    _validate_v4_state(config, pre_state)
    _require(event.schema == config.schema, "v4 event schema is stale")
    _require(event.status == config.status, "v4 event status is stale")
    _require(event.config_sha256 == _v4_config_sha256(config), "v4 event config hash is stale")
    pre_words = _v4_words(
        event.pre_step_words,
        name="v4 event pre_step_words",
        require_immutable=True,
    )
    post_words = _v4_words(
        event.post_step_words,
        name="v4 event post_step_words",
        require_immutable=True,
    )
    _require(np.array_equal(pre_words, pre_state.step_words), "v4 event pre words are stale")
    _require(
        np.array_equal(post_words, _v4_increment_words(pre_words)),
        "v4 event post words are not the exact checked increment",
    )
    _validate_event_structure(
        _v4_to_v3_config(config),
        pre_state.structural_state,
        event.structural_event,
        check_integrity=True,
    )
    _require(
        event.integrity_sha256 == generated_birth_identity_ledger_v4_event_sha256(event),
        "v4 event integrity hash does not reconstruct",
    )


def build_generated_birth_identity_event_v4(
    config: GeneratedBirthIdentityLedgerV4Config,
    pre_state: GeneratedBirthIdentityLedgerV4State,
    *,
    post_step_words: UInt32Array,
    generator_policy_sampled: bool,
    generator_policy_id: int,
    active_parent_a: Int32Array,
    active_parent_b: Int32Array,
    active_ops: Int32Array,
    active_depth: Int32Array,
    active_generator_policy: Int32Array,
    candidate_staged_parent_a: Int32Array,
    candidate_staged_parent_b: Int32Array,
    candidate_staged_ops: Int32Array,
    candidate_staged_depth: Int32Array,
    candidate_staged_generator_policy: Int32Array,
    candidate_parent_a: Int32Array,
    candidate_parent_b: Int32Array,
    candidate_ops: Int32Array,
    candidate_depth: Int32Array,
    candidate_generator_policy: Int32Array,
    promotion_active_slot: int = -1,
    promotion_candidate_slot: int = -1,
    direct_active_replacement_slot: int = -1,
    cascade_refill_mask: BoolArray | None = None,
    ordinary_candidate_refresh_slot: int = -1,
    post_promotion_candidate_refresh_slot: int | None = None,
) -> GeneratedBirthIdentityLedgerV4Event:
    """Build all structural masks while binding the exact v4 counter advance."""

    _validate_v4_state(config, pre_state)
    supplied_post = _v4_words(post_step_words, name="post_step_words")
    _require(
        np.array_equal(supplied_post, _v4_increment_words(pre_state.step_words)),
        "post_step_words are not the exact checked increment",
    )
    structural = build_generated_birth_identity_event(
        _v4_to_v3_config(config),
        pre_state.structural_state,
        learner_step=1,
        generator_policy_sampled=generator_policy_sampled,
        generator_policy_id=generator_policy_id,
        active_parent_a=active_parent_a,
        active_parent_b=active_parent_b,
        active_ops=active_ops,
        active_depth=active_depth,
        active_generator_policy=active_generator_policy,
        candidate_staged_parent_a=candidate_staged_parent_a,
        candidate_staged_parent_b=candidate_staged_parent_b,
        candidate_staged_ops=candidate_staged_ops,
        candidate_staged_depth=candidate_staged_depth,
        candidate_staged_generator_policy=candidate_staged_generator_policy,
        candidate_parent_a=candidate_parent_a,
        candidate_parent_b=candidate_parent_b,
        candidate_ops=candidate_ops,
        candidate_depth=candidate_depth,
        candidate_generator_policy=candidate_generator_policy,
        promotion_active_slot=promotion_active_slot,
        promotion_candidate_slot=promotion_candidate_slot,
        direct_active_replacement_slot=direct_active_replacement_slot,
        cascade_refill_mask=cascade_refill_mask,
        ordinary_candidate_refresh_slot=ordinary_candidate_refresh_slot,
        post_promotion_candidate_refresh_slot=post_promotion_candidate_refresh_slot,
    )
    event = GeneratedBirthIdentityLedgerV4Event(
        schema=config.schema,
        status=config.status,
        config_sha256=_v4_config_sha256(config),
        pre_step_words=_readonly_copy(pre_state.step_words),
        post_step_words=_readonly_copy(supplied_post),
        structural_event=structural,
        integrity_sha256="0" * 64,
    )
    event = dataclasses.replace(
        event,
        integrity_sha256=generated_birth_identity_ledger_v4_event_sha256(event),
    )
    _validate_v4_event(config, pre_state, event)
    return event


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedBirthIdentityLedgerV4Audit:
    """Canonical integrity chain and explicit schema-v4 non-authority record."""

    schema: str
    status: str
    config_sha256: str
    pre_state_sha256: str
    event_sha256: str
    assignments_sha256: str
    post_state_sha256: str
    transaction_sha256: str
    pre_step_words: tuple[int, int]
    post_step_words: tuple[int, int]
    promotion_transfer_count: int
    applied_identity_event_count: int
    canonical_step_words_bound: bool
    scalar_step_is_telemetry_only: bool
    transition_collision_observed: bool
    development_only: bool
    execution_authorized: bool
    runner_authorized: bool
    artifact_writes_authorized: bool
    evidence_authorized: bool
    scientific_promotion_allowed: bool


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedBirthIdentityLedgerV4Transaction:
    """One canonical-word transition with fixed-shape assignments."""

    assignments: GeneratedBirthIdentityAssignments
    post_state: GeneratedBirthIdentityLedgerV4State
    audit: GeneratedBirthIdentityLedgerV4Audit


def _v4_audit_payload(
    audit: GeneratedBirthIdentityLedgerV4Audit,
    *,
    include_transaction_sha256: bool,
) -> dict[str, object]:
    result = {field.name: getattr(audit, field.name) for field in dataclasses.fields(audit)}
    if not include_transaction_sha256:
        result.pop("transaction_sha256")
    return result


def _v4_transaction_payload(
    transaction: GeneratedBirthIdentityLedgerV4Transaction,
    *,
    include_transaction_sha256: bool,
) -> dict[str, object]:
    return {
        "assignments": {
            **_assignments_payload(transaction.assignments),
            "integrity_sha256": transaction.assignments.integrity_sha256,
        },
        "post_state": {
            **_v4_state_payload(transaction.post_state),
            "integrity_sha256": transaction.post_state.integrity_sha256,
        },
        "audit": _v4_audit_payload(
            transaction.audit,
            include_transaction_sha256=include_transaction_sha256,
        ),
    }


def generated_birth_identity_ledger_v4_transaction_sha256(
    transaction: GeneratedBirthIdentityLedgerV4Transaction,
) -> str:
    return _sha256(_v4_transaction_payload(transaction, include_transaction_sha256=False))


def _count_assignment_rows(bank: UInt8Array) -> int:
    return int(np.count_nonzero(np.any(bank != np.uint8(0), axis=1)))


def _validate_v4_audit(
    config: GeneratedBirthIdentityLedgerV4Config,
    pre_state: GeneratedBirthIdentityLedgerV4State,
    event: GeneratedBirthIdentityLedgerV4Event,
    transaction: GeneratedBirthIdentityLedgerV4Transaction,
) -> None:
    audit = transaction.audit
    _require(type(audit) is GeneratedBirthIdentityLedgerV4Audit, "v4 audit type is invalid")
    _require(audit.schema == config.schema, "v4 audit schema is stale")
    _require(audit.status == config.status, "v4 audit status is stale")
    _require(audit.config_sha256 == _v4_config_sha256(config), "v4 audit config hash is stale")
    _require(audit.pre_state_sha256 == pre_state.integrity_sha256, "v4 audit pre hash is stale")
    _require(audit.event_sha256 == event.integrity_sha256, "v4 audit event hash is stale")
    _require(
        audit.assignments_sha256 == transaction.assignments.integrity_sha256,
        "v4 audit assignment hash is stale",
    )
    _require(
        audit.post_state_sha256 == transaction.post_state.integrity_sha256,
        "v4 audit post hash is stale",
    )
    expected_pre_words = tuple(int(word) for word in pre_state.step_words)
    expected_post_words = tuple(int(word) for word in event.post_step_words)
    _require(audit.pre_step_words == expected_pre_words, "v4 audit pre words are stale")
    _require(audit.post_step_words == expected_post_words, "v4 audit post words are stale")
    transfer_count = _count_assignment_rows(
        transaction.assignments.promotion_transfer_active_identity
    )
    applied_count = sum(
        _count_assignment_rows(getattr(transaction.assignments, name))
        for name in (
            "direct_active_birth_identity",
            "cascade_active_birth_identity",
            "ordinary_candidate_birth_identity",
            "post_promotion_candidate_birth_identity",
            "candidate_rebound_identity",
            "candidate_overdepth_regeneration_identity",
        )
    )
    _require(audit.promotion_transfer_count == transfer_count, "v4 transfer count is stale")
    _require(audit.applied_identity_event_count == applied_count, "v4 event count is stale")
    for name in (
        "canonical_step_words_bound",
        "scalar_step_is_telemetry_only",
        "transition_collision_observed",
        "development_only",
        "execution_authorized",
        "runner_authorized",
        "artifact_writes_authorized",
        "evidence_authorized",
        "scientific_promotion_allowed",
    ):
        _exact_bool(getattr(audit, name), name=f"v4 audit {name}")
    _require(
        audit.canonical_step_words_bound
        and audit.scalar_step_is_telemetry_only
        and audit.development_only,
        "v4 audit counter disclosures are stale",
    )
    _require(
        not any(
            (
                audit.transition_collision_observed,
                audit.execution_authorized,
                audit.runner_authorized,
                audit.artifact_writes_authorized,
                audit.evidence_authorized,
                audit.scientific_promotion_allowed,
            )
        ),
        "v4 audit cannot report collision or grant authority",
    )
    _require(
        audit.transaction_sha256
        == generated_birth_identity_ledger_v4_transaction_sha256(transaction),
        "v4 transaction hash does not reconstruct",
    )


def build_generated_birth_identity_transaction_v4(
    config: GeneratedBirthIdentityLedgerV4Config,
    pre_state: GeneratedBirthIdentityLedgerV4State,
    event: GeneratedBirthIdentityLedgerV4Event,
) -> GeneratedBirthIdentityLedgerV4Transaction:
    """Apply one exact schema-v4 event in the canonical word identity domain."""

    _validate_v4_event(config, pre_state, event)
    v3_config = _v4_to_v3_config(config)
    transition = _compute_transition_arrays(
        v3_config,
        pre_state.structural_state,
        event.structural_event,
        identity_step_words=event.post_step_words,
    )
    _require(
        np.array_equal(
            event.structural_event.candidate_rebound_mask,
            transition.canonical_rebound_mask,
        ),
        "v4 candidate rebound mask is stale",
    )
    _require(
        np.array_equal(
            event.structural_event.candidate_overdepth_regeneration_mask,
            transition.canonical_overdepth_regeneration_mask,
        ),
        "v4 candidate overdepth mask is stale",
    )
    structural_post = _make_state(
        v3_config,
        seed=pre_state.paired_development_life_seed,
        step=0,
        active_identity=transition.post_active,
        active_parent_a=event.structural_event.active_parent_a,
        active_parent_b=event.structural_event.active_parent_b,
        active_ops=event.structural_event.active_ops,
        active_depth=event.structural_event.active_depth,
        active_generator_policy=event.structural_event.active_generator_policy,
        active_generator_policy_sampled=(transition.post_active_generator_policy_sampled),
        candidate_identity=transition.post_candidate,
        candidate_parent_a=event.structural_event.candidate_parent_a,
        candidate_parent_b=event.structural_event.candidate_parent_b,
        candidate_ops=event.structural_event.candidate_ops,
        candidate_depth=event.structural_event.candidate_depth,
        candidate_generator_policy=(event.structural_event.candidate_generator_policy),
        candidate_generator_policy_sampled=(transition.post_candidate_generator_policy_sampled),
    )
    post_state = _make_v4_state(
        config,
        step_words=event.post_step_words,
        structural_state=structural_post,
    )
    transfer_count = _count_assignment_rows(
        transition.assignments.promotion_transfer_active_identity
    )
    applied_count = sum(
        _count_assignment_rows(getattr(transition.assignments, name))
        for name in (
            "direct_active_birth_identity",
            "cascade_active_birth_identity",
            "ordinary_candidate_birth_identity",
            "post_promotion_candidate_birth_identity",
            "candidate_rebound_identity",
            "candidate_overdepth_regeneration_identity",
        )
    )
    audit = GeneratedBirthIdentityLedgerV4Audit(
        schema=config.schema,
        status=config.status,
        config_sha256=_v4_config_sha256(config),
        pre_state_sha256=pre_state.integrity_sha256,
        event_sha256=event.integrity_sha256,
        assignments_sha256=transition.assignments.integrity_sha256,
        post_state_sha256=post_state.integrity_sha256,
        transaction_sha256="0" * 64,
        pre_step_words=(int(pre_state.step_words[0]), int(pre_state.step_words[1])),
        post_step_words=(int(event.post_step_words[0]), int(event.post_step_words[1])),
        promotion_transfer_count=transfer_count,
        applied_identity_event_count=applied_count,
        canonical_step_words_bound=True,
        scalar_step_is_telemetry_only=True,
        transition_collision_observed=False,
        development_only=True,
        execution_authorized=False,
        runner_authorized=False,
        artifact_writes_authorized=False,
        evidence_authorized=False,
        scientific_promotion_allowed=False,
    )
    transaction = GeneratedBirthIdentityLedgerV4Transaction(
        assignments=transition.assignments,
        post_state=post_state,
        audit=audit,
    )
    transaction = dataclasses.replace(
        transaction,
        audit=dataclasses.replace(
            audit,
            transaction_sha256=(generated_birth_identity_ledger_v4_transaction_sha256(transaction)),
        ),
    )
    _validate_assignments(v3_config, transaction.assignments)
    _validate_v4_state(config, transaction.post_state)
    _validate_v4_audit(config, pre_state, event, transaction)
    return transaction


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedBirthIdentityLedgerV4Validation:
    """Successful independent canonical reconstruction of one v4 transaction."""

    valid: bool
    canonical_transaction_sha256: str
    supplied_transaction_sha256: str
    canonical_step_words_bound: bool
    development_only: bool
    execution_authorized: bool
    runner_authorized: bool
    artifact_writes_authorized: bool
    evidence_authorized: bool
    scientific_promotion_allowed: bool

    def __post_init__(self) -> None:
        _require(self.valid, "v4 validation must be true")
        _require(self.canonical_step_words_bound, "v4 validation must bind counter words")
        _require(self.development_only, "v4 validation must remain development-only")
        _require(
            not any(
                (
                    self.execution_authorized,
                    self.runner_authorized,
                    self.artifact_writes_authorized,
                    self.evidence_authorized,
                    self.scientific_promotion_allowed,
                )
            ),
            "v4 validation cannot grant authority",
        )


def validate_generated_birth_identity_transaction_v4(
    transaction: GeneratedBirthIdentityLedgerV4Transaction,
    *,
    config: GeneratedBirthIdentityLedgerV4Config,
    pre_state: GeneratedBirthIdentityLedgerV4State,
    event: GeneratedBirthIdentityLedgerV4Event,
) -> GeneratedBirthIdentityLedgerV4Validation:
    """Independently rebuild and byte-compare one complete v4 transaction."""

    _require(
        type(transaction) is GeneratedBirthIdentityLedgerV4Transaction,
        "v4 transaction type is invalid",
    )
    canonical = build_generated_birth_identity_transaction_v4(config, pre_state, event)
    canonical_bytes = _canonical_json_bytes(
        _v4_transaction_payload(canonical, include_transaction_sha256=True)
    )
    supplied_bytes = _canonical_json_bytes(
        _v4_transaction_payload(transaction, include_transaction_sha256=True)
    )
    _require(canonical_bytes == supplied_bytes, "v4 transaction is not canonical")
    return GeneratedBirthIdentityLedgerV4Validation(
        valid=True,
        canonical_transaction_sha256=canonical.audit.transaction_sha256,
        supplied_transaction_sha256=transaction.audit.transaction_sha256,
        canonical_step_words_bound=True,
        development_only=True,
        execution_authorized=False,
        runner_authorized=False,
        artifact_writes_authorized=False,
        evidence_authorized=False,
        scientific_promotion_allowed=False,
    )
