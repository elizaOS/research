"""Inert exact structural measurement ledger for matched Forager v3.

This module is the first additive implementation beneath the source-only
algorithmic-resource contract.  It records committed transaction counts and
logical, simultaneously-live owned-array trees.  It does not import or invoke a
runner, inspect a reward-bearing artifact, authorize execution, compare a
ceiling, or issue a qualification record.

The ledger intentionally accepts explicit structural observations rather than
introspecting arbitrary Python or JAX objects.  A future pinned family producer
must classify each live leaf, observe every allocation-lifecycle boundary, and
commit transactions only after the corresponding runner operation succeeds.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Literal, NoReturn, cast

from alberta_framework.benchmarks import (
    forager_matched_v3_algorithmic_resource_contract as resource_contract,
)

MEASUREMENT_LEDGER_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.algorithmic_resource_measurement_ledger_descriptor.v1"
)
MEASUREMENT_BASIS_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.algorithmic_resource_measurement_basis.v1"
)
MEASUREMENT_LEDGER_STATUS: Final = (
    "implemented_inert_structural_ledger_uninvoked_no_production_identity"
)
MEASUREMENT_BASIS_STATUS: Final = "inert_structural_measurement_basis_unqualified_non_authorizing"
MEASUREMENT_CLASSIFICATION: Final = (
    "score_blind_structural_algorithmic_resource_measurement_non_authorizing"
)

# Independently replayed after the descriptor audit.  The descriptor never
# embeds this value (or its source-file digest), avoiding a self-referential
# identity.  A producing upstream component supplies the independently audited
# measurement-source digest when it creates a ledger.
PINNED_MEASUREMENT_LEDGER_DESCRIPTOR_SHA256: Final = (
    "627eba09823a2f914df7957d1fd441907116a5e11a88ebef03bd92de3e3fb950"
)

UPSTREAM_ALGORITHMIC_RESOURCE_CONTRACT_SOURCE_SHA256: Final = (
    "c0df02b504d3d5695782f0b68b1518ae4b549a5e13074c7a5ce6dd39313abef3"
)

MAX_MEASUREMENT_BASIS_BYTES: Final = 4 * 1024 * 1024
MAX_TEXT_LENGTH: Final = 16_384
MAX_JSON_DEPTH: Final = 32
MAX_JSON_NODES: Final = 40_000
MAX_ARRAY_RANK: Final = 16
MAX_ARRAY_DIMENSION: Final = 2**31 - 1
MAX_LEAVES_PER_SNAPSHOT: Final = 16_384
MAX_INTEGER: Final = 2**63 - 1

ArrayCategory = Literal[
    "trainable_parameters",
    "frozen_parameters",
    "optimizer_state",
    "target_copy",
    "replay_storage",
    "rollout_storage",
    "recurrent_carry",
    "rtrl_sensitivity",
    "eligibility_trace",
]
StorageKind = Literal["owned_array", "array_view"]

ARRAY_CATEGORIES: Final[tuple[ArrayCategory, ...]] = (
    "trainable_parameters",
    "frozen_parameters",
    "optimizer_state",
    "target_copy",
    "replay_storage",
    "rollout_storage",
    "recurrent_carry",
    "rtrl_sensitivity",
    "eligibility_trace",
)
COUPLED_FIELD_PAIRS: Final = resource_contract.COUPLED_RESOURCE_FIELD_PAIRS

_COUNTER_SUBJECTS: Final = (
    "optimizer_updates",
    "gradient_updates",
    "sample_updates",
)
_ABSENCE_SUBJECTS: Final = _COUNTER_SUBJECTS + ARRAY_CATEGORIES
ABSENCE_KIND_BY_SUBJECT: Final[Mapping[str, str]] = MappingProxyType(
    {
        "optimizer_updates": "optimizer_subsystem_absent",
        "gradient_updates": "gradient_update_path_absent",
        "sample_updates": "sample_update_path_absent",
        "trainable_parameters": "trainable_parameter_tree_absent",
        "frozen_parameters": "frozen_parameter_tree_absent",
        "optimizer_state": "optimizer_state_tree_absent",
        "target_copy": "target_copy_tree_absent",
        "replay_storage": "replay_subsystem_absent",
        "rollout_storage": "rollout_storage_absent",
        "recurrent_carry": "recurrent_carry_absent",
        "rtrl_sensitivity": "rtrl_sensitivity_absent",
        "eligibility_trace": "eligibility_trace_absent",
    }
)

_CATEGORY_TO_FIELDS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "trainable_parameters": ("max_trainable_parameters",),
        "frozen_parameters": ("max_frozen_parameters",),
        "optimizer_state": (
            "max_optimizer_state_elements",
            "max_optimizer_state_bytes",
        ),
        "target_copy": ("max_target_copy_elements", "max_target_copy_bytes"),
        "replay_storage": (
            "max_replay_capacity_transitions",
            "max_replay_peak_bytes",
        ),
        "rollout_storage": (
            "max_rollout_storage_elements",
            "max_rollout_peak_bytes",
        ),
        "recurrent_carry": (
            "max_recurrent_carry_elements",
            "max_recurrent_carry_bytes",
        ),
        "rtrl_sensitivity": (
            "max_rtrl_sensitivity_elements",
            "max_rtrl_sensitivity_bytes",
        ),
        "eligibility_trace": (
            "max_eligibility_elements",
            "max_eligibility_bytes",
        ),
    }
)

_FIELD_TO_SUBJECT: Final[Mapping[str, str]] = MappingProxyType(
    {
        "max_optimizer_updates": "optimizer_updates",
        "max_gradient_updates": "gradient_updates",
        "max_sample_updates": "sample_updates",
        **{
            field_name: category
            for category, field_names in _CATEGORY_TO_FIELDS.items()
            for field_name in field_names
        },
    }
)

_DTYPE_ITEMSIZE: Final[Mapping[str, int]] = MappingProxyType(
    {
        "bool": 1,
        "int8": 1,
        "uint8": 1,
        "int16": 2,
        "uint16": 2,
        "bfloat16": 2,
        "float16": 2,
        "int32": 4,
        "uint32": 4,
        "float32": 4,
        "complex64": 8,
        "int64": 8,
        "uint64": 8,
        "float64": 8,
        "complex128": 16,
    }
)
_STORAGE_KINDS: Final = frozenset(("owned_array", "array_view"))
_FORBIDDEN_FIELD_TOKENS: Final = ("reward", "score", "return", "rank", "outcome")
_RTU_CARRY_ROOTS: Final = frozenset(("actor_rtu_state", "critic_rtu_state"))
_RTU_SENSITIVITY_ROOTS: Final = frozenset(
    (
        "actor_sensitivities",
        "critic_sensitivities",
        "actor_taylor_trace",
        "critic_taylor_trace",
    )
)
_ELIGIBILITY_ROOTS: Final = frozenset(("actor_traces", "critic_traces"))
_LEGACY_RTU_SENSITIVITY_COMPONENTS: Final = frozenset(
    ("memory_grad", "sensitivities", "taylor_trace")
)
_ELIGIBILITY_COMPONENTS: Final = frozenset(
    (
        "eligibility_trace",
        "eligibility_traces",
        "bias_eligibility_trace",
        "eligibility_tree",
    )
)
_IDENTIFIER_RE: Final = re.compile(r"[a-z][a-z0-9_]*\Z")
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")


class AlgorithmicResourceMeasurementError(ValueError):
    """Raised when an inert structural measurement fails closed."""


def _fail(message: str) -> NoReturn:
    raise AlgorithmicResourceMeasurementError(message)


def _contains_forbidden_token(value: str) -> bool:
    lowered = value.lower()
    return any(token in lowered for token in _FORBIDDEN_FIELD_TOKENS)


def _require_safe_text(value: object, label: str, *, maximum: int = MAX_TEXT_LENGTH) -> str:
    if type(value) is not str or not value or len(value) > maximum:
        _fail(f"{label} must be nonempty bounded exact text")
    if not value.isascii() or any(ord(character) < 32 for character in value):
        _fail(f"{label} must be printable ASCII text")
    if _contains_forbidden_token(value):
        _fail(f"{label} contains a forbidden field token")
    return value


def _require_identifier(value: object, label: str) -> str:
    exact = _require_safe_text(value, label, maximum=128)
    if _IDENTIFIER_RE.fullmatch(exact) is None:
        _fail(f"{label} must be a canonical lower-case identifier")
    return exact


def _require_sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        _fail(f"{label} must be an exact lower-case SHA-256 digest")
    return value


def _require_int(
    value: object,
    label: str,
    *,
    minimum: int = 0,
    maximum: int = MAX_INTEGER,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail(f"{label} must be an exact bounded integer")
    return value


def _checked_add(left: int, right: int, label: str) -> int:
    result = left + right
    if result > MAX_INTEGER:
        _fail(f"{label} exceeds the exact integer bound")
    return result


def _checked_product(values: tuple[int, ...], label: str) -> int:
    result = 1
    for value in values:
        if value != 0 and result > MAX_INTEGER // value:
            _fail(f"{label} exceeds the exact integer bound")
        result *= value
    return result


def _reject_forbidden_field_names(value: object, label: str = "payload") -> None:
    stack: list[tuple[object, int]] = [(value, 0)]
    nodes = 0
    seen: set[int] = set()
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > MAX_JSON_NODES:
            _fail(f"{label} has too many JSON nodes")
        if depth > MAX_JSON_DEPTH:
            _fail(f"{label} exceeds the JSON depth bound")
        if type(current) is dict:
            identity = id(current)
            if identity in seen:
                _fail(f"{label} contains an aliased or cyclic object")
            seen.add(identity)
            for key, item in current.items():
                if type(key) is not str:
                    _fail(f"{label} contains a non-text field name")
                if _contains_forbidden_token(key):
                    _fail(f"{label} contains a forbidden field name")
                stack.append((item, depth + 1))
        elif type(current) is list:
            identity = id(current)
            if identity in seen:
                _fail(f"{label} contains an aliased or cyclic array")
            seen.add(identity)
            stack.extend((item, depth + 1) for item in current)


def _plain_json(value: object, label: str, *, depth: int = 0) -> Any:
    if depth > MAX_JSON_DEPTH:
        _fail(f"{label} exceeds the JSON depth bound")
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        return _require_int(value, label, minimum=-MAX_INTEGER)
    if type(value) is float:
        if not math.isfinite(value):
            _fail(f"{label} contains a non-finite float")
        return value
    if type(value) is str:
        if len(value) > MAX_TEXT_LENGTH:
            _fail(f"{label} text exceeds the bound")
        if not value.isascii():
            _fail(f"{label} text must be ASCII")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str or not key or len(key) > 128:
                _fail(f"{label} field names must be bounded exact text")
            if _contains_forbidden_token(key):
                _fail(f"{label} contains a forbidden field name")
            result[key] = _plain_json(item, f"{label}.{key}", depth=depth + 1)
        _reject_forbidden_field_names(result, label)
        return result
    if type(value) in {list, tuple}:
        items = cast(list[object] | tuple[object, ...], value)
        return [
            _plain_json(item, f"{label}[{index}]", depth=depth + 1)
            for index, item in enumerate(items)
        ]
    _fail(f"{label} contains a non-JSON value")


def _freeze_json(value: Any) -> Any:
    if type(value) is dict:
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_thaw_json(item) for item in value]
    return copy.deepcopy(value)


def _exact_json_equal(left: object, right: object) -> bool:
    """Compare plain JSON recursively without Python's bool/int coercions."""

    if type(left) is not type(right):
        return False
    if type(left) is dict:
        left_dict = cast(dict[str, Any], left)
        right_dict = cast(dict[str, Any], right)
        return left_dict.keys() == right_dict.keys() and all(
            _exact_json_equal(left_dict[key], right_dict[key]) for key in left_dict
        )
    if type(left) is list:
        left_list = left
        right_list = cast(list[Any], right)
        return len(left_list) == len(right_list) and all(
            _exact_json_equal(left_item, right_item)
            for left_item, right_item in zip(left_list, right_list, strict=True)
        )
    return bool(left == right)


def _canonical_json(value: object, *, maximum: int = MAX_MEASUREMENT_BASIS_BYTES) -> bytes:
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise AlgorithmicResourceMeasurementError("value is not canonical JSON") from exc
    if len(raw) + 1 > maximum:
        _fail("canonical artifact exceeds its byte bound")
    return raw + b"\n"


def _body_sha256(value: Mapping[str, Any]) -> str:
    raw = _canonical_json(dict(value))[:-1]
    return hashlib.sha256(raw).hexdigest()


def _false_capabilities() -> dict[str, bool]:
    return {
        "execution_performed": False,
        "family_producer_available": False,
        "production_receipt_available": False,
        "runner_invoked": False,
        "runtime_qualified": False,
    }


def _false_authority() -> dict[str, bool]:
    return {
        "execution_authorized": False,
        "issuance_performed": False,
        "publication_authority_granted": False,
        "qualification_granted": False,
    }


def _false_claims() -> dict[str, bool]:
    return {
        "performance_claim_allowed": False,
        "promotion_allowed": False,
        "resource_matched": False,
        "scientific_evidence_created": False,
        "source_qualified": False,
    }


def _false_readiness() -> dict[str, bool]:
    return {
        "descriptor_audited": False,
        "producer_identity_pinned": False,
        "qualification_ready": False,
        "source_audited": False,
    }


def _canonical_leaf_path(components: tuple[str, ...]) -> str:
    return "/" + "/".join(components)


def classify_rtu_leaf_path(
    leaf_path: tuple[str, ...],
) -> Literal["recurrent_carry", "rtrl_sensitivity", "eligibility_trace"]:
    """Classify one exact local RTU or eligibility-tree leaf path."""

    if type(leaf_path) is not tuple or not leaf_path:
        _fail("RTU leaf path must be a nonempty exact tuple")
    components = tuple(_require_identifier(item, "RTU leaf path component") for item in leaf_path)
    if components[0] in _RTU_SENSITIVITY_ROOTS or any(
        component in _LEGACY_RTU_SENSITIVITY_COMPONENTS for component in components
    ):
        return "rtrl_sensitivity"
    if components[0] in _RTU_CARRY_ROOTS:
        if len(components) == 2 and components[1] in {"real", "imaginary"}:
            return "recurrent_carry"
        _fail("local RTU state path must select its exact real or imaginary leaf")
    if components[0] == "hstate" and components[-1] in {"real", "imaginary"}:
        return "recurrent_carry"
    if components[0] in _ELIGIBILITY_ROOTS or any(
        component in _ELIGIBILITY_COMPONENTS for component in components
    ):
        return "eligibility_trace"
    _fail("leaf path has no exact RTU carry, sensitivity, or eligibility classification")


@dataclass(frozen=True, slots=True)
class ArrayLeafObservationV1:
    """One explicitly classified logical array leaf."""

    leaf_path: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    owner: str
    category: ArrayCategory
    lifecycle: str
    alias_id: str
    storage_id: str
    storage_kind: StorageKind = "owned_array"

    def __post_init__(self) -> None:
        if type(self.leaf_path) is not tuple or not 1 <= len(self.leaf_path) <= 32:
            _fail("leaf path must be a nonempty bounded exact tuple")
        for component in self.leaf_path:
            _require_identifier(component, "leaf path component")
        if type(self.shape) is not tuple or len(self.shape) > MAX_ARRAY_RANK:
            _fail("array shape must be an exact bounded tuple")
        exact_shape = tuple(
            _require_int(dimension, "array dimension", maximum=MAX_ARRAY_DIMENSION)
            for dimension in self.shape
        )
        _checked_product(exact_shape, "array element count")
        if type(self.dtype) is not str or self.dtype not in _DTYPE_ITEMSIZE:
            _fail("array dtype is not in the exact logical byte table")
        _require_identifier(self.owner, "array owner")
        if self.category not in ARRAY_CATEGORIES:
            _fail("array category differs from the frozen inventory")
        _require_identifier(self.lifecycle, "array lifecycle")
        _require_identifier(self.alias_id, "array alias ID")
        _require_identifier(self.storage_id, "array storage ID")
        if self.storage_kind not in _STORAGE_KINDS:
            _fail("array storage kind differs from the frozen inventory")

        path_tokens = set(self.leaf_path)
        if "target_params" in path_tokens and self.category != "target_copy":
            _fail("target parameters belong only to the target-copy category")
        if "globally_trainable_stop_gradient" in self.owner and (
            self.category != "trainable_parameters"
        ):
            _fail("globally trainable stop-gradient arrays remain trainable parameters")
        if (
            self.leaf_path[0] == "hstate"
            or self.leaf_path[0] in _RTU_CARRY_ROOTS
            or self.leaf_path[0] in _RTU_SENSITIVITY_ROOTS
            or self.leaf_path[0] in _ELIGIBILITY_ROOTS
            or path_tokens.intersection(_LEGACY_RTU_SENSITIVITY_COMPONENTS)
            or path_tokens.intersection(_ELIGIBILITY_COMPONENTS)
        ):
            expected = classify_rtu_leaf_path(self.leaf_path)
            if self.category != expected:
                _fail("state leaf category differs from its exact path classification")

    @property
    def canonical_leaf_path(self) -> str:
        return _canonical_leaf_path(self.leaf_path)

    @property
    def elements(self) -> int:
        return _checked_product(self.shape, "array element count")

    @property
    def itemsize_bytes(self) -> int:
        return _DTYPE_ITEMSIZE[self.dtype]

    @property
    def logical_owned_bytes(self) -> int:
        if self.storage_kind == "array_view":
            return 0
        return _checked_product(
            (self.elements, self.itemsize_bytes),
            "logical owned-array byte count",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "alias_id": self.alias_id,
            "canonical_leaf_path": self.canonical_leaf_path,
            "category": self.category,
            "dtype": self.dtype,
            "elements": self.elements,
            "itemsize_bytes": self.itemsize_bytes,
            "lifecycle": self.lifecycle,
            "logical_owned_bytes": self.logical_owned_bytes,
            "owner": self.owner,
            "shape": list(self.shape),
            "storage_id": self.storage_id,
            "storage_kind": self.storage_kind,
        }


@dataclass(frozen=True, slots=True)
class ReplayCapacityObservationV1:
    """One simultaneously-live replay subsystem capacity."""

    subsystem_id: str
    capacity_transitions: int

    def __post_init__(self) -> None:
        _require_identifier(self.subsystem_id, "replay subsystem ID")
        _require_int(
            self.capacity_transitions,
            "replay capacity",
            minimum=1,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "capacity_transitions": self.capacity_transitions,
            "subsystem_id": self.subsystem_id,
        }


@dataclass(frozen=True, slots=True)
class CategorySnapshotV1:
    """One complete simultaneously-live snapshot for one semantic category."""

    snapshot_id: str
    category: ArrayCategory
    lifecycle: str
    leaves: tuple[ArrayLeafObservationV1, ...]
    replay_capacities: tuple[ReplayCapacityObservationV1, ...] = ()

    def __post_init__(self) -> None:
        _require_identifier(self.snapshot_id, "category snapshot ID")
        if self.category not in ARRAY_CATEGORIES:
            _fail("snapshot category differs from the frozen inventory")
        _require_identifier(self.lifecycle, "snapshot lifecycle")
        if type(self.leaves) is not tuple or not 1 <= len(self.leaves) <= MAX_LEAVES_PER_SNAPSHOT:
            _fail("snapshot leaves must be a nonempty bounded exact tuple")
        if type(self.replay_capacities) is not tuple:
            _fail("replay capacities must be an exact tuple")
        if self.category == "replay_storage":
            if not self.replay_capacities:
                _fail("replay snapshot requires exact simultaneous capacities")
        elif self.replay_capacities:
            _fail("non-replay snapshot cannot declare replay capacity")

        paths: set[str] = set()
        aliases: set[str] = set()
        storage_groups: dict[str, list[ArrayLeafObservationV1]] = {}
        for leaf in self.leaves:
            if type(leaf) is not ArrayLeafObservationV1:
                _fail("snapshot leaf type differs")
            if leaf.category != self.category:
                _fail("snapshot and leaf category differ")
            if leaf.lifecycle != self.lifecycle:
                _fail("snapshot and leaf lifecycle differ")
            if leaf.canonical_leaf_path in paths:
                _fail("snapshot leaf paths must be unique")
            if leaf.alias_id in aliases:
                _fail("snapshot alias IDs must be unique")
            paths.add(leaf.canonical_leaf_path)
            aliases.add(leaf.alias_id)
            storage_groups.setdefault(leaf.storage_id, []).append(leaf)

        for storage_id, group in storage_groups.items():
            owned = [leaf for leaf in group if leaf.storage_kind == "owned_array"]
            if not owned:
                _fail(f"array view storage {storage_id} has no owned base in its category snapshot")
            first = owned[0]
            for leaf in owned[1:]:
                if leaf.shape != first.shape or leaf.dtype != first.dtype:
                    _fail("one storage ID has inconsistent owned-array shape or dtype")
            for leaf in group:
                if leaf.storage_kind == "array_view" and (
                    leaf.dtype != first.dtype or leaf.elements > first.elements
                ):
                    _fail("array view differs from or exceeds its owned base storage")

        subsystem_ids: set[str] = set()
        total_capacity = 0
        for capacity in self.replay_capacities:
            if type(capacity) is not ReplayCapacityObservationV1:
                _fail("replay capacity observation type differs")
            if capacity.subsystem_id in subsystem_ids:
                _fail("replay subsystem capacity is duplicated")
            subsystem_ids.add(capacity.subsystem_id)
            total_capacity = _checked_add(
                total_capacity,
                capacity.capacity_transitions,
                "simultaneously-live replay capacity",
            )

    @property
    def _owned_by_storage(self) -> dict[str, ArrayLeafObservationV1]:
        result: dict[str, ArrayLeafObservationV1] = {}
        for leaf in sorted(self.leaves, key=lambda item: item.canonical_leaf_path):
            if leaf.storage_kind == "owned_array" and leaf.storage_id not in result:
                result[leaf.storage_id] = leaf
        return result

    @property
    def elements(self) -> int:
        result = 0
        for leaf in self._owned_by_storage.values():
            result = _checked_add(result, leaf.elements, "snapshot element count")
        return result

    @property
    def logical_owned_bytes(self) -> int:
        result = 0
        for leaf in self._owned_by_storage.values():
            result = _checked_add(
                result,
                leaf.logical_owned_bytes,
                "snapshot logical owned-array bytes",
            )
        return result

    @property
    def replay_capacity_transitions(self) -> int:
        result = 0
        for capacity in self.replay_capacities:
            result = _checked_add(
                result,
                capacity.capacity_transitions,
                "snapshot replay capacity",
            )
        return result

    @property
    def storage_aliases(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for leaf in self.leaves:
            result.setdefault(leaf.storage_id, []).append(leaf.alias_id)
        return {storage: sorted(alias_ids) for storage, alias_ids in sorted(result.items())}

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "elements": self.elements,
            "leaves": [
                leaf.to_dict()
                for leaf in sorted(self.leaves, key=lambda item: item.canonical_leaf_path)
            ],
            "lifecycle": self.lifecycle,
            "logical_owned_bytes": self.logical_owned_bytes,
            "replay_capacities": [
                item.to_dict()
                for item in sorted(
                    self.replay_capacities,
                    key=lambda item: item.subsystem_id,
                )
            ],
            "replay_capacity_transitions": self.replay_capacity_transitions,
            "snapshot_id": self.snapshot_id,
            "storage_aliases": self.storage_aliases,
        }


@dataclass(frozen=True, slots=True)
class StructuralAbsenceProofV1:
    """One exact runtime structural-absence assertion."""

    subject: str
    absence_kind: str
    proof_id: str
    evidence_kind: str
    details: Mapping[str, Any]

    def __post_init__(self) -> None:
        if type(self.subject) is not str or self.subject not in ABSENCE_KIND_BY_SUBJECT:
            _fail("structural absence subject differs from the frozen inventory")
        if self.absence_kind != ABSENCE_KIND_BY_SUBJECT[self.subject]:
            _fail("structural absence kind differs from its subject")
        _require_identifier(self.proof_id, "structural absence proof ID")
        if self.evidence_kind != "exact_runtime_subsystem_inventory":
            _fail("structural absence evidence kind must be exact runtime inventory")
        plain = _plain_json(self.details, "structural absence proof details")
        if type(plain) is not dict or not plain:
            _fail("structural absence proof details must be a nonempty object")
        object.__setattr__(self, "details", cast(Mapping[str, Any], _freeze_json(plain)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "absence_kind": self.absence_kind,
            "details": _thaw_json(self.details),
            "evidence_kind": self.evidence_kind,
            "proof_id": self.proof_id,
            "subject": self.subject,
        }


@dataclass(frozen=True, slots=True)
class FieldMeasurementV1:
    """One measured field before a later receipt binds the basis digest."""

    field_name: str
    observed_value: int
    measurement_kind: str
    measurement_scope: str
    structural_absence_kind: str
    absence_proof_id: str | None

    def __post_init__(self) -> None:
        if self.field_name not in resource_contract.ALGORITHMIC_RESOURCE_FIELDS:
            _fail("measurement field differs from the exact first-20 inventory")
        value = _require_int(self.observed_value, f"measurement {self.field_name}")
        policy = resource_contract.matched_v3_algorithmic_resource_field_policy()[
            resource_contract.ALGORITHMIC_RESOURCE_FIELDS.index(self.field_name)
        ]
        if self.measurement_scope != policy.measurement_scope:
            _fail(f"measurement scope differs for {self.field_name}")
        if value > 0:
            if (
                self.measurement_kind != policy.positive_measurement_kind
                or self.structural_absence_kind != resource_contract.NOT_ABSENT
                or self.absence_proof_id is not None
            ):
                _fail(f"positive measurement contract differs for {self.field_name}")
        elif policy.zero_structural_absence_kind == resource_contract.ZERO_FORBIDDEN:
            _fail(f"zero is forbidden for {self.field_name}")
        elif (
            self.measurement_kind != resource_contract.STRUCTURAL_ABSENCE
            or self.structural_absence_kind != policy.zero_structural_absence_kind
            or self.absence_proof_id is None
        ):
            _fail(f"zero measurement lacks an exact structural absence for {self.field_name}")
        if self.absence_proof_id is not None:
            _require_identifier(self.absence_proof_id, "measurement absence proof ID")

    def to_dict(self) -> dict[str, Any]:
        return {
            "absence_proof_id": self.absence_proof_id,
            "field_name": self.field_name,
            "measurement_kind": self.measurement_kind,
            "measurement_scope": self.measurement_scope,
            "observed_value": self.observed_value,
            "structural_absence_kind": self.structural_absence_kind,
        }


def _extend_chain(previous_sha256: str, value: Mapping[str, Any]) -> str:
    _require_sha256(previous_sha256, "observation-chain predecessor")
    event_sha256 = hashlib.sha256(_canonical_json(dict(value))[:-1]).digest()
    return hashlib.sha256(bytes.fromhex(previous_sha256) + event_sha256).hexdigest()


class _CategoryAccumulator:
    """Bounded retained high-water state for one category."""

    __slots__ = (
        "byte_snapshot",
        "capacity_snapshot",
        "category",
        "chain_sha256",
        "element_snapshot",
        "max_bytes",
        "max_capacity",
        "max_elements",
        "observation_count",
    )

    def __init__(self, category: ArrayCategory) -> None:
        self.category = category
        self.observation_count = 0
        self.chain_sha256 = "0" * 64
        self.max_elements = -1
        self.max_bytes = -1
        self.max_capacity = -1
        self.element_snapshot: dict[str, Any] | None = None
        self.byte_snapshot: dict[str, Any] | None = None
        self.capacity_snapshot: dict[str, Any] | None = None

    def observe(self, snapshot: CategorySnapshotV1) -> None:
        serialized = snapshot.to_dict()
        next_count = _checked_add(
            self.observation_count,
            1,
            "category snapshot observation count",
        )
        next_chain_sha256 = _extend_chain(self.chain_sha256, serialized)
        self.observation_count = next_count
        self.chain_sha256 = next_chain_sha256
        if snapshot.elements > self.max_elements:
            self.max_elements = snapshot.elements
            self.element_snapshot = serialized
        if snapshot.logical_owned_bytes > self.max_bytes:
            self.max_bytes = snapshot.logical_owned_bytes
            self.byte_snapshot = serialized
        if (
            self.category == "replay_storage"
            and snapshot.replay_capacity_transitions > self.max_capacity
        ):
            self.max_capacity = snapshot.replay_capacity_transitions
            self.capacity_snapshot = serialized

    @property
    def observed(self) -> bool:
        return self.observation_count > 0

    @property
    def exact_max_elements(self) -> int:
        return max(self.max_elements, 0)

    @property
    def exact_max_bytes(self) -> int:
        return max(self.max_bytes, 0)

    @property
    def exact_max_capacity(self) -> int:
        return max(self.max_capacity, 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "byte_snapshot": copy.deepcopy(self.byte_snapshot),
            "capacity_snapshot": copy.deepcopy(self.capacity_snapshot),
            "category": self.category,
            "element_snapshot": copy.deepcopy(self.element_snapshot),
            "max_elements": self.exact_max_elements,
            "max_logical_owned_bytes": self.exact_max_bytes,
            "max_replay_capacity_transitions": self.exact_max_capacity,
            "observation_chain_sha256": self.chain_sha256,
            "observation_count": self.observation_count,
        }


_BASIS_LIMITATIONS: Final = (
    "no_runner_or_candidate_execution",
    "no_family_producer_identity",
    "no_physical_allocator_measurement",
    "no_resource_match_decision",
    "no_scientific_evidence_status",
)


@dataclass(frozen=True, slots=True)
class AlgorithmicResourceMeasurementBasisV1:
    """Sealed, canonical, non-authorizing structural measurement basis."""

    ledger_id: str
    ledger_descriptor_sha256: str
    ledger_source_sha256: str
    fields: tuple[FieldMeasurementV1, ...]
    category_high_water: Mapping[str, Any]
    structural_absence_proofs: tuple[StructuralAbsenceProofV1, ...]
    observation_chain: Mapping[str, Any]
    transaction_accounting: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_identifier(self.ledger_id, "measurement ledger ID")
        _require_sha256(self.ledger_descriptor_sha256, "ledger descriptor")
        if _require_sha256(self.ledger_source_sha256, "ledger source") == "0" * 64:
            _fail("ledger source must be supplied by an upstream audited identity")
        if type(self.fields) is not tuple or any(
            type(item) is not FieldMeasurementV1 for item in self.fields
        ):
            _fail("measurement basis field type differs")
        if tuple(item.field_name for item in self.fields) != (
            resource_contract.ALGORITHMIC_RESOURCE_FIELDS
        ):
            _fail("measurement basis fields differ from the exact first-20 order")
        if type(self.structural_absence_proofs) is not tuple or any(
            type(item) is not StructuralAbsenceProofV1 for item in self.structural_absence_proofs
        ):
            _fail("measurement basis absence proof type differs")
        for name in (
            "category_high_water",
            "observation_chain",
            "transaction_accounting",
        ):
            plain = _plain_json(getattr(self, name), f"measurement basis {name}")
            if type(plain) is not dict:
                _fail(f"measurement basis {name} must be an exact object")
            object.__setattr__(self, name, _freeze_json(plain))
        _validate_parsed_basis(
            self.to_body_dict(),
            expected_measurement_source_sha256=self.ledger_source_sha256,
        )
        emitted = self.to_dict()
        _reject_forbidden_field_names(emitted, "complete emitted measurement basis")
        _canonical_json(emitted)

    def to_body_dict(self) -> dict[str, Any]:
        fields = [item.to_dict() for item in self.fields]
        return {
            "authority": _false_authority(),
            "capabilities": _false_capabilities(),
            "category_high_water": _thaw_json(self.category_high_water),
            "claims": _false_claims(),
            "classification": MEASUREMENT_CLASSIFICATION,
            "field_inventory_sha256": hashlib.sha256(
                _canonical_json({"fields": fields})[:-1]
            ).hexdigest(),
            "fields": fields,
            "ledger_descriptor_sha256": self.ledger_descriptor_sha256,
            "ledger_id": self.ledger_id,
            "ledger_source_sha256": self.ledger_source_sha256,
            "limitations": list(_BASIS_LIMITATIONS),
            "observation_chain": _thaw_json(self.observation_chain),
            "readiness": _false_readiness(),
            "schema_version": MEASUREMENT_BASIS_SCHEMA_VERSION,
            "status": MEASUREMENT_BASIS_STATUS,
            "structural_absence_proofs": [
                item.to_dict() for item in self.structural_absence_proofs
            ],
            "transaction_accounting": _thaw_json(self.transaction_accounting),
        }

    @property
    def body_sha256(self) -> str:
        return _body_sha256(self.to_body_dict())

    def to_dict(self) -> dict[str, Any]:
        result = self.to_body_dict()
        result["measurement_basis_body_sha256"] = self.body_sha256
        return result


@dataclass(frozen=True, slots=True, eq=False)
class _LedgerTransaction:
    """Single-use frozen handle; its mirrored fields are never authoritative."""

    _ledger: AlgorithmicResourceMeasurementLedger
    _transaction_id: str
    _kind: Literal["environment", "learning"]
    _adaptation_kind: str | None = None
    _optimizer_applied: bool = False
    _gradient_applied: bool = False
    _sample_contributions: int = 0

    def commit(self) -> None:
        self._ledger._complete_transaction(self, committed=True)

    def abort(self) -> None:
        self._ledger._complete_transaction(self, committed=False)


@dataclass(frozen=True, slots=True)
class _LedgerTransactionRecord:
    """Authoritative immutable transaction payload retained only by the ledger."""

    transaction_id: str
    kind: Literal["environment", "learning"]
    adaptation_kind: str | None
    optimizer_applied: bool
    gradient_applied: bool
    sample_contributions: int

    def event_dict(self, *, committed: bool) -> dict[str, Any]:
        result: dict[str, Any] = {
            "committed": committed,
            "transaction_id": self.transaction_id,
            "transaction_kind": self.kind,
        }
        if self.kind == "learning":
            result.update(
                {
                    "adaptation_kind": self.adaptation_kind,
                    "gradient_applied": self.gradient_applied,
                    "optimizer_applied": self.optimizer_applied,
                    "sample_contributions": self.sample_contributions,
                }
            )
        return result


class AlgorithmicResourceMeasurementLedger:
    """Mutable explicit ledger that becomes immutable after finalization."""

    def __init__(self, ledger_id: str, *, measurement_source_sha256: str) -> None:
        self._ledger_id = _require_identifier(ledger_id, "measurement ledger ID")
        self._measurement_source_sha256 = _require_sha256(
            measurement_source_sha256,
            "externally supplied measurement source",
        )
        if self._measurement_source_sha256 == "0" * 64:
            _fail("externally supplied measurement source cannot be the zero digest")
        self._state: Literal["active", "aborted", "sealed"] = "active"
        self._transaction_status: dict[str, Literal["open", "committed", "aborted"]] = {}
        self._open_transaction_handles: dict[str, _LedgerTransaction] = {}
        self._open_transaction_records: dict[str, _LedgerTransactionRecord] = {}
        self._environment_interactions = 0
        self._optimizer_updates = 0
        self._gradient_updates = 0
        self._sample_updates = 0
        self._committed_learning_transactions = 0
        self._aborted_transactions = 0
        self._transaction_event_count = 0
        self._transaction_chain_sha256 = "0" * 64
        self._snapshot_ids: set[str] = set()
        self._snapshot_count = 0
        self._snapshot_chain_sha256 = "0" * 64
        self._category = {category: _CategoryAccumulator(category) for category in ARRAY_CATEGORIES}
        self._absence_by_subject: dict[str, StructuralAbsenceProofV1] = {}
        self._proof_ids: set[str] = set()
        self._sealed_basis: AlgorithmicResourceMeasurementBasisV1 | None = None
        self._abort_reason: str | None = None

    def _require_active(self) -> None:
        if self._state == "aborted":
            _fail("measurement ledger is aborted")
        if self._state == "sealed":
            _fail("measurement ledger is sealed")

    def _register_transaction(
        self,
        transaction_id: str,
        kind: Literal["environment", "learning"],
        *,
        adaptation_kind: str | None = None,
        optimizer_applied: bool = False,
        gradient_applied: bool = False,
        sample_contributions: int = 0,
    ) -> _LedgerTransaction:
        self._require_active()
        exact_id = _require_identifier(transaction_id, "transaction ID")
        if exact_id in self._transaction_status:
            _fail("transaction ID was already consumed")
        transaction = _LedgerTransaction(
            self,
            exact_id,
            kind,
            _adaptation_kind=adaptation_kind,
            _optimizer_applied=optimizer_applied,
            _gradient_applied=gradient_applied,
            _sample_contributions=sample_contributions,
        )
        self._transaction_status[exact_id] = "open"
        self._open_transaction_handles[exact_id] = transaction
        self._open_transaction_records[exact_id] = _LedgerTransactionRecord(
            transaction_id=exact_id,
            kind=kind,
            adaptation_kind=adaptation_kind,
            optimizer_applied=optimizer_applied,
            gradient_applied=gradient_applied,
            sample_contributions=sample_contributions,
        )
        return transaction

    def begin_environment_transaction(self, transaction_id: str) -> _LedgerTransaction:
        """Open one environment transition; only commit increments the field."""

        return self._register_transaction(transaction_id, "environment")

    def begin_learning_transaction(
        self,
        transaction_id: str,
        *,
        adaptation_kind: str,
        optimizer_applied: bool,
        gradient_applied: bool,
        sample_contributions: int,
    ) -> _LedgerTransaction:
        """Open one explicit learning/adaptation data-consumption transaction."""

        exact_adaptation = _require_identifier(adaptation_kind, "adaptation kind")
        if type(optimizer_applied) is not bool or type(gradient_applied) is not bool:
            _fail("learning transaction flags must be exact booleans")
        if gradient_applied and not optimizer_applied:
            _fail("a gradient application transaction must apply an optimizer update")
        exact_samples = _require_int(
            sample_contributions,
            "learning sample contributions",
            minimum=1,
        )
        return self._register_transaction(
            transaction_id,
            "learning",
            adaptation_kind=exact_adaptation,
            optimizer_applied=optimizer_applied,
            gradient_applied=gradient_applied,
            sample_contributions=exact_samples,
        )

    def _complete_transaction(
        self,
        transaction: _LedgerTransaction,
        *,
        committed: bool,
    ) -> None:
        self._require_active()
        if type(transaction) is not _LedgerTransaction or transaction._ledger is not self:
            _fail("transaction belongs to a different measurement ledger")
        status = self._transaction_status.get(transaction._transaction_id)
        if (
            status != "open"
            or self._open_transaction_handles.get(transaction._transaction_id) is not transaction
        ):
            _fail("transaction is not open and cannot be counted again")
        if type(committed) is not bool:
            _fail("transaction terminal state must be an exact boolean")
        record = self._open_transaction_records.get(transaction._transaction_id)
        if record is None:
            _fail("transaction has no ledger-owned immutable record")
        if committed and record.kind == "learning":
            positive_subjects = ["sample_updates"]
            if record.optimizer_applied:
                positive_subjects.append("optimizer_updates")
            if record.gradient_applied:
                positive_subjects.append("gradient_updates")
            if any(subject in self._absence_by_subject for subject in positive_subjects):
                _fail("committed learning transaction contradicts structural absence")

        next_event_count = _checked_add(
            self._transaction_event_count,
            1,
            "transaction event count",
        )
        next_chain_sha256 = _extend_chain(
            self._transaction_chain_sha256,
            record.event_dict(committed=committed),
        )
        next_aborted = self._aborted_transactions
        next_environment = self._environment_interactions
        next_learning = self._committed_learning_transactions
        next_samples = self._sample_updates
        next_optimizer = self._optimizer_updates
        next_gradient = self._gradient_updates
        if not committed:
            next_aborted = _checked_add(
                self._aborted_transactions,
                1,
                "aborted transaction count",
            )
        elif record.kind == "environment":
            next_environment = _checked_add(
                self._environment_interactions,
                1,
                "environment interaction count",
            )
        else:
            next_learning = _checked_add(
                self._committed_learning_transactions,
                1,
                "committed learning transaction count",
            )
            next_samples = _checked_add(
                self._sample_updates,
                record.sample_contributions,
                "learning sample contribution count",
            )
            if record.optimizer_applied:
                next_optimizer = _checked_add(
                    self._optimizer_updates,
                    1,
                    "optimizer update count",
                )
            if record.gradient_applied:
                next_gradient = _checked_add(
                    self._gradient_updates,
                    1,
                    "gradient update count",
                )

        new_status: Literal["committed", "aborted"] = "committed" if committed else "aborted"
        self._transaction_status[transaction._transaction_id] = new_status
        del self._open_transaction_handles[transaction._transaction_id]
        del self._open_transaction_records[transaction._transaction_id]
        self._transaction_event_count = next_event_count
        self._transaction_chain_sha256 = next_chain_sha256
        self._aborted_transactions = next_aborted
        self._environment_interactions = next_environment
        self._committed_learning_transactions = next_learning
        self._sample_updates = next_samples
        self._optimizer_updates = next_optimizer
        self._gradient_updates = next_gradient

    def observe_category_snapshot(self, snapshot: CategorySnapshotV1) -> None:
        """Record one explicitly complete simultaneous-live lifecycle snapshot."""

        self._require_active()
        if type(snapshot) is not CategorySnapshotV1:
            _fail("category snapshot type differs")
        if snapshot.snapshot_id in self._snapshot_ids:
            _fail("category snapshot ID was already consumed")
        if snapshot.category in self._absence_by_subject:
            _fail("category snapshot contradicts structural absence")
        serialized = snapshot.to_dict()
        next_snapshot_count = _checked_add(
            self._snapshot_count,
            1,
            "category snapshot count",
        )
        next_snapshot_chain = _extend_chain(
            self._snapshot_chain_sha256,
            serialized,
        )
        self._category[snapshot.category].observe(snapshot)
        self._snapshot_ids.add(snapshot.snapshot_id)
        self._snapshot_count = next_snapshot_count
        self._snapshot_chain_sha256 = next_snapshot_chain

    def declare_structural_absence(self, proof: StructuralAbsenceProofV1) -> None:
        """Declare one observed runtime subsystem absence, never a config zero."""

        self._require_active()
        if type(proof) is not StructuralAbsenceProofV1:
            _fail("structural absence proof type differs")
        if proof.subject in self._absence_by_subject or proof.proof_id in self._proof_ids:
            _fail("structural absence proof subject or ID was already consumed")
        positive_counter = {
            "optimizer_updates": self._optimizer_updates,
            "gradient_updates": self._gradient_updates,
            "sample_updates": self._sample_updates,
        }.get(proof.subject)
        if positive_counter is not None and positive_counter > 0:
            _fail("structural absence contradicts a committed positive counter")
        if proof.subject in self._category and self._category[proof.subject].observed:
            _fail("structural absence contradicts an observed category snapshot")
        self._absence_by_subject[proof.subject] = proof
        self._proof_ids.add(proof.proof_id)

    def abort(self, reason: str) -> None:
        """Irreversibly abort the whole ledger without producing a basis."""

        self._require_active()
        self._abort_reason = _require_identifier(reason, "measurement ledger abort reason")
        self._state = "aborted"

    def _field_values(self) -> dict[str, int]:
        category = self._category
        return {
            "max_environment_interactions": self._environment_interactions,
            "max_optimizer_updates": self._optimizer_updates,
            "max_gradient_updates": self._gradient_updates,
            "max_sample_updates": self._sample_updates,
            "max_trainable_parameters": category["trainable_parameters"].exact_max_elements,
            "max_frozen_parameters": category["frozen_parameters"].exact_max_elements,
            "max_optimizer_state_elements": category["optimizer_state"].exact_max_elements,
            "max_optimizer_state_bytes": category["optimizer_state"].exact_max_bytes,
            "max_target_copy_elements": category["target_copy"].exact_max_elements,
            "max_target_copy_bytes": category["target_copy"].exact_max_bytes,
            "max_replay_capacity_transitions": category["replay_storage"].exact_max_capacity,
            "max_replay_peak_bytes": category["replay_storage"].exact_max_bytes,
            "max_rollout_storage_elements": category["rollout_storage"].exact_max_elements,
            "max_rollout_peak_bytes": category["rollout_storage"].exact_max_bytes,
            "max_recurrent_carry_elements": category["recurrent_carry"].exact_max_elements,
            "max_recurrent_carry_bytes": category["recurrent_carry"].exact_max_bytes,
            "max_rtrl_sensitivity_elements": category["rtrl_sensitivity"].exact_max_elements,
            "max_rtrl_sensitivity_bytes": category["rtrl_sensitivity"].exact_max_bytes,
            "max_eligibility_elements": category["eligibility_trace"].exact_max_elements,
            "max_eligibility_bytes": category["eligibility_trace"].exact_max_bytes,
        }

    def _validate_stationary_state(self, values: Mapping[str, int]) -> None:
        if any(status == "open" for status in self._transaction_status.values()):
            _fail("measurement ledger has an open transaction")
        if self._open_transaction_handles:
            _fail("measurement ledger retains an open transaction handle")
        if self._open_transaction_records:
            _fail("measurement ledger retains an open transaction record")
        if self._environment_interactions == 0:
            _fail("environment interaction measurement must be positive")
        if (self._committed_learning_transactions == 0) != (self._sample_updates == 0):
            _fail("committed learning and consumed sample zero states differ")
        if self._sample_updates < self._committed_learning_transactions:
            _fail("consumed sample count is below committed learning transactions")
        counters = {
            "optimizer_updates": self._optimizer_updates,
            "gradient_updates": self._gradient_updates,
            "sample_updates": self._sample_updates,
        }
        for subject, value in counters.items():
            has_proof = subject in self._absence_by_subject
            if (value == 0) != has_proof:
                _fail(f"counter {subject} lacks an exact structural absence proof")
        for category, accumulator in self._category.items():
            has_proof = category in self._absence_by_subject
            if accumulator.observed == has_proof:
                _fail(f"category {category} lacks an exact structural absence proof")
            if accumulator.observed and (
                accumulator.exact_max_elements == 0 or accumulator.exact_max_bytes == 0
            ):
                _fail(f"observed category {category} must have positive logical storage")
            if category == "replay_storage" and accumulator.observed:
                if accumulator.exact_max_capacity == 0:
                    _fail("observed replay category must have positive capacity")
        if self._optimizer_updates == 0 and (
            values["max_optimizer_state_elements"] != 0 or values["max_optimizer_state_bytes"] != 0
        ):
            _fail("an absent optimizer subsystem cannot retain optimizer state")
        if self._optimizer_updates > 0 and (
            self._sample_updates == 0 or values["max_trainable_parameters"] == 0
        ):
            _fail("positive optimizer updates require samples and trainable parameters")
        if self._gradient_updates > 0 and (
            self._optimizer_updates == 0
            or self._sample_updates == 0
            or values["max_trainable_parameters"] == 0
        ):
            _fail(
                "positive gradient updates require optimizer updates, samples, "
                "and trainable parameters"
            )
        for left, right in COUPLED_FIELD_PAIRS:
            if (values[left] == 0) != (values[right] == 0):
                _fail(f"coupled measurement zero/nonzero state differs for {left}")

    def _build_fields(self, values: Mapping[str, int]) -> tuple[FieldMeasurementV1, ...]:
        fields: list[FieldMeasurementV1] = []
        for policy in resource_contract.matched_v3_algorithmic_resource_field_policy():
            value = values[policy.field_name]
            if value > 0:
                measurement_kind = policy.positive_measurement_kind
                absence_kind = resource_contract.NOT_ABSENT
                proof_id = None
            else:
                subject = _FIELD_TO_SUBJECT[policy.field_name]
                proof = self._absence_by_subject[subject]
                measurement_kind = resource_contract.STRUCTURAL_ABSENCE
                absence_kind = proof.absence_kind
                proof_id = proof.proof_id
            fields.append(
                FieldMeasurementV1(
                    field_name=policy.field_name,
                    observed_value=value,
                    measurement_kind=measurement_kind,
                    measurement_scope=policy.measurement_scope,
                    structural_absence_kind=absence_kind,
                    absence_proof_id=proof_id,
                )
            )
        return tuple(fields)

    def finalize(self) -> AlgorithmicResourceMeasurementBasisV1:
        """Validate, seal, and return one idempotent canonical measurement basis."""

        if self._state == "sealed":
            if self._sealed_basis is None:
                _fail("sealed measurement ledger has no basis")
            return self._sealed_basis
        self._require_active()
        values = self._field_values()
        self._validate_stationary_state(values)
        category_high_water: dict[str, Any] = {
            category: self._category[category].to_dict() for category in ARRAY_CATEGORIES
        }
        observation_chain = {
            "category_observation_chain_sha256": self._snapshot_chain_sha256,
            "category_observation_count": self._snapshot_count,
            "chain_algorithm": "sha256_predecessor_plus_event_digest_v1",
            "initial_sha256": "0" * 64,
            "transaction_event_chain_sha256": self._transaction_chain_sha256,
            "transaction_event_count": self._transaction_event_count,
        }
        transaction_accounting = {
            "aborted_transactions": self._aborted_transactions,
            "committed_environment_transactions": self._environment_interactions,
            "committed_gradient_transactions": self._gradient_updates,
            "committed_learning_transactions": self._committed_learning_transactions,
            "committed_optimizer_transactions": self._optimizer_updates,
            "consumed_sample_contributions": self._sample_updates,
            "open_transactions": 0,
            "registered_transactions": len(self._transaction_status),
        }
        basis = AlgorithmicResourceMeasurementBasisV1(
            ledger_id=self._ledger_id,
            ledger_descriptor_sha256=(_active_measurement_ledger_descriptor_sha256()),
            ledger_source_sha256=self._measurement_source_sha256,
            fields=self._build_fields(values),
            category_high_water=category_high_water,
            structural_absence_proofs=tuple(
                self._absence_by_subject[subject]
                for subject in _ABSENCE_SUBJECTS
                if subject in self._absence_by_subject
            ),
            observation_chain=observation_chain,
            transaction_accounting=transaction_accounting,
        )
        # Force the public whole-artifact validation and serialization path
        # before the state transition, avoiding a half-sealed ledger.
        canonical_algorithmic_resource_measurement_basis_bytes(basis)
        self._sealed_basis = basis
        self._state = "sealed"
        return basis


def _measurement_descriptor_body() -> dict[str, Any]:
    field_policy = resource_contract.matched_v3_algorithmic_resource_field_policy()
    return {
        "algorithmic_resource_fields": list(resource_contract.ALGORITHMIC_RESOURCE_FIELDS),
        "authority": _false_authority(),
        "byte_policy": {
            "array_payload_bytes_included": True,
            "array_view_payload_double_counted": False,
            "device_allocator_fragmentation_included": False,
            "physical_allocator_bytes_included": False,
            "python_object_header_bytes_included": False,
            "quantity": "logical_owned_array_bytes",
        },
        "capabilities": _false_capabilities(),
        "category_inventory": list(ARRAY_CATEGORIES),
        "claims": _false_claims(),
        "classification": MEASUREMENT_CLASSIFICATION,
        "coupled_field_pairs": [list(pair) for pair in COUPLED_FIELD_PAIRS],
        "dtype_itemsize_bytes": dict(_DTYPE_ITEMSIZE),
        "field_policy": [item.to_dict() for item in field_policy],
        "forbidden_field_tokens": list(_FORBIDDEN_FIELD_TOKENS),
        "lifecycle_policy": {
            "complete_simultaneous_live_snapshot_required": True,
            "element_and_byte_high_water_retained_independently": True,
            "every_allocation_lifecycle_boundary_required": True,
            "one_snapshot_contains_one_semantic_category": True,
            "unobserved_lifecycle_inference_allowed": False,
        },
        "limits": {
            "maximum_array_dimension": MAX_ARRAY_DIMENSION,
            "maximum_array_rank": MAX_ARRAY_RANK,
            "maximum_basis_bytes": MAX_MEASUREMENT_BASIS_BYTES,
            "maximum_integer": MAX_INTEGER,
            "maximum_json_depth": MAX_JSON_DEPTH,
            "maximum_json_nodes": MAX_JSON_NODES,
            "maximum_leaves_per_snapshot": MAX_LEAVES_PER_SNAPSHOT,
            "maximum_text_length": MAX_TEXT_LENGTH,
        },
        "readiness": _false_readiness(),
        "replay_capacity_policy": {
            "capacity_quantity": "simultaneously_live_addressable_transitions",
            "multiple_live_subsystems_summed": True,
            "replay_array_storage_measured_separately": True,
        },
        "rtu_path_policy": {
            "carry_components": ["real", "imaginary"],
            "carry_root": "hstate",
            "local_carry_roots": sorted(_RTU_CARRY_ROOTS),
            "local_eligibility_roots": sorted(_ELIGIBILITY_ROOTS),
            "local_sensitivity_roots": sorted(_RTU_SENSITIVITY_ROOTS),
            "eligibility_components": sorted(_ELIGIBILITY_COMPONENTS),
            "eligibility_requires_explicit_tree": True,
            "sensitivity_markers": sorted(_LEGACY_RTU_SENSITIVITY_COMPONENTS),
        },
        "sample_contribution_policy": {
            "definition": "data_items_that_affect_learning_or_adaptation",
            "drqn_burn_in_samples_included": True,
            "ppo_joint_loss_sample_count": "once_per_sample_per_epoch",
            "pt_inner_loop_samples_included": True,
            "redo_replay_evaluation_samples_included": True,
            "separate_actor_critic_condition": (
                "separate_learning_transactions_each_consume_the_transition"
            ),
            "separate_actor_critic_transactions_count_separately": True,
        },
        "schema_version": MEASUREMENT_LEDGER_DESCRIPTOR_SCHEMA_VERSION,
        "status": MEASUREMENT_LEDGER_STATUS,
        "structural_absence_policy": {
            "configuration_only_zero_allowed": False,
            "coupled_fields_share_subject_proof": True,
            "exact_runtime_subsystem_inventory_required": True,
            "observed_zero_sized_tree_establishes_absence": False,
        },
        "self_identity_policy": {
            "descriptor_bytes_embed_own_descriptor_sha256": False,
            "descriptor_bytes_embed_own_source_sha256": False,
            "descriptor_repository_literal_external_to_descriptor": True,
            "parser_reauthenticates_externally_supplied_source_sha256": True,
            "source_sha256_supplied_by_upstream_audited_identity": True,
        },
        "transaction_policy": {
            "aborted_transaction_counted": False,
            "commit_required": True,
            "committed_learning_zero_iff_consumed_samples_zero": True,
            "consumed_samples_at_least_committed_learning_transactions": True,
            "duplicate_transaction_id_allowed": False,
            "environment_transaction_increment": 1,
            "handle_fields_are_non_authoritative": True,
            "ledger_owned_frozen_pending_record_controls_accounting": True,
            "open_transaction_at_finalization_allowed": False,
        },
        "tree_role_policy": {
            "aliases_recorded": True,
            "cross_category_deduplication": False,
            "deduplication_scope": "within_one_category_snapshot_by_storage_id",
            "stop_gradient_global_trainable_role": "trainable_parameters",
            "target_copy_role": "target_only_not_frozen",
            "view_policy": "view_requires_owned_base_in_same_category_snapshot",
        },
        "upstream_algorithmic_resource_contract_identity": {
            "schema_version": (
                resource_contract.ALGORITHMIC_RESOURCE_CONTRACT_DESCRIPTOR_SCHEMA_VERSION
            ),
            "descriptor_sha256": (
                resource_contract.matched_v3_algorithmic_resource_contract_descriptor_sha256()
            ),
            "source_sha256": UPSTREAM_ALGORITHMIC_RESOURCE_CONTRACT_SOURCE_SHA256,
        },
    }


_MEASUREMENT_DESCRIPTOR: Final = _freeze_json(_measurement_descriptor_body())


def algorithmic_resource_measurement_ledger_descriptor() -> dict[str, Any]:
    """Return the inert ledger descriptor, which contains no self-identity."""

    return cast(dict[str, Any], _thaw_json(_MEASUREMENT_DESCRIPTOR))


def canonical_algorithmic_resource_measurement_ledger_descriptor_bytes() -> bytes:
    """Return canonical bytes for the inert ledger descriptor."""

    return _canonical_json(algorithmic_resource_measurement_ledger_descriptor())


def algorithmic_resource_measurement_ledger_descriptor_sha256() -> str:
    """Return the computed descriptor digest for repository-pin verification."""

    return hashlib.sha256(
        canonical_algorithmic_resource_measurement_ledger_descriptor_bytes()
    ).hexdigest()


def _active_measurement_ledger_descriptor_sha256() -> str:
    """Use the external repository literal once finalized, else the inert digest."""

    observed = algorithmic_resource_measurement_ledger_descriptor_sha256()
    pinned = _require_sha256(
        PINNED_MEASUREMENT_LEDGER_DESCRIPTOR_SHA256,
        "measurement ledger descriptor repository literal",
    )
    if pinned == "0" * 64:
        return observed
    if not hmac.compare_digest(observed, pinned):
        _fail("measurement ledger descriptor drifted from its repository literal")
    return pinned


def canonical_algorithmic_resource_measurement_basis_bytes(
    basis: AlgorithmicResourceMeasurementBasisV1,
) -> bytes:
    """Serialize one sealed basis within the exact artifact bound."""

    if type(basis) is not AlgorithmicResourceMeasurementBasisV1:
        _fail("measurement basis type differs")
    _validate_parsed_basis(
        basis.to_body_dict(),
        expected_measurement_source_sha256=basis.ledger_source_sha256,
    )
    emitted = basis.to_dict()
    _reject_forbidden_field_names(emitted, "complete emitted measurement basis")
    return _canonical_json(emitted)


def _json_object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("canonical JSON contains a duplicate field name")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> NoReturn:
    _fail(f"canonical JSON contains a non-finite constant: {value}")


def _expect_exact_keys(value: object, expected: frozenset[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or frozenset(value) != expected:
        _fail(f"{label} field inventory differs")
    return cast(dict[str, Any], value)


_LEAF_KEYS: Final = frozenset(
    {
        "alias_id",
        "canonical_leaf_path",
        "category",
        "dtype",
        "elements",
        "itemsize_bytes",
        "lifecycle",
        "logical_owned_bytes",
        "owner",
        "shape",
        "storage_id",
        "storage_kind",
    }
)
_SNAPSHOT_KEYS: Final = frozenset(
    {
        "category",
        "elements",
        "leaves",
        "lifecycle",
        "logical_owned_bytes",
        "replay_capacities",
        "replay_capacity_transitions",
        "snapshot_id",
        "storage_aliases",
    }
)
_CATEGORY_HIGH_WATER_KEYS: Final = frozenset(
    {
        "byte_snapshot",
        "capacity_snapshot",
        "category",
        "element_snapshot",
        "max_elements",
        "max_logical_owned_bytes",
        "max_replay_capacity_transitions",
        "observation_chain_sha256",
        "observation_count",
    }
)


def _parse_leaf_dict(value: object, category: ArrayCategory) -> ArrayLeafObservationV1:
    item = _expect_exact_keys(value, _LEAF_KEYS, "array leaf")
    canonical_path = item["canonical_leaf_path"]
    if type(canonical_path) is not str or not canonical_path.startswith("/"):
        _fail("array leaf canonical path differs")
    components = canonical_path[1:].split("/")
    if not components or any(not component for component in components):
        _fail("array leaf canonical path differs")
    shape = item["shape"]
    if type(shape) is not list:
        _fail("array leaf shape differs")
    leaf = ArrayLeafObservationV1(
        leaf_path=tuple(components),
        shape=tuple(shape),
        dtype=item["dtype"],
        owner=item["owner"],
        category=category,
        lifecycle=item["lifecycle"],
        alias_id=item["alias_id"],
        storage_id=item["storage_id"],
        storage_kind=item["storage_kind"],
    )
    if not _exact_json_equal(leaf.to_dict(), item):
        _fail("array leaf derived accounting differs")
    return leaf


def _parse_snapshot_dict(value: object, category: ArrayCategory) -> CategorySnapshotV1:
    item = _expect_exact_keys(value, _SNAPSHOT_KEYS, "category snapshot")
    if item["category"] != category:
        _fail("retained snapshot category differs")
    leaves = item["leaves"]
    capacities = item["replay_capacities"]
    if type(leaves) is not list or type(capacities) is not list:
        _fail("retained snapshot arrays differ")
    parsed_capacities: list[ReplayCapacityObservationV1] = []
    for raw_capacity in capacities:
        capacity = _expect_exact_keys(
            raw_capacity,
            frozenset({"capacity_transitions", "subsystem_id"}),
            "replay capacity",
        )
        parsed_capacities.append(
            ReplayCapacityObservationV1(
                subsystem_id=capacity["subsystem_id"],
                capacity_transitions=capacity["capacity_transitions"],
            )
        )
    snapshot = CategorySnapshotV1(
        snapshot_id=item["snapshot_id"],
        category=category,
        lifecycle=item["lifecycle"],
        leaves=tuple(_parse_leaf_dict(leaf, category) for leaf in leaves),
        replay_capacities=tuple(parsed_capacities),
    )
    if not _exact_json_equal(snapshot.to_dict(), item):
        _fail("retained snapshot derived accounting differs")
    return snapshot


def _validate_parsed_fields_and_proofs(
    value: Mapping[str, Any],
) -> tuple[dict[str, FieldMeasurementV1], dict[str, StructuralAbsenceProofV1]]:
    raw_fields = value["fields"]
    raw_proofs = value["structural_absence_proofs"]
    if type(raw_fields) is not list or type(raw_proofs) is not list:
        _fail("measurement fields and absence proofs must be exact arrays")
    field_keys = frozenset(
        {
            "absence_proof_id",
            "field_name",
            "measurement_kind",
            "measurement_scope",
            "observed_value",
            "structural_absence_kind",
        }
    )
    fields: list[FieldMeasurementV1] = []
    for raw_field in raw_fields:
        item = _expect_exact_keys(raw_field, field_keys, "measurement field")
        field = FieldMeasurementV1(**item)
        if not _exact_json_equal(field.to_dict(), item):
            _fail("measurement field encoding differs")
        fields.append(field)
    if tuple(item.field_name for item in fields) != (resource_contract.ALGORITHMIC_RESOURCE_FIELDS):
        _fail("measurement field order differs from the exact first-20 inventory")

    proof_keys = frozenset({"absence_kind", "details", "evidence_kind", "proof_id", "subject"})
    proofs: list[StructuralAbsenceProofV1] = []
    for raw_proof in raw_proofs:
        item = _expect_exact_keys(raw_proof, proof_keys, "structural absence proof")
        proof = StructuralAbsenceProofV1(**item)
        if not _exact_json_equal(proof.to_dict(), item):
            _fail("structural absence proof encoding differs")
        proofs.append(proof)
    if len({item.proof_id for item in proofs}) != len(proofs) or len(
        {item.subject for item in proofs}
    ) != len(proofs):
        _fail("structural absence proof ID or subject is duplicated")
    proof_by_id = {item.proof_id: item for item in proofs}
    expected_subjects: list[str] = []
    by_name = {item.field_name: item for item in fields}
    for subject in _ABSENCE_SUBJECTS:
        subject_fields = (
            (f"max_{subject}",) if subject in _COUNTER_SUBJECTS else _CATEGORY_TO_FIELDS[subject]
        )
        if all(by_name[field_name].observed_value == 0 for field_name in subject_fields):
            expected_subjects.append(subject)
    if [item.subject for item in proofs] != expected_subjects:
        _fail("structural absence proof subject order or coverage differs")
    for field in fields:
        if field.observed_value == 0:
            if field.absence_proof_id not in proof_by_id:
                _fail("zero measurement absence proof ID is not bound")
            proof = proof_by_id[field.absence_proof_id]
            if proof.subject != _FIELD_TO_SUBJECT[field.field_name]:
                _fail("zero measurement absence proof subject differs")
            if proof.absence_kind != field.structural_absence_kind:
                _fail("zero measurement absence kind differs from its proof")
    for left, right in COUPLED_FIELD_PAIRS:
        left_field = by_name[left]
        right_field = by_name[right]
        if (left_field.observed_value == 0) != (right_field.observed_value == 0):
            _fail(f"coupled measurement zero/nonzero state differs for {left}")
        if left_field.observed_value == 0 and (
            left_field.absence_proof_id != right_field.absence_proof_id
        ):
            _fail(f"coupled structural absence proof differs for {left}")
    return by_name, proof_by_id


def _validate_parsed_categories(
    value: object,
    fields: Mapping[str, FieldMeasurementV1],
    proof_by_id: Mapping[str, StructuralAbsenceProofV1],
) -> int:
    categories = _expect_exact_keys(
        value,
        frozenset(ARRAY_CATEGORIES),
        "category high-water inventory",
    )
    total_observations = 0
    for category in ARRAY_CATEGORIES:
        item = _expect_exact_keys(
            categories[category],
            _CATEGORY_HIGH_WATER_KEYS,
            f"category high-water {category}",
        )
        if item["category"] != category:
            _fail("category high-water label differs")
        count = _require_int(item["observation_count"], "category observation count")
        max_elements = _require_int(item["max_elements"], "category maximum elements")
        max_bytes = _require_int(
            item["max_logical_owned_bytes"],
            "category maximum logical bytes",
        )
        max_capacity = _require_int(
            item["max_replay_capacity_transitions"],
            "category maximum replay capacity",
        )
        chain_sha256 = _require_sha256(
            item["observation_chain_sha256"],
            "category observation chain",
        )
        total_observations = _checked_add(
            total_observations,
            count,
            "category observation count",
        )
        snapshots: dict[str, CategorySnapshotV1 | None] = {}
        for key in ("element_snapshot", "byte_snapshot", "capacity_snapshot"):
            raw_snapshot = item[key]
            snapshots[key] = (
                None if raw_snapshot is None else _parse_snapshot_dict(raw_snapshot, category)
            )
        subject_proofs = [proof for proof in proof_by_id.values() if proof.subject == category]
        if count == 0:
            if (
                chain_sha256 != "0" * 64
                or max_elements != 0
                or max_bytes != 0
                or max_capacity != 0
                or any(snapshot is not None for snapshot in snapshots.values())
                or len(subject_proofs) != 1
            ):
                _fail("unobserved category high-water state differs")
        else:
            if chain_sha256 == "0" * 64 or max_elements == 0 or max_bytes == 0 or subject_proofs:
                _fail("observed category high-water must be positive and non-absent")
            element_snapshot = snapshots["element_snapshot"]
            byte_snapshot = snapshots["byte_snapshot"]
            if element_snapshot is None or element_snapshot.elements != max_elements:
                _fail("retained element high-water snapshot differs")
            if byte_snapshot is None or byte_snapshot.logical_owned_bytes != max_bytes:
                _fail("retained byte high-water snapshot differs")
            capacity_snapshot = snapshots["capacity_snapshot"]
            if category == "replay_storage":
                if (
                    max_capacity == 0
                    or capacity_snapshot is None
                    or capacity_snapshot.replay_capacity_transitions != max_capacity
                ):
                    _fail("retained replay-capacity high-water snapshot differs")
            elif max_capacity != 0 or capacity_snapshot is not None:
                _fail("non-replay category contains a replay-capacity high water")

        expected_values = {
            field_name: (
                max_capacity
                if field_name == "max_replay_capacity_transitions"
                else max_bytes
                if field_name.endswith("_bytes")
                or field_name in {"max_replay_peak_bytes", "max_rollout_peak_bytes"}
                else max_elements
            )
            for field_name in _CATEGORY_TO_FIELDS[category]
        }
        for field_name, expected_value in expected_values.items():
            if fields[field_name].observed_value != expected_value:
                _fail(f"field differs from category high water for {field_name}")
    return total_observations


def _validate_parsed_accounting(
    value: Mapping[str, Any],
    fields: Mapping[str, FieldMeasurementV1],
    category_observation_count: int,
) -> None:
    observation = _expect_exact_keys(
        value["observation_chain"],
        frozenset(
            {
                "category_observation_chain_sha256",
                "category_observation_count",
                "chain_algorithm",
                "initial_sha256",
                "transaction_event_chain_sha256",
                "transaction_event_count",
            }
        ),
        "observation chain",
    )
    if observation["chain_algorithm"] != "sha256_predecessor_plus_event_digest_v1":
        _fail("observation-chain algorithm differs")
    if observation["initial_sha256"] != "0" * 64:
        _fail("observation-chain initial digest differs")
    observed_category_count = _require_int(
        observation["category_observation_count"],
        "category observation count",
    )
    event_count = _require_int(
        observation["transaction_event_count"],
        "transaction event count",
    )
    category_chain = _require_sha256(
        observation["category_observation_chain_sha256"],
        "category observation chain",
    )
    transaction_chain = _require_sha256(
        observation["transaction_event_chain_sha256"],
        "transaction event chain",
    )
    if observed_category_count != category_observation_count:
        _fail("category observation count differs from high-water inventory")
    if (observed_category_count == 0) != (category_chain == "0" * 64):
        _fail("category observation-chain empty state differs")
    if (event_count == 0) != (transaction_chain == "0" * 64):
        _fail("transaction event-chain empty state differs")

    accounting = _expect_exact_keys(
        value["transaction_accounting"],
        frozenset(
            {
                "aborted_transactions",
                "committed_environment_transactions",
                "committed_gradient_transactions",
                "committed_learning_transactions",
                "committed_optimizer_transactions",
                "consumed_sample_contributions",
                "open_transactions",
                "registered_transactions",
            }
        ),
        "transaction accounting",
    )
    exact = {
        key: _require_int(item, f"transaction accounting {key}") for key, item in accounting.items()
    }
    if exact["open_transactions"] != 0:
        _fail("sealed basis contains an open transaction")
    if exact["registered_transactions"] != event_count:
        _fail("registered transaction count differs from event count")
    if exact["registered_transactions"] != (
        exact["aborted_transactions"]
        + exact["committed_environment_transactions"]
        + exact["committed_learning_transactions"]
    ):
        _fail("transaction terminal-state accounting differs")
    field_counter_pairs = {
        "committed_environment_transactions": "max_environment_interactions",
        "committed_gradient_transactions": "max_gradient_updates",
        "committed_optimizer_transactions": "max_optimizer_updates",
        "consumed_sample_contributions": "max_sample_updates",
    }
    for accounting_name, field_name in field_counter_pairs.items():
        if exact[accounting_name] != fields[field_name].observed_value:
            _fail(f"transaction accounting differs for {field_name}")
    if exact["committed_gradient_transactions"] > exact["committed_optimizer_transactions"]:
        _fail("gradient transaction count exceeds optimizer transaction count")
    if exact["committed_optimizer_transactions"] > exact["committed_learning_transactions"]:
        _fail("optimizer transaction count exceeds learning transaction count")
    committed_learning = exact["committed_learning_transactions"]
    consumed_samples = exact["consumed_sample_contributions"]
    if (committed_learning == 0) != (consumed_samples == 0):
        _fail("committed learning and consumed sample zero states differ")
    if consumed_samples < committed_learning:
        _fail("consumed sample count is below committed learning transactions")


_BASIS_BODY_KEYS: Final = frozenset(
    {
        "authority",
        "capabilities",
        "category_high_water",
        "claims",
        "classification",
        "field_inventory_sha256",
        "fields",
        "ledger_descriptor_sha256",
        "ledger_id",
        "ledger_source_sha256",
        "limitations",
        "observation_chain",
        "readiness",
        "schema_version",
        "status",
        "structural_absence_proofs",
        "transaction_accounting",
    }
)


def _validate_parsed_basis(
    value: dict[str, Any],
    *,
    expected_measurement_source_sha256: str,
) -> None:
    _expect_exact_keys(
        {key: item for key, item in value.items() if key != "measurement_basis_body_sha256"},
        _BASIS_BODY_KEYS,
        "measurement basis body",
    )
    if value["schema_version"] != MEASUREMENT_BASIS_SCHEMA_VERSION:
        _fail("measurement basis schema differs")
    if value["status"] != MEASUREMENT_BASIS_STATUS:
        _fail("measurement basis status differs")
    if value["classification"] != MEASUREMENT_CLASSIFICATION:
        _fail("measurement basis classification differs")
    _require_identifier(value["ledger_id"], "measurement ledger ID")
    descriptor_sha256 = _require_sha256(
        value["ledger_descriptor_sha256"],
        "measurement ledger descriptor",
    )
    if descriptor_sha256 != _active_measurement_ledger_descriptor_sha256():
        _fail("measurement ledger descriptor digest differs")
    source_sha256 = _require_sha256(value["ledger_source_sha256"], "measurement ledger source")
    expected_source_sha256 = _require_sha256(
        expected_measurement_source_sha256,
        "externally supplied expected measurement source",
    )
    if expected_source_sha256 == "0" * 64 or not hmac.compare_digest(
        source_sha256,
        expected_source_sha256,
    ):
        _fail("measurement ledger source differs from its externally supplied identity")
    for key, expected in (
        ("authority", _false_authority()),
        ("capabilities", _false_capabilities()),
        ("claims", _false_claims()),
        ("readiness", _false_readiness()),
    ):
        if not _exact_json_equal(value[key], expected) or any(expected.values()):
            _fail(f"measurement basis {key} differs from all-false policy")
    if not _exact_json_equal(value["limitations"], list(_BASIS_LIMITATIONS)):
        _fail("measurement basis limitations differ")

    fields, proof_by_id = _validate_parsed_fields_and_proofs(value)
    field_inventory_sha256 = _require_sha256(
        value["field_inventory_sha256"],
        "measurement field inventory",
    )
    if (
        field_inventory_sha256
        != hashlib.sha256(_canonical_json({"fields": value["fields"]})[:-1]).hexdigest()
    ):
        _fail("measurement field inventory digest differs")
    if fields["max_environment_interactions"].observed_value == 0:
        _fail("environment interaction measurement must be positive")
    category_observations = _validate_parsed_categories(
        value["category_high_water"],
        fields,
        proof_by_id,
    )
    _validate_parsed_accounting(value, fields, category_observations)
    if fields["max_optimizer_updates"].observed_value == 0 and (
        fields["max_optimizer_state_elements"].observed_value != 0
        or fields["max_optimizer_state_bytes"].observed_value != 0
    ):
        _fail("an absent optimizer subsystem cannot retain optimizer state")
    if fields["max_optimizer_updates"].observed_value > 0 and (
        fields["max_sample_updates"].observed_value == 0
        or fields["max_trainable_parameters"].observed_value == 0
    ):
        _fail("positive optimizer updates require samples and trainable parameters")
    if fields["max_gradient_updates"].observed_value > 0 and (
        fields["max_optimizer_updates"].observed_value == 0
        or fields["max_sample_updates"].observed_value == 0
        or fields["max_trainable_parameters"].observed_value == 0
    ):
        _fail(
            "positive gradient updates require optimizer updates, samples, and trainable parameters"
        )


def parse_algorithmic_resource_measurement_basis(
    raw: bytes,
    *,
    expected_file_sha256: str,
    expected_measurement_source_sha256: str,
) -> dict[str, Any]:
    """Parse a basis under exact file and externally audited source identities."""

    if type(raw) is not bytes or not raw or len(raw) > MAX_MEASUREMENT_BASIS_BYTES:
        _fail("measurement basis file bytes are empty, oversized, or not exact bytes")
    expected = _require_sha256(expected_file_sha256, "measurement basis file")
    if expected == "0" * 64 or not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected):
        _fail("measurement basis file SHA-256 differs")
    try:
        decoded = raw.decode("ascii")
        parsed = json.loads(
            decoded,
            object_pairs_hook=_json_object_without_duplicates,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise AlgorithmicResourceMeasurementError(
            "measurement basis is not bounded canonical JSON"
        ) from exc
    if type(parsed) is not dict:
        _fail("measurement basis root must be an exact object")
    _reject_forbidden_field_names(parsed, "measurement basis")
    if _canonical_json(parsed) != raw:
        _fail("measurement basis bytes are not canonical")
    expected_top_keys = _BASIS_BODY_KEYS | frozenset({"measurement_basis_body_sha256"})
    exact = _expect_exact_keys(parsed, expected_top_keys, "measurement basis")
    provided_body_sha256 = _require_sha256(
        exact["measurement_basis_body_sha256"],
        "measurement basis body digest",
    )
    body = {
        key: copy.deepcopy(item)
        for key, item in exact.items()
        if key != "measurement_basis_body_sha256"
    }
    if not hmac.compare_digest(provided_body_sha256, _body_sha256(body)):
        _fail("measurement basis body digest differs")
    _validate_parsed_basis(
        exact,
        expected_measurement_source_sha256=expected_measurement_source_sha256,
    )
    return copy.deepcopy(exact)


__all__ = [
    "ABSENCE_KIND_BY_SUBJECT",
    "ARRAY_CATEGORIES",
    "COUPLED_FIELD_PAIRS",
    "MAX_ARRAY_DIMENSION",
    "MAX_ARRAY_RANK",
    "MAX_INTEGER",
    "MAX_JSON_DEPTH",
    "MAX_JSON_NODES",
    "MAX_LEAVES_PER_SNAPSHOT",
    "MAX_MEASUREMENT_BASIS_BYTES",
    "MAX_TEXT_LENGTH",
    "MEASUREMENT_BASIS_SCHEMA_VERSION",
    "MEASUREMENT_BASIS_STATUS",
    "MEASUREMENT_CLASSIFICATION",
    "MEASUREMENT_LEDGER_DESCRIPTOR_SCHEMA_VERSION",
    "MEASUREMENT_LEDGER_STATUS",
    "PINNED_MEASUREMENT_LEDGER_DESCRIPTOR_SHA256",
    "UPSTREAM_ALGORITHMIC_RESOURCE_CONTRACT_SOURCE_SHA256",
    "ArrayCategory",
    "StorageKind",
    "AlgorithmicResourceMeasurementBasisV1",
    "AlgorithmicResourceMeasurementError",
    "AlgorithmicResourceMeasurementLedger",
    "ArrayLeafObservationV1",
    "CategorySnapshotV1",
    "FieldMeasurementV1",
    "ReplayCapacityObservationV1",
    "StructuralAbsenceProofV1",
    "algorithmic_resource_measurement_ledger_descriptor",
    "algorithmic_resource_measurement_ledger_descriptor_sha256",
    "canonical_algorithmic_resource_measurement_basis_bytes",
    "canonical_algorithmic_resource_measurement_ledger_descriptor_bytes",
    "classify_rtu_leaf_path",
    "parse_algorithmic_resource_measurement_basis",
]
