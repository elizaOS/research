"""Authenticated development-only identity rollover for one structural scrub.

The production birth-identity ledger advances only through public curation
traces.  An evaluator-owned expanded-lineage scrub deliberately changes the
same descriptor banks outside that path.  This module closes that one boundary
without pretending that the scrub was a learner update: it independently
validates the complete scrub and fresh-epoch plan, retires every masked
lifetime, and returns an ordinary schema-v4 ledger state which can be consumed
by the next production trace authentication.

This is host-only bookkeeping.  It executes no learner update, applies no
fresh key, runs no freeze window, writes no artifact, and grants no execution
or evidence authority.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import struct
from typing import Any, Final, cast

import jax
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.core.compositional_features import (
    CompositionalFeatureState,
)
from alberta_framework.evaluation import generated_birth_identity_ledger as _ledger
from alberta_framework.evaluation.generated_birth_identity_ledger import (
    GeneratedBirthIdentityLedgerV4Config,
    GeneratedBirthIdentityLedgerV4State,
)
from alberta_framework.evaluation.generated_class_lifecycle_scrub import (
    GeneratedClassScrubConfig,
    compositional_state_leaf_paths,
)
from alberta_framework.evaluation.generated_class_recurrence import GeneratedExpression
from alberta_framework.evaluation.generated_expression_lineage import (
    ExpandedExpressionLineageConfig,
    ExpandedExpressionLineagePlan,
    ExpandedExpressionScrubValidation,
    validate_post_scrub_expanded_expression_absence,
)
from alberta_framework.evaluation.generated_reacquisition_epoch import (
    GeneratedReacquisitionEpochConfig,
    GeneratedReacquisitionEpochPlan,
    build_generated_reacquisition_epoch_plan,
    validate_generated_reacquisition_epoch_plan,
)

GENERATED_BIRTH_IDENTITY_SCRUB_EPOCH_SCHEMA: Final = (
    "alberta.generated-birth-identity-scrub-epoch-rollover.development.v0"
)
GENERATED_BIRTH_IDENTITY_SCRUB_EPOCH_STATUS: Final = (
    "DEVELOPMENT_HOST_ONLY_NO_EXECUTION_OR_EVIDENCE_AUTHORITY"
)
_IDENTITY_BYTES: Final = 32
_IDENTITY_DOMAIN: Final = "expanded-lineage-scrub-placeholder-birth-v0"
_PRNG_IMPL: Final = "threefry2x32"


class GeneratedBirthIdentityScrubEpochError(RuntimeError):
    """Raised when an external scrub cannot be authenticated canonically."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GeneratedBirthIdentityScrubEpochError(message)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _exact_record(value: object, *, path: str) -> object:
    if isinstance(value, Array) and jax.dtypes.issubdtype(  # type: ignore[attr-defined]
        value.dtype,
        jax.dtypes.prng_key,
    ):
        return {
            "kind": "typed-prng",
            "implementation": str(jr.key_impl(value)),
            "key_data": _exact_record(jr.key_data(value), path=f"{path}.key_data"),
        }
    if isinstance(value, Array):
        array = np.asarray(value)
        return {
            "kind": "jax-array",
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "raw_hex": np.ascontiguousarray(array).tobytes(order="C").hex(),
        }
    if isinstance(value, np.ndarray):
        return {
            "kind": "numpy-array",
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "writeable": bool(value.flags.writeable),
            "raw_hex": np.ascontiguousarray(value).tobytes(order="C").hex(),
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "kind": "dataclass",
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                field.name: _exact_record(
                    getattr(value, field.name),
                    path=f"{path}.{field.name}",
                )
                for field in dataclasses.fields(value)
            },
        }
    if type(value) is tuple:
        return {
            "kind": "tuple",
            "items": [
                _exact_record(item, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            ],
        }
    if type(value) is list:
        return {
            "kind": "list",
            "items": [
                _exact_record(item, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            ],
        }
    if type(value) is dict:
        if not all(type(key) is str for key in value):
            raise TypeError(f"mapping at {path} must have exact string keys")
        mapping = cast(dict[str, object], value)
        return {
            "kind": "dict",
            "items": {
                key: _exact_record(mapping[key], path=f"{path}.{key}")
                for key in sorted(mapping)
            },
        }
    if type(value) is str:
        return {"kind": "str", "value": value}
    if type(value) is bool:
        return {"kind": "bool", "value": value}
    if type(value) is int:
        return {"kind": "int", "value": value}
    if type(value) is float:
        return {"kind": "float", "raw_hex": struct.pack(">d", value).hex()}
    if value is None:
        return {"kind": "none"}
    raise TypeError(f"unsupported exact record value at {path}: {type(value)!r}")


def _same_exact(left: object, right: object, *, name: str) -> None:
    _require(
        _exact_record(left, path=f"left.{name}")
        == _exact_record(right, path=f"right.{name}"),
        f"{name} differs in exact type, dtype, shape, mutability, or raw bits",
    )


def _path_text(path: tuple[Any, ...]) -> str:
    names: list[str] = []
    for key in path:
        name = getattr(key, "name", None)
        if not isinstance(name, str):
            raise TypeError(f"unsupported compositional state path key: {key!r}")
        names.append(name)
    return ".".join(names)


def _normalized_array_bytes(array: np.ndarray[Any, Any]) -> bytes:
    if array.dtype.hasobject:
        raise TypeError("object arrays cannot be state-hash leaves")
    if array.dtype.byteorder == "|":
        normalized = np.ascontiguousarray(array)
    else:
        normalized = np.ascontiguousarray(
            array.astype(array.dtype.newbyteorder(">"), copy=False)
        )
    return normalized.tobytes(order="C")


def _typed_key_data(key: object, *, name: str) -> tuple[int, int]:
    if not isinstance(key, Array) or not jax.dtypes.issubdtype(  # type: ignore[attr-defined]
        key.dtype,
        jax.dtypes.prng_key,
    ):
        raise TypeError(f"{name} must be a typed JAX key")
    if str(jr.key_impl(key)) != _PRNG_IMPL:
        raise ValueError(f"{name} must use {_PRNG_IMPL}")
    data = np.asarray(jr.key_data(key))
    if key.shape != () or data.shape != (2,) or data.dtype != np.uint32:
        raise ValueError(f"{name} must contain exactly two uint32 words")
    return int(data[0]), int(data[1])


def generated_birth_identity_scrub_epoch_core_state_sha256(
    state: CompositionalFeatureState,
) -> str:
    """Hash every core-state leaf with exact dtype, shape, and raw bits."""

    if type(state) is not CompositionalFeatureState:
        raise TypeError("state must be an exact CompositionalFeatureState")
    compositional_state_leaf_paths(state)
    digest = hashlib.sha256()
    for path, leaf in jax.tree_util.tree_flatten_with_path(state)[0]:
        path_name = _path_text(path)
        if isinstance(leaf, Array) and jax.dtypes.issubdtype(  # type: ignore[attr-defined]
            leaf.dtype,
            jax.dtypes.prng_key,
        ):
            key_data = _typed_key_data(leaf, name=f"state.{path_name}")
            array = np.asarray(key_data, dtype=">u4")
            kind = "typed-prng-threefry2x32"
            dtype = ">u4"
            raw = array.tobytes(order="C")
        elif isinstance(leaf, Array):
            array = np.asarray(leaf)
            kind = "jax-array"
            dtype = array.dtype.newbyteorder(">").str
            raw = _normalized_array_bytes(array)
        elif type(leaf) is float:
            array = np.asarray(leaf, dtype=">f8")
            kind = "python-float"
            dtype = ">f8"
            raw = struct.pack(">d", leaf)
        else:
            raise TypeError(
                f"unsupported state-hash leaf at {path_name}: {type(leaf)!r}"
            )
        metadata = _canonical_json_bytes(
            {
                "path": path_name,
                "kind": kind,
                "dtype": dtype,
                "shape": list(array.shape),
                "nbytes": len(raw),
            }
        )
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _array(
    value: object,
    *,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray[Any, Any]:
    array = np.asarray(value)
    if array.dtype != dtype or array.shape != shape:
        raise TypeError(
            f"{name} must have exact dtype {dtype} and shape {shape}; "
            f"got {array.dtype} and {array.shape}"
        )
    return np.ascontiguousarray(array)


def _readonly(value: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    result = np.array(value, copy=True, order="C")
    result.setflags(write=False)
    return result


def _same_bits(left: object, right: object, *, name: str) -> None:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    _require(
        left_array.dtype == right_array.dtype
        and left_array.shape == right_array.shape
        and left_array.tobytes(order="C") == right_array.tobytes(order="C"),
        f"{name} differs in dtype, shape, or raw bits",
    )


def _descriptors(
    state: CompositionalFeatureState,
    config: GeneratedBirthIdentityLedgerV4Config,
) -> dict[str, np.ndarray[Any, Any]]:
    if type(state) is not CompositionalFeatureState:
        raise TypeError("core state must be an exact CompositionalFeatureState")
    result: dict[str, np.ndarray[Any, Any]] = {}
    for name, shape in (
        ("ops", (config.active_slots,)),
        ("parent_a", (config.active_slots,)),
        ("parent_b", (config.active_slots,)),
        ("depth", (config.active_slots,)),
        ("feature_generator_policy", (config.active_slots,)),
        ("candidate_ops", (config.candidate_slots,)),
        ("candidate_parent_a", (config.candidate_slots,)),
        ("candidate_parent_b", (config.candidate_slots,)),
        ("candidate_depth", (config.candidate_slots,)),
        ("candidate_generator_policy", (config.candidate_slots,)),
    ):
        result[name] = _array(
            getattr(state, name),
            dtype=np.dtype(np.int32),
            shape=shape,
            name=f"core_state.{name}",
        )
    return result


def _validate_pre_ledger_binding(
    config: GeneratedBirthIdentityLedgerV4Config,
    pre_ledger_state: GeneratedBirthIdentityLedgerV4State,
    pre_core_state: CompositionalFeatureState,
) -> dict[str, np.ndarray[Any, Any]]:
    if type(config) is not GeneratedBirthIdentityLedgerV4Config:
        raise TypeError("config must be an exact GeneratedBirthIdentityLedgerV4Config")
    _ledger._validate_v4_state(config, pre_ledger_state)  # noqa: SLF001
    descriptors = _descriptors(pre_core_state, config)
    for ledger_name, core_name in (
        ("active_ops", "ops"),
        ("active_parent_a", "parent_a"),
        ("active_parent_b", "parent_b"),
        ("active_depth", "depth"),
        ("active_generator_policy", "feature_generator_policy"),
        ("candidate_ops", "candidate_ops"),
        ("candidate_parent_a", "candidate_parent_a"),
        ("candidate_parent_b", "candidate_parent_b"),
        ("candidate_depth", "candidate_depth"),
        ("candidate_generator_policy", "candidate_generator_policy"),
    ):
        _same_bits(
            getattr(pre_ledger_state, ledger_name),
            descriptors[core_name],
            name=f"pre ledger/core {ledger_name}",
        )
    core_words = _array(
        pre_core_state.step_words,
        dtype=np.dtype(np.uint32),
        shape=(2,),
        name="pre_core_state.step_words",
    )
    _same_bits(pre_ledger_state.step_words, core_words, name="pre ledger/core step words")
    return descriptors


def _identity(
    *,
    config: GeneratedBirthIdentityLedgerV4Config,
    pre_ledger_state: GeneratedBirthIdentityLedgerV4State,
    pre_core_sha256: str,
    post_core_sha256: str,
    lineage_plan_sha256: str,
    scrub_validation_sha256: str,
    epoch_plan: GeneratedReacquisitionEpochPlan,
    bank: str,
    slot: int,
    ordinal: int,
) -> np.ndarray[Any, Any]:
    payload = {
        "schema": GENERATED_BIRTH_IDENTITY_SCRUB_EPOCH_SCHEMA,
        "identity_domain": _IDENTITY_DOMAIN,
        "ledger_schema": config.schema,
        "ledger_namespace": config.namespace,
        "paired_development_life_seed": pre_ledger_state.paired_development_life_seed,
        "step_words_uint32_be": [int(word) for word in pre_ledger_state.step_words],
        "pre_ledger_state_sha256": pre_ledger_state.integrity_sha256,
        "pre_core_state_sha256": pre_core_sha256,
        "post_core_state_sha256": post_core_sha256,
        "lineage_plan_sha256": lineage_plan_sha256,
        "scrub_validation_sha256": scrub_validation_sha256,
        "reacquisition_contract_sha256": epoch_plan.contract.contract_sha256,
        "fresh_learner_key_data_uint32": list(
            _typed_key_data(epoch_plan.fresh_learner_key, name="fresh learner key")
        ),
        "bank": bank,
        "slot": slot,
        "ordinal": ordinal,
    }
    return np.frombuffer(hashlib.sha256(_canonical_json_bytes(payload)).digest(), dtype=np.uint8)


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedBirthIdentityScrubEpochAudit:
    """Complete hash chain and non-authority disclosures for one rollover."""

    schema: str
    status: str
    ledger_config_sha256: str
    pre_ledger_state_sha256: str
    pre_core_state_sha256: str
    post_core_state_sha256: str
    lineage_plan_sha256: str
    scrub_validation_sha256: str
    reacquisition_contract_sha256: str
    fresh_learner_key_data_uint32: tuple[int, int]
    step_words_uint32: tuple[int, int]
    core_step_count: int
    core_replacement_phase: int
    active_scrub_count: int
    candidate_scrub_count: int
    new_identity_count: int
    unmasked_identities_preserved: bool
    scrubbed_identities_replaced: bool
    new_identities_nonzero_unique: bool
    parent_identity_snapshots_rebuilt: bool
    structural_scrub_valid: bool
    reacquisition_epoch_plan_valid: bool
    core_step_words_unchanged: bool
    core_step_count_unchanged: bool
    core_replacement_phase_unchanged: bool
    core_learner_key_unchanged: bool
    transaction_sha256: str
    development_only: bool
    execution_authorized: bool
    runner_authorized: bool
    campaign_authorized: bool
    artifact_writes_authorized: bool
    threshold_authorized: bool
    evidence_authorized: bool
    scientific_promotion_allowed: bool


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedBirthIdentityScrubEpochTransaction:
    """Canonical external scrub assignments and the rebound normal v4 state."""

    active_scrub_mask: np.ndarray[Any, Any]
    candidate_scrub_mask: np.ndarray[Any, Any]
    active_scrub_birth_identity: np.ndarray[Any, Any]
    candidate_scrub_birth_identity: np.ndarray[Any, Any]
    structural_scrub_validation: ExpandedExpressionScrubValidation
    reacquisition_epoch_plan: GeneratedReacquisitionEpochPlan
    post_ledger_state: GeneratedBirthIdentityLedgerV4State
    audit: GeneratedBirthIdentityScrubEpochAudit


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedBirthIdentityScrubEpochValidation:
    """Independent exact reconstruction result for one untrusted transaction."""

    valid: bool
    canonical_transaction_sha256: str
    supplied_transaction_sha256: str
    output_is_normal_v4_state: bool
    structural_scrub_valid: bool
    reacquisition_epoch_plan_valid: bool
    development_only: bool
    execution_authorized: bool
    runner_authorized: bool
    campaign_authorized: bool
    artifact_writes_authorized: bool
    threshold_authorized: bool
    evidence_authorized: bool
    scientific_promotion_allowed: bool


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedBirthIdentityScrubEpochInputs:
    """Exact original inputs required to revalidate an untrusted rollover."""

    config: GeneratedBirthIdentityLedgerV4Config
    pre_ledger_state: GeneratedBirthIdentityLedgerV4State
    pre_core_state: CompositionalFeatureState
    post_core_state: CompositionalFeatureState
    target: GeneratedExpression
    lineage_plan: ExpandedExpressionLineagePlan
    lineage_config: ExpandedExpressionLineageConfig
    scrub_config: GeneratedClassScrubConfig
    epoch_config: GeneratedReacquisitionEpochConfig


def _transaction_payload(
    transaction: GeneratedBirthIdentityScrubEpochTransaction,
    *,
    include_transaction_sha256: bool,
) -> dict[str, object]:
    audit = dataclasses.asdict(transaction.audit)
    if not include_transaction_sha256:
        audit.pop("transaction_sha256")
    return {
        "active_scrub_mask": np.asarray(transaction.active_scrub_mask).astype(
            np.uint8
        ).tolist(),
        "candidate_scrub_mask": np.asarray(transaction.candidate_scrub_mask).astype(
            np.uint8
        ).tolist(),
        "active_scrub_birth_identity_hex": [
            bytes(row).hex() for row in np.asarray(transaction.active_scrub_birth_identity)
        ],
        "candidate_scrub_birth_identity_hex": [
            bytes(row).hex()
            for row in np.asarray(transaction.candidate_scrub_birth_identity)
        ],
        "structural_scrub_validation": dataclasses.asdict(
            transaction.structural_scrub_validation
        ),
        "reacquisition_epoch_contract": dataclasses.asdict(
            transaction.reacquisition_epoch_plan.contract
        ),
        "reacquisition_epoch_fresh_key_data_uint32": list(
            _typed_key_data(
                transaction.reacquisition_epoch_plan.fresh_learner_key,
                name="transaction fresh learner key",
            )
        ),
        "post_ledger_state_sha256": transaction.post_ledger_state.integrity_sha256,
        "audit": audit,
    }


def generated_birth_identity_scrub_epoch_transaction_sha256(
    transaction: GeneratedBirthIdentityScrubEpochTransaction,
) -> str:
    """Hash one complete external rollover, excluding its self-hash field."""

    return _sha256_json(_transaction_payload(transaction, include_transaction_sha256=False))


def _build_transaction(
    config: GeneratedBirthIdentityLedgerV4Config,
    pre_ledger_state: GeneratedBirthIdentityLedgerV4State,
    pre_core_state: CompositionalFeatureState,
    post_core_state: CompositionalFeatureState,
    target: GeneratedExpression,
    lineage_plan: ExpandedExpressionLineagePlan,
    *,
    lineage_config: ExpandedExpressionLineageConfig,
    scrub_config: GeneratedClassScrubConfig,
    epoch_config: GeneratedReacquisitionEpochConfig,
) -> GeneratedBirthIdentityScrubEpochTransaction:
    _validate_pre_ledger_binding(config, pre_ledger_state, pre_core_state)
    if type(epoch_config) is not GeneratedReacquisitionEpochConfig:
        raise TypeError("epoch_config must be an exact GeneratedReacquisitionEpochConfig")
    _require(
        epoch_config.paired_life_seed
        == pre_ledger_state.paired_development_life_seed,
        "reacquisition epoch life seed differs from the birth-identity ledger life",
    )
    if type(lineage_plan) is not ExpandedExpressionLineagePlan:
        raise TypeError("lineage_plan must be an exact ExpandedExpressionLineagePlan")
    structural = validate_post_scrub_expanded_expression_absence(
        pre_core_state,
        post_core_state,
        target,
        lineage_plan,
        config=lineage_config,
        scrub_config=scrub_config,
    )
    _require(structural.valid, "expanded-expression scrub validation rejected")
    epoch_plan = build_generated_reacquisition_epoch_plan(
        pre_core_state,
        post_core_state,
        target,
        lineage_plan,
        lineage_config=lineage_config,
        scrub_config=scrub_config,
        config=epoch_config,
    )
    epoch_validation = validate_generated_reacquisition_epoch_plan(
        epoch_plan,
        pre_core_state,
        post_core_state,
        target,
        lineage_plan,
        lineage_config=lineage_config,
        scrub_config=scrub_config,
        config=epoch_config,
    )
    _require(epoch_validation.valid, "reacquisition epoch plan validation rejected")

    active_mask = _array(
        lineage_plan.active_mask,
        dtype=np.dtype(np.bool_),
        shape=(config.active_slots,),
        name="lineage_plan.active_mask",
    )
    candidate_mask = _array(
        lineage_plan.candidate_mask,
        dtype=np.dtype(np.bool_),
        shape=(config.candidate_slots,),
        name="lineage_plan.candidate_mask",
    )
    _require(bool(np.any(active_mask) or np.any(candidate_mask)), "scrub mask is empty")

    pre_words = _array(
        pre_core_state.step_words,
        dtype=np.dtype(np.uint32),
        shape=(2,),
        name="pre_core_state.step_words",
    )
    post_words = _array(
        post_core_state.step_words,
        dtype=np.dtype(np.uint32),
        shape=(2,),
        name="post_core_state.step_words",
    )
    _same_bits(pre_words, post_words, name="scrub step words")
    _same_bits(pre_core_state.step_count, post_core_state.step_count, name="scrub step_count")
    _same_bits(
        pre_core_state.replacement_phase,
        post_core_state.replacement_phase,
        name="scrub replacement_phase",
    )
    _require(
        _typed_key_data(pre_core_state.key, name="pre scrub learner key")
        == _typed_key_data(post_core_state.key, name="post scrub learner key"),
        "scrub changed learner key",
    )

    pre_core_sha256 = generated_birth_identity_scrub_epoch_core_state_sha256(pre_core_state)
    post_core_sha256 = generated_birth_identity_scrub_epoch_core_state_sha256(post_core_state)
    post_descriptors = _descriptors(post_core_state, config)
    active_assignments = np.zeros(
        (config.active_slots, _IDENTITY_BYTES), dtype=np.uint8
    )
    candidate_assignments = np.zeros(
        (config.candidate_slots, _IDENTITY_BYTES), dtype=np.uint8
    )
    ordinal = 0
    for slot in np.flatnonzero(active_mask):
        active_assignments[int(slot)] = _identity(
            config=config,
            pre_ledger_state=pre_ledger_state,
            pre_core_sha256=pre_core_sha256,
            post_core_sha256=post_core_sha256,
            lineage_plan_sha256=lineage_plan.audit.plan_sha256,
            scrub_validation_sha256=structural.validation_sha256,
            epoch_plan=epoch_plan,
            bank="active",
            slot=int(slot),
            ordinal=ordinal,
        )
        ordinal += 1
    for slot in np.flatnonzero(candidate_mask):
        candidate_assignments[int(slot)] = _identity(
            config=config,
            pre_ledger_state=pre_ledger_state,
            pre_core_sha256=pre_core_sha256,
            post_core_sha256=post_core_sha256,
            lineage_plan_sha256=lineage_plan.audit.plan_sha256,
            scrub_validation_sha256=structural.validation_sha256,
            epoch_plan=epoch_plan,
            bank="candidate",
            slot=int(slot),
            ordinal=ordinal,
        )
        ordinal += 1

    post_active_identity = np.array(pre_ledger_state.active_identity, copy=True)
    post_candidate_identity = np.array(pre_ledger_state.candidate_identity, copy=True)
    post_active_identity[active_mask] = active_assignments[active_mask]
    post_candidate_identity[candidate_mask] = candidate_assignments[candidate_mask]
    active_sampled = np.array(
        pre_ledger_state.structural_state.active_generator_policy_sampled,
        copy=True,
    )
    candidate_sampled = np.array(
        pre_ledger_state.structural_state.candidate_generator_policy_sampled,
        copy=True,
    )
    active_sampled[active_mask] = False
    candidate_sampled[candidate_mask] = False

    v3_config = _ledger._v4_to_v3_config(config)  # noqa: SLF001
    structural_post = _ledger._make_state(  # noqa: SLF001
        v3_config,
        seed=pre_ledger_state.paired_development_life_seed,
        step=0,
        active_identity=post_active_identity,
        active_parent_a=post_descriptors["parent_a"],
        active_parent_b=post_descriptors["parent_b"],
        active_ops=post_descriptors["ops"],
        active_depth=post_descriptors["depth"],
        active_generator_policy=post_descriptors["feature_generator_policy"],
        active_generator_policy_sampled=active_sampled,
        candidate_identity=post_candidate_identity,
        candidate_parent_a=post_descriptors["candidate_parent_a"],
        candidate_parent_b=post_descriptors["candidate_parent_b"],
        candidate_ops=post_descriptors["candidate_ops"],
        candidate_depth=post_descriptors["candidate_depth"],
        candidate_generator_policy=post_descriptors["candidate_generator_policy"],
        candidate_generator_policy_sampled=candidate_sampled,
    )
    post_ledger = _ledger._make_v4_state(  # noqa: SLF001
        config,
        step_words=cast(Any, post_words),
        structural_state=structural_post,
    )

    new_rows = np.concatenate(
        (active_assignments[active_mask], candidate_assignments[candidate_mask]),
        axis=0,
    )
    new_identity_bytes = [bytes(row) for row in new_rows]
    nonzero_unique = bool(
        all(value != bytes(_IDENTITY_BYTES) for value in new_identity_bytes)
        and len(new_identity_bytes) == len(set(new_identity_bytes))
    )
    _require(nonzero_unique, "scrub placeholder identities are zero or collide")
    unmasked_preserved = bool(
        np.array_equal(
            post_ledger.active_identity[~active_mask],
            pre_ledger_state.active_identity[~active_mask],
        )
        and np.array_equal(
            post_ledger.candidate_identity[~candidate_mask],
            pre_ledger_state.candidate_identity[~candidate_mask],
        )
    )
    replaced = bool(
        np.all(
            np.any(
                post_ledger.active_identity[active_mask]
                != pre_ledger_state.active_identity[active_mask],
                axis=1,
            )
        )
        and np.all(
            np.any(
                post_ledger.candidate_identity[candidate_mask]
                != pre_ledger_state.candidate_identity[candidate_mask],
                axis=1,
            )
        )
    )
    _require(unmasked_preserved, "unmasked identities changed")
    _require(replaced, "one or more scrubbed identities were retained")

    audit = GeneratedBirthIdentityScrubEpochAudit(
        schema=GENERATED_BIRTH_IDENTITY_SCRUB_EPOCH_SCHEMA,
        status=GENERATED_BIRTH_IDENTITY_SCRUB_EPOCH_STATUS,
        ledger_config_sha256=pre_ledger_state.config_sha256,
        pre_ledger_state_sha256=pre_ledger_state.integrity_sha256,
        pre_core_state_sha256=pre_core_sha256,
        post_core_state_sha256=post_core_sha256,
        lineage_plan_sha256=lineage_plan.audit.plan_sha256,
        scrub_validation_sha256=structural.validation_sha256,
        reacquisition_contract_sha256=epoch_plan.contract.contract_sha256,
        fresh_learner_key_data_uint32=_typed_key_data(
            epoch_plan.fresh_learner_key,
            name="fresh learner key",
        ),
        step_words_uint32=(int(post_words[0]), int(post_words[1])),
        core_step_count=int(np.asarray(post_core_state.step_count)),
        core_replacement_phase=int(np.asarray(post_core_state.replacement_phase)),
        active_scrub_count=int(np.count_nonzero(active_mask)),
        candidate_scrub_count=int(np.count_nonzero(candidate_mask)),
        new_identity_count=ordinal,
        unmasked_identities_preserved=True,
        scrubbed_identities_replaced=True,
        new_identities_nonzero_unique=True,
        parent_identity_snapshots_rebuilt=True,
        structural_scrub_valid=True,
        reacquisition_epoch_plan_valid=True,
        core_step_words_unchanged=True,
        core_step_count_unchanged=True,
        core_replacement_phase_unchanged=True,
        core_learner_key_unchanged=True,
        transaction_sha256="0" * 64,
        development_only=True,
        execution_authorized=False,
        runner_authorized=False,
        campaign_authorized=False,
        artifact_writes_authorized=False,
        threshold_authorized=False,
        evidence_authorized=False,
        scientific_promotion_allowed=False,
    )
    transaction = GeneratedBirthIdentityScrubEpochTransaction(
        active_scrub_mask=_readonly(active_mask),
        candidate_scrub_mask=_readonly(candidate_mask),
        active_scrub_birth_identity=_readonly(active_assignments),
        candidate_scrub_birth_identity=_readonly(candidate_assignments),
        structural_scrub_validation=structural,
        reacquisition_epoch_plan=epoch_plan,
        post_ledger_state=post_ledger,
        audit=audit,
    )
    transaction = dataclasses.replace(
        transaction,
        audit=dataclasses.replace(
            audit,
            transaction_sha256=generated_birth_identity_scrub_epoch_transaction_sha256(
                transaction
            ),
        ),
    )
    return transaction


def build_generated_birth_identity_scrub_epoch_transaction(
    config: GeneratedBirthIdentityLedgerV4Config,
    pre_ledger_state: GeneratedBirthIdentityLedgerV4State,
    pre_core_state: CompositionalFeatureState,
    post_core_state: CompositionalFeatureState,
    target: GeneratedExpression,
    lineage_plan: ExpandedExpressionLineagePlan,
    *,
    lineage_config: ExpandedExpressionLineageConfig,
    scrub_config: GeneratedClassScrubConfig,
    epoch_config: GeneratedReacquisitionEpochConfig,
) -> GeneratedBirthIdentityScrubEpochTransaction:
    """Build one canonical, non-executable external scrub rollover."""

    return _build_transaction(
        config,
        pre_ledger_state,
        pre_core_state,
        post_core_state,
        target,
        lineage_plan,
        lineage_config=lineage_config,
        scrub_config=scrub_config,
        epoch_config=epoch_config,
    )


def validate_generated_birth_identity_scrub_epoch_transaction(
    transaction: GeneratedBirthIdentityScrubEpochTransaction,
    *,
    config: GeneratedBirthIdentityLedgerV4Config,
    pre_ledger_state: GeneratedBirthIdentityLedgerV4State,
    pre_core_state: CompositionalFeatureState,
    post_core_state: CompositionalFeatureState,
    target: GeneratedExpression,
    lineage_plan: ExpandedExpressionLineagePlan,
    lineage_config: ExpandedExpressionLineageConfig,
    scrub_config: GeneratedClassScrubConfig,
    epoch_config: GeneratedReacquisitionEpochConfig,
) -> GeneratedBirthIdentityScrubEpochValidation:
    """Independently rebuild and byte-compare an untrusted rollover."""

    if type(transaction) is not GeneratedBirthIdentityScrubEpochTransaction:
        raise TypeError(
            "transaction must be an exact GeneratedBirthIdentityScrubEpochTransaction"
        )
    canonical = _build_transaction(
        config,
        pre_ledger_state,
        pre_core_state,
        post_core_state,
        target,
        lineage_plan,
        lineage_config=lineage_config,
        scrub_config=scrub_config,
        epoch_config=epoch_config,
    )
    _ledger._validate_v4_state(config, transaction.post_ledger_state)  # noqa: SLF001
    supplied_sha256 = generated_birth_identity_scrub_epoch_transaction_sha256(transaction)
    _require(
        transaction.audit.transaction_sha256 == supplied_sha256,
        "supplied rollover self-hash is stale",
    )
    canonical_bytes = _canonical_json_bytes(
        _transaction_payload(canonical, include_transaction_sha256=True)
    )
    supplied_bytes = _canonical_json_bytes(
        _transaction_payload(transaction, include_transaction_sha256=True)
    )
    _require(
        supplied_bytes == canonical_bytes,
        "rollover differs from the strict independent canonical rebuild",
    )
    _same_exact(
        transaction,
        canonical,
        name="supplied/canonical complete scrub rollover",
    )
    return GeneratedBirthIdentityScrubEpochValidation(
        valid=True,
        canonical_transaction_sha256=canonical.audit.transaction_sha256,
        supplied_transaction_sha256=transaction.audit.transaction_sha256,
        output_is_normal_v4_state=True,
        structural_scrub_valid=True,
        reacquisition_epoch_plan_valid=True,
        development_only=True,
        execution_authorized=False,
        runner_authorized=False,
        campaign_authorized=False,
        artifact_writes_authorized=False,
        threshold_authorized=False,
        evidence_authorized=False,
        scientific_promotion_allowed=False,
    )


def validate_generated_birth_identity_scrub_epoch_transaction_from_inputs(
    transaction: GeneratedBirthIdentityScrubEpochTransaction,
    inputs: GeneratedBirthIdentityScrubEpochInputs,
) -> GeneratedBirthIdentityScrubEpochValidation:
    """Strictly validate a rollover from one immutable complete input bundle."""

    if type(inputs) is not GeneratedBirthIdentityScrubEpochInputs:
        raise TypeError("inputs must be an exact GeneratedBirthIdentityScrubEpochInputs")
    return validate_generated_birth_identity_scrub_epoch_transaction(
        transaction,
        config=inputs.config,
        pre_ledger_state=inputs.pre_ledger_state,
        pre_core_state=inputs.pre_core_state,
        post_core_state=inputs.post_core_state,
        target=inputs.target,
        lineage_plan=inputs.lineage_plan,
        lineage_config=inputs.lineage_config,
        scrub_config=inputs.scrub_config,
        epoch_config=inputs.epoch_config,
    )


__all__ = [
    "GENERATED_BIRTH_IDENTITY_SCRUB_EPOCH_SCHEMA",
    "GENERATED_BIRTH_IDENTITY_SCRUB_EPOCH_STATUS",
    "GeneratedBirthIdentityScrubEpochAudit",
    "GeneratedBirthIdentityScrubEpochError",
    "GeneratedBirthIdentityScrubEpochInputs",
    "GeneratedBirthIdentityScrubEpochTransaction",
    "GeneratedBirthIdentityScrubEpochValidation",
    "build_generated_birth_identity_scrub_epoch_transaction",
    "generated_birth_identity_scrub_epoch_core_state_sha256",
    "generated_birth_identity_scrub_epoch_transaction_sha256",
    "validate_generated_birth_identity_scrub_epoch_transaction",
    "validate_generated_birth_identity_scrub_epoch_transaction_from_inputs",
]
