"""Authenticated identity-chain adapter for one generation-freeze due slot.

The canonical generated-class freeze contains exactly one scheduled curation
opportunity.  This adapter executes and source-replay-authenticates both sides
of that evaluator intervention:

* the exact production learner forms the attempted proposal and a normal v4
  event transaction;
* an exact-base shadow learner whose public config differs only by disabling
  the replacement interval performs the same ordinary online-learning update
  without a structural commit and yields a normal v4 no-event transaction.

The attempted ledger branch is sealed and abandoned.  The shadow branch is the
only carried identity chain.  A trace-mask-selective reconstruction must turn
the attempted state into the shadow state bit-for-bit, proving that ordinary
weight, trace, utility, key, counter, and scheduler updates were retained.  At
the declared freeze endpoint the evaluator's preregistered fresh key is then
bound as the sole change in a separate returned core state; the ledger is
unchanged because it intentionally does not track learner RNG state.

This remains development-only host instrumentation.  It writes no artifact and
grants no runner, campaign, evidence, threshold, or promotion authority.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import struct
from typing import Any, Final, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.core.compositional_features import (
    CompositionalCurationTrace,
    CompositionalFeatureLearner,
    CompositionalFeatureState,
    CompositionalFeatureUpdateResult,
)
from alberta_framework.evaluation import generated_birth_identity_ledger as _ledger
from alberta_framework.evaluation.generated_birth_identity_ledger import (
    GeneratedBirthIdentityLedgerV4Config,
    GeneratedBirthIdentityLedgerV4State,
    validate_generated_birth_identity_transaction_v4,
)
from alberta_framework.evaluation.generated_birth_identity_scrub_epoch import (
    GeneratedBirthIdentityScrubEpochInputs,
    GeneratedBirthIdentityScrubEpochTransaction,
    generated_birth_identity_scrub_epoch_core_state_sha256,
    validate_generated_birth_identity_scrub_epoch_transaction_from_inputs,
)
from alberta_framework.evaluation.generated_birth_identity_trace_binding import (
    GeneratedBirthIdentityTraceBinding,
    authenticate_generated_birth_identity_trace_by_source_replay,
)
from alberta_framework.evaluation.generated_class_lifecycle_scrub import (
    ACTIVE_MASKED_LEAF_PATHS,
    CANDIDATE_MASKED_LEAF_PATHS,
    CROSS_MASKED_LEAF_PATHS,
    scrub_compositional_feature_state,
)
from alberta_framework.evaluation.generated_reacquisition_epoch import (
    GeneratedReacquisitionEpochConfig,
    GeneratedReacquisitionEpochPlan,
    derive_generated_reacquisition_epoch_key,
)

GENERATED_BIRTH_IDENTITY_FREEZE_SCHEMA: Final = (
    "alberta.generated-birth-identity-generation-freeze.development.v0"
)
GENERATED_BIRTH_IDENTITY_FREEZE_STATUS: Final = (
    "DEVELOPMENT_DUAL_REPLAY_NO_EXECUTION_OR_EVIDENCE_AUTHORITY"
)
_PRNG_IMPL: Final = "threefry2x32"
CAUSAL_FREEZE_ARM: Final = "causal"
MATCHED_SHAM_FREEZE_ARM: Final = "matched_sham"
_ACTIVE_SLOT_AXIS_ONE: Final = frozenset(
    {
        "feature_score_residual_trace",
        "output_weights",
        "utility_contribution_trace",
    }
)
_CANDIDATE_SLOT_AXIS_ONE: Final = frozenset(
    {
        "candidate_output_weights",
        "candidate_score_residual_trace",
        "candidate_utility_contribution_trace",
    }
)


class GeneratedBirthIdentityFreezeError(RuntimeError):
    """Raised when the scheduled freeze transition cannot be authenticated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GeneratedBirthIdentityFreezeError(message)


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


def _typed_key_record(key: object, *, name: str) -> dict[str, object]:
    if not isinstance(key, Array) or not jax.dtypes.issubdtype(  # type: ignore[attr-defined]
        key.dtype,
        jax.dtypes.prng_key,
    ):
        raise TypeError(f"{name} must be a typed JAX key")
    implementation = str(jr.key_impl(key))
    data = np.asarray(jr.key_data(key))
    if implementation != _PRNG_IMPL:
        raise ValueError(f"{name} must use {_PRNG_IMPL}")
    if key.shape != () or data.shape != (2,) or data.dtype != np.uint32:
        raise ValueError(f"{name} must contain exactly two uint32 words")
    return {
        "implementation": implementation,
        "dtype": data.dtype.str,
        "shape": list(data.shape),
        "raw_hex": np.ascontiguousarray(data).tobytes(order="C").hex(),
        "words_uint32": [int(data[0]), int(data[1])],
    }


def _value_record(value: object, *, path: str) -> object:
    if isinstance(value, Array) and jax.dtypes.issubdtype(  # type: ignore[attr-defined]
        value.dtype,
        jax.dtypes.prng_key,
    ):
        return {"kind": "typed-prng", **_typed_key_record(value, name=path)}
    if isinstance(value, Array):
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError(f"object array at {path} is not hashable")
        return {
            "kind": "jax-array",
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "raw_hex": np.ascontiguousarray(array).tobytes(order="C").hex(),
        }
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError(f"object array at {path} is not hashable")
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
                field.name: _value_record(
                    getattr(value, field.name),
                    path=f"{path}.{field.name}",
                )
                for field in dataclasses.fields(value)
            },
        }
    if type(value) is str:
        return {"kind": "str", "value": value}
    if type(value) is bool:
        return {"kind": "bool", "value": value}
    if type(value) is int:
        return {"kind": "int", "value": value}
    if type(value) is float:
        return {"kind": "python-float", "raw_hex": struct.pack(">d", value).hex()}
    if value is None:
        return {"kind": "none"}
    if type(value) is tuple:
        return {
            "kind": "tuple",
            "items": [
                _value_record(item, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            ],
        }
    if type(value) is list:
        return {
            "kind": "list",
            "items": [
                _value_record(item, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            ],
        }
    if type(value) is dict:
        _require(
            all(type(key) is str for key in value),
            f"mapping keys at {path} must be exact strings",
        )
        return {
            "kind": "dict",
            "items": {
                key: _value_record(value[key], path=f"{path}.{key}")
                for key in sorted(cast(dict[str, object], value))
            },
        }
    raise TypeError(f"unsupported hash value at {path}: {type(value)!r}")


def _value_sha256(value: object, *, path: str) -> str:
    return _sha256_json(_value_record(value, path=path))


def _array_bits_equal(left: object, right: object) -> bool:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    return bool(
        left_array.dtype == right_array.dtype
        and left_array.shape == right_array.shape
        and np.ascontiguousarray(left_array).tobytes(order="C")
        == np.ascontiguousarray(right_array).tobytes(order="C")
    )


def _same_value_bits(left: object, right: object, *, name: str) -> None:
    _require(
        _value_record(left, path=f"left.{name}")
        == _value_record(right, path=f"right.{name}"),
        f"{name} differs in type, shape, dtype, PRNG implementation, or raw bits",
    )


def _scalar_i32(value: object, *, name: str) -> int:
    array = np.asarray(value)
    if array.shape != () or array.dtype != np.int32:
        raise TypeError(f"{name} must be a scalar int32 array")
    return int(array)


def _scalar_bool(value: object, *, name: str) -> bool:
    array = np.asarray(value)
    if array.shape != () or array.dtype != np.bool_:
        raise TypeError(f"{name} must be a scalar bool array")
    return bool(array)


def _step_words(value: object, *, name: str) -> np.ndarray[Any, Any]:
    array = np.asarray(value)
    if array.shape != (2,) or array.dtype != np.uint32:
        raise TypeError(f"{name} must be an exact uint32[2] array")
    return np.ascontiguousarray(array)


def _words_integer(words: np.ndarray[Any, Any]) -> int:
    return (int(words[0]) << 32) | int(words[1])


def _validate_core_ledger_descriptors(
    config: GeneratedBirthIdentityLedgerV4Config,
    ledger_state: GeneratedBirthIdentityLedgerV4State,
    core_state: CompositionalFeatureState,
) -> None:
    _ledger._validate_v4_state(config, ledger_state)  # noqa: SLF001
    for ledger_name, core_name in (
        ("active_parent_a", "parent_a"),
        ("active_parent_b", "parent_b"),
        ("active_ops", "ops"),
        ("active_depth", "depth"),
        ("active_generator_policy", "feature_generator_policy"),
        ("candidate_parent_a", "candidate_parent_a"),
        ("candidate_parent_b", "candidate_parent_b"),
        ("candidate_ops", "candidate_ops"),
        ("candidate_depth", "candidate_depth"),
        ("candidate_generator_policy", "candidate_generator_policy"),
    ):
        _require(
            _array_bits_equal(
                getattr(ledger_state, ledger_name),
                getattr(core_state, core_name),
            ),
            f"ledger/core descriptor mismatch for {ledger_name}",
        )
    _require(
        _array_bits_equal(ledger_state.step_words, core_state.step_words),
        "ledger/core lifetime words differ",
    )


def _mask(trace_value: object, *, size: int, name: str) -> np.ndarray[Any, Any]:
    array = np.asarray(trace_value)
    if array.shape != (size,) or array.dtype != np.bool_:
        raise TypeError(f"{name} must be an exact bool[{size}] array")
    return np.ascontiguousarray(array)


def _masked_merge_axis(
    attempted: Array,
    shadow: Array,
    mask: np.ndarray[Any, Any],
    *,
    axis: int,
    name: str,
) -> Array:
    attempted_array = np.asarray(attempted)
    shadow_array = np.asarray(shadow)
    _require(
        attempted_array.dtype == shadow_array.dtype
        and attempted_array.shape == shadow_array.shape,
        f"{name} attempted/shadow shape or dtype differs",
    )
    _require(
        attempted_array.shape[axis] == mask.shape[0],
        f"{name} curation mask axis is stale",
    )
    unmasked_indices = np.flatnonzero(~mask)
    if unmasked_indices.size:
        _require(
            _array_bits_equal(
                np.take(attempted_array, unmasked_indices, axis=axis),
                np.take(shadow_array, unmasked_indices, axis=axis),
            ),
            f"{name} differs outside trace-selected curation slots",
        )
    shape = [1] * attempted_array.ndim
    shape[axis] = mask.shape[0]
    selector = mask.reshape(shape)
    return jnp.asarray(np.where(selector, shadow_array, attempted_array))


def _selective_rollback_state(
    attempted: CompositionalFeatureState,
    shadow: CompositionalFeatureState,
    trace: CompositionalCurationTrace,
    *,
    active_slots: int,
    candidate_slots: int,
) -> tuple[CompositionalFeatureState, np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Merge only trace-selected curation cells and reject every other delta."""

    active_mask = _mask(
        trace.active_change_mask,
        size=active_slots,
        name="trace.active_change_mask",
    )
    candidate_mask = np.zeros((candidate_slots,), dtype=np.bool_)
    for name in (
        "candidate_refresh_mask",
        "candidate_rebound_mask",
        "candidate_overdepth_regeneration_mask",
    ):
        candidate_mask |= _mask(
            getattr(trace, name),
            size=candidate_slots,
            name=f"trace.{name}",
        )
    _require(
        bool(np.any(active_mask) or np.any(candidate_mask)),
        "attempted event selected no structural curation slots",
    )

    changes: dict[str, object] = {}
    for field in dataclasses.fields(CompositionalFeatureState):  # type: ignore[arg-type]
        name = field.name
        attempted_value = getattr(attempted, name)
        shadow_value = getattr(shadow, name)
        if name in ACTIVE_MASKED_LEAF_PATHS:
            axis = 1 if name in _ACTIVE_SLOT_AXIS_ONE else 0
            changes[name] = _masked_merge_axis(
                attempted_value,
                shadow_value,
                active_mask,
                axis=axis,
                name=name,
            )
        elif name in CANDIDATE_MASKED_LEAF_PATHS:
            axis = 1 if name in _CANDIDATE_SLOT_AXIS_ONE else 0
            changes[name] = _masked_merge_axis(
                attempted_value,
                shadow_value,
                candidate_mask,
                axis=axis,
                name=name,
            )
        elif name in CROSS_MASKED_LEAF_PATHS:
            attempted_array = np.asarray(attempted_value)
            shadow_array = np.asarray(shadow_value)
            expected_shape = (candidate_slots, active_slots)
            _require(
                attempted_array.dtype == shadow_array.dtype
                and attempted_array.shape == shadow_array.shape == expected_shape,
                f"{name} shape or dtype differs",
            )
            selector = candidate_mask[:, None] | active_mask[None, :]
            _require(
                _array_bits_equal(attempted_array[~selector], shadow_array[~selector]),
                f"{name} differs outside trace-selected rows/columns",
            )
            changes[name] = jnp.asarray(
                np.where(selector, shadow_array, attempted_array)
            )
        else:
            _same_value_bits(
                attempted_value,
                shadow_value,
                name=f"non-curation state leaf {name}",
            )
            changes[name] = attempted_value
    reconstructed = cast(
        CompositionalFeatureState,
        attempted.replace(**changes),  # type: ignore[attr-defined]
    )
    _same_value_bits(
        reconstructed,
        shadow,
        name="selective rollback reconstruction/shadow committed state",
    )
    return reconstructed, active_mask, candidate_mask


def _validate_scrub_rollover_receipt(
    transaction: GeneratedBirthIdentityScrubEpochTransaction,
    inputs: GeneratedBirthIdentityScrubEpochInputs,
    config: GeneratedBirthIdentityLedgerV4Config,
) -> GeneratedReacquisitionEpochPlan:
    if type(transaction) is not GeneratedBirthIdentityScrubEpochTransaction:
        raise TypeError(
            "scrub_rollover must be an exact GeneratedBirthIdentityScrubEpochTransaction"
        )
    if type(inputs) is not GeneratedBirthIdentityScrubEpochInputs:
        raise TypeError("scrub_inputs must be exact GeneratedBirthIdentityScrubEpochInputs")
    _require(inputs.config == config, "scrub validation config differs from freeze config")
    strict = validate_generated_birth_identity_scrub_epoch_transaction_from_inputs(
        transaction,
        inputs,
    )
    _require(strict.valid, "strict scrub rollover validation rejected")
    plan = transaction.reacquisition_epoch_plan
    if type(plan) is not GeneratedReacquisitionEpochPlan:
        raise TypeError("scrub rollover epoch plan type is invalid")
    contract_payload = dataclasses.asdict(plan.contract)
    contract_sha256 = cast(str, contract_payload.pop("contract_sha256"))
    _require(
        contract_sha256 == _sha256_json(contract_payload),
        "reacquisition epoch contract self-hash is stale",
    )
    epoch_config = GeneratedReacquisitionEpochConfig(
        paired_life_seed=plan.contract.paired_life_seed
    )
    expected_key = derive_generated_reacquisition_epoch_key(
        epoch_config.reacquisition_epoch_counter,
        config=epoch_config,
    )
    _same_value_bits(plan.fresh_learner_key, expected_key, name="fresh epoch key")
    return plan


def _validate_freeze_position(
    plan: GeneratedReacquisitionEpochPlan,
    freeze_start_core_state: CompositionalFeatureState,
    freeze_start_ledger_state: GeneratedBirthIdentityLedgerV4State,
    ledger_pre_state: GeneratedBirthIdentityLedgerV4State,
    learner_pre_state: CompositionalFeatureState,
    learner: CompositionalFeatureLearner,
) -> tuple[int, int, int, int, int]:
    contract = plan.contract
    interval_value = learner.to_config().get("replacement_interval")
    _require(
        type(interval_value) is int and interval_value > 0,
        "base replacement interval is invalid",
    )
    interval = cast(int, interval_value)
    _require(interval == contract.curation_interval, "base cadence differs from epoch contract")
    _require(
        contract.generation_write_freeze_updates == interval,
        "freeze window and curation interval differ",
    )
    _require(
        contract.scheduled_curation_decision_slots_in_freeze == 1,
        "epoch contract does not contain exactly one scheduled curation slot",
    )
    start = contract.generation_write_freeze_start_state_step_count
    end = contract.generation_write_freeze_end_state_step_count
    _require(end - start == interval, "freeze endpoint does not match interval")
    start_phase = _scalar_i32(
        freeze_start_core_state.replacement_phase,
        name="freeze-start replacement_phase",
    )
    _require(0 <= start_phase < interval, "post-scrub phase is outside the cadence")
    updates_until_due = interval - start_phase
    due_pre_step = start + updates_until_due - 1
    due_post_step = due_pre_step + 1
    _require(
        start < due_post_step <= end,
        "phase-derived due slot is outside the freeze window",
    )
    pre_step = _scalar_i32(learner_pre_state.step_count, name="learner_pre_state.step_count")
    _require(pre_step == due_pre_step, "freeze adapter is not at the phase-derived due slot")
    _require(
        _scalar_i32(
            learner_pre_state.replacement_phase,
            name="learner_pre_state.replacement_phase",
        )
        == interval - 1,
        "freeze due pre-state is not at the exact bounded phase boundary",
    )
    start_words = _step_words(
        freeze_start_ledger_state.step_words,
        name="freeze-start ledger step_words",
    )
    pre_words = _step_words(ledger_pre_state.step_words, name="ledger pre step_words")
    core_pre_words = _step_words(
        learner_pre_state.step_words,
        name="learner pre step_words",
    )
    _require(
        _array_bits_equal(pre_words, core_pre_words),
        "freeze ledger/core pre words differ",
    )
    _require(
        _words_integer(pre_words) - _words_integer(start_words) == updates_until_due - 1,
        "freeze prefix did not reach the exact phase-derived due coordinate",
    )
    _require(
        _array_bits_equal(
            ledger_pre_state.active_identity,
            freeze_start_ledger_state.active_identity,
        )
        and _array_bits_equal(
            ledger_pre_state.candidate_identity,
            freeze_start_ledger_state.candidate_identity,
        ),
        "freeze prefix changed a live identity before the unique due slot",
    )
    for name in (
        "active_parent_a",
        "active_parent_b",
        "active_ops",
        "active_depth",
        "active_generator_policy",
        "candidate_parent_a",
        "candidate_parent_b",
        "candidate_ops",
        "candidate_depth",
        "candidate_generator_policy",
    ):
        _require(
            _array_bits_equal(
                getattr(ledger_pre_state, name),
                getattr(freeze_start_ledger_state, name),
            ),
            f"freeze prefix changed descriptor bank {name} before the unique due slot",
        )
    return start, end, due_pre_step, due_post_step, updates_until_due


def _validate_ordinary_no_event_chain(
    learner: CompositionalFeatureLearner,
    config: GeneratedBirthIdentityLedgerV4Config,
    steps: tuple[GeneratedBirthIdentityFreezeOrdinaryStep, ...],
    *,
    start_core_state: CompositionalFeatureState,
    start_ledger_state: GeneratedBirthIdentityLedgerV4State,
    expected_end_core_state: CompositionalFeatureState,
    expected_end_ledger_state: GeneratedBirthIdentityLedgerV4State,
    expected_count: int,
    name: str,
) -> str:
    """Source-replay every ordinary step and bind the complete core/ledger chain."""

    if type(steps) is not tuple or not all(
        type(step) is GeneratedBirthIdentityFreezeOrdinaryStep for step in steps
    ):
        raise TypeError(f"{name} steps must be an exact tuple of exact step receipts")
    _require(len(steps) == expected_count, f"{name} step count is stale")
    current_core = start_core_state
    current_ledger = start_ledger_state
    for index, step in enumerate(steps):
        _same_value_bits(
            step.learner_pre_state,
            current_core,
            name=f"{name}[{index}] pre-state chain",
        )
        result = step.supplied_update_result
        if type(result) is not CompositionalFeatureUpdateResult:
            raise TypeError(f"{name}[{index}] result type is invalid")
        canonical_binding = authenticate_generated_birth_identity_trace_by_source_replay(
            learner,
            config,
            current_ledger,
            learner_pre_state=current_core,
            learner_post_state=result.state,
            supplied_update_result=result,
            observation=step.observation,
            targets=step.targets,
            context_id=step.context_id,
        )
        _same_value_bits(
            step.binding,
            canonical_binding,
            name=f"{name}[{index}] supplied/canonical source binding",
        )
        _require(
            canonical_binding.source_replay_authenticated
            and not _scalar_bool(
                result.curation_trace.should_try_replace,
                name=f"{name}[{index}] should_try_replace",
            )
            and not _scalar_bool(
                result.curation_trace.has_event,
                name=f"{name}[{index}] has_event",
            ),
            f"{name}[{index}] is not an authenticated ordinary no-event update",
        )
        current_core = result.state
        current_ledger = canonical_binding.transaction.post_state
    _same_value_bits(current_core, expected_end_core_state, name=f"{name} end core state")
    _same_value_bits(
        current_ledger,
        expected_end_ledger_state,
        name=f"{name} end ledger state",
    )
    return _value_sha256(steps, path=name)


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedBirthIdentityFreezeWorkAccounting:
    """Exact logical calls for one builder plus one independent validator."""

    production_attempt_calls_per_build: int
    production_attempt_source_replay_calls_per_build: int
    shadow_commit_calls_per_build: int
    shadow_commit_source_replay_calls_per_build: int
    supplied_prefix_update_results_per_build: int
    prefix_receipt_direct_calls: int
    prefix_receipt_source_replay_calls: int
    prefix_source_replay_calls_per_build: int
    learner_update_calls_per_build: int
    learner_update_calls_per_independent_validation: int
    total_learner_update_calls_for_validated_transaction: int
    trace_bindings_validated_per_build: int
    selective_rollback_reconstructions_per_build: int
    matched_sham_scrub_kernel_calls_before_due: int
    matched_sham_scrub_kernel_calls_per_build: int
    matched_sham_scrub_kernel_calls_per_independent_validation: int
    total_matched_sham_scrub_kernel_calls: int
    matched_sham_required_total_learner_update_calls: int
    matched_sham_work_parity_required: bool
    operation_accounting_scope: str


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedBirthIdentityFreezeAudit:
    """Hash-bound attempted/abandoned and committed/carried branch receipt."""

    schema: str
    status: str
    arm_mode: str
    matched_sham_start_sha256: str | None
    matched_sham_scrub_work_executed: bool
    scrub_rollover_transaction_sha256: str
    reacquisition_contract_sha256: str
    base_config_sha256: str
    shadow_config_sha256: str
    configs_differ_only_in_replacement_interval: bool
    shared_observation_sha256: str
    shared_targets_sha256: str
    shared_context_id: int
    shared_core_source_sha256: str
    shared_ledger_source_sha256: str
    freeze_start_step: int
    freeze_end_step: int
    phase_derived_due_pre_step: int
    phase_derived_due_post_step: int
    prefix_update_count: int
    suffix_update_count: int
    scheduled_curation_slots: int
    prefix_chain_sha256: str
    prefix_every_core_and_ledger_bit_source_replayed: bool
    base_pre_state_sha256: str
    shadow_pre_state_sha256: str
    phase_normalization_only_pre_state_change: bool
    attempted_result_sha256: str
    shadow_result_sha256: str
    selective_rollback_state_sha256: str
    committed_due_core_state_sha256: str
    planned_fresh_key_data_uint32: tuple[int, int]
    attempted_ledger_transaction_sha256: str
    carried_ledger_transaction_sha256: str
    attempted_branch_abandoned: bool
    shadow_no_event_branch_carried: bool
    attempted_event_authenticated: bool
    shadow_no_event_authenticated: bool
    active_rollback_count: int
    candidate_rollback_count: int
    ordinary_learning_preserved_bit_exactly: bool
    work: GeneratedBirthIdentityFreezeWorkAccounting
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
class GeneratedBirthIdentityFreezeTransaction:
    """Complete dual replay plus the state/ledger pair a runner must carry."""

    attempted_result: CompositionalFeatureUpdateResult
    shadow_pre_state: CompositionalFeatureState
    shadow_result: CompositionalFeatureUpdateResult
    selective_rollback_state: CompositionalFeatureState
    attempted_binding: GeneratedBirthIdentityTraceBinding
    carried_binding: GeneratedBirthIdentityTraceBinding
    attempted_abandoned_ledger_state: GeneratedBirthIdentityLedgerV4State
    carried_ledger_state: GeneratedBirthIdentityLedgerV4State
    committed_core_state: CompositionalFeatureState
    audit: GeneratedBirthIdentityFreezeAudit


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedBirthIdentityFreezeValidation:
    """Independent exact reconstruction result for an untrusted freeze receipt."""

    valid: bool
    canonical_transaction_sha256: str
    supplied_transaction_sha256: str
    attempted_event_branch_validated_and_abandoned: bool
    shadow_no_event_branch_validated_and_carried: bool
    selective_rollback_bit_exact: bool
    output_core_ledger_pair_ready_for_suffix: bool
    total_learner_update_calls_accounted: int
    total_matched_sham_scrub_kernel_calls_accounted: int
    matched_sham_work_parity_required: bool
    development_only: bool
    execution_authorized: bool
    runner_authorized: bool
    campaign_authorized: bool
    artifact_writes_authorized: bool
    threshold_authorized: bool
    evidence_authorized: bool
    scientific_promotion_allowed: bool


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedBirthIdentityMatchedShamStartAudit:
    """Exact matched sham scrub execution bound to the causal epoch plan."""

    schema: str
    status: str
    causal_scrub_transaction_sha256: str
    reacquisition_contract_sha256: str
    start_core_state_sha256: str
    start_ledger_state_sha256: str
    matched_sham_scrub_executed: bool
    matched_sham_scrub_commit_requested: bool
    matched_sham_scrub_noop_validated: bool
    causal_and_sham_start_coordinates_equal: bool
    sham_live_branch_untouched: bool
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
class GeneratedBirthIdentityMatchedShamStart:
    """Unchanged sham core/ledger start plus the causal arm's fresh epoch plan."""

    start_core_state: CompositionalFeatureState
    start_ledger_state: GeneratedBirthIdentityLedgerV4State
    epoch_plan: GeneratedReacquisitionEpochPlan
    audit: GeneratedBirthIdentityMatchedShamStartAudit


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedBirthIdentityFreezeOrdinaryStep:
    """Raw inputs/result plus the supplied source-authenticated no-event binding."""

    learner_pre_state: CompositionalFeatureState
    supplied_update_result: CompositionalFeatureUpdateResult
    observation: Array
    targets: Array
    binding: GeneratedBirthIdentityTraceBinding
    context_id: Array | int = 0


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedBirthIdentityFreezeDueInputs:
    """Complete original inputs required to revalidate prefix and dual due replay."""

    learner: CompositionalFeatureLearner
    config: GeneratedBirthIdentityLedgerV4Config
    ledger_pre_state: GeneratedBirthIdentityLedgerV4State
    learner_pre_state: CompositionalFeatureState
    observation: Array
    targets: Array
    scrub_rollover: GeneratedBirthIdentityScrubEpochTransaction
    scrub_inputs: GeneratedBirthIdentityScrubEpochInputs
    prefix_steps: tuple[GeneratedBirthIdentityFreezeOrdinaryStep, ...]
    matched_sham_start: GeneratedBirthIdentityMatchedShamStart | None = None
    context_id: Array | int = 0


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedBirthIdentityFreezeEndpointInputs:
    """Complete inputs needed to revalidate an endpoint and its suffix."""

    due_transaction: GeneratedBirthIdentityFreezeTransaction
    due_inputs: GeneratedBirthIdentityFreezeDueInputs
    freeze_end_core_state: CompositionalFeatureState
    freeze_end_ledger_state: GeneratedBirthIdentityLedgerV4State
    suffix_steps: tuple[GeneratedBirthIdentityFreezeOrdinaryStep, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedBirthIdentityFreezeEndpointAudit:
    """Exact suffix coordinate and fresh-key-only endpoint intervention."""

    schema: str
    status: str
    due_transaction_sha256: str
    scrub_rollover_transaction_sha256: str
    reacquisition_contract_sha256: str
    freeze_start_step: int
    due_post_step: int
    freeze_end_step: int
    suffix_update_count: int
    freeze_end_core_state_sha256: str
    freeze_end_ledger_state_sha256: str
    fresh_key_applied_core_state_sha256: str
    fresh_key_data_uint32: tuple[int, int]
    due_transaction_strictly_revalidated: bool
    suffix_chain_sha256: str
    suffix_every_core_and_ledger_bit_source_replayed: bool
    suffix_identities_unchanged: bool
    suffix_descriptors_unchanged: bool
    suffix_words_exact: bool
    fresh_key_is_only_endpoint_state_change: bool
    endpoint_core_ledger_pair_ready_for_next_trace: bool
    previously_validated_due_transaction_learner_update_calls: int
    previously_validated_due_matched_sham_scrub_kernel_calls: int
    learner_update_calls_for_due_revalidation_per_build: int
    matched_sham_scrub_kernel_calls_for_due_revalidation_per_build: int
    supplied_suffix_update_results_per_build: int
    suffix_receipt_direct_calls: int
    suffix_receipt_source_replay_calls: int
    suffix_source_replay_calls_per_build: int
    learner_update_calls_per_endpoint_build: int
    learner_update_calls_per_independent_endpoint_validation: int
    total_learner_update_calls_for_validated_endpoint: int
    matched_sham_scrub_kernel_calls_per_independent_endpoint_validation: int
    total_matched_sham_scrub_kernel_calls_for_validated_endpoint: int
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
class GeneratedBirthIdentityFreezeEndpointTransaction:
    """Freeze-end state, unchanged ledger, and exact fresh-key-applied state."""

    freeze_end_core_state: CompositionalFeatureState
    carried_ledger_state: GeneratedBirthIdentityLedgerV4State
    fresh_key_applied_core_state: CompositionalFeatureState
    audit: GeneratedBirthIdentityFreezeEndpointAudit


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedBirthIdentityFreezeEndpointValidation:
    """Independent endpoint reconstruction with all authority held closed."""

    valid: bool
    canonical_transaction_sha256: str
    supplied_transaction_sha256: str
    due_transaction_strictly_revalidated: bool
    suffix_chain_structurally_continuous: bool
    fresh_key_application_bit_exact: bool
    output_core_ledger_pair_ready_for_next_trace: bool
    total_learner_update_calls_accounted: int
    total_matched_sham_scrub_kernel_calls_accounted: int
    development_only: bool
    execution_authorized: bool
    runner_authorized: bool
    campaign_authorized: bool
    artifact_writes_authorized: bool
    threshold_authorized: bool
    evidence_authorized: bool
    scientific_promotion_allowed: bool


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedBirthIdentityPairedFreezeAudit:
    """Actual causal and discarded sham paths with learner-update/replay parity."""

    schema: str
    status: str
    causal_endpoint_transaction_sha256: str
    sham_endpoint_transaction_sha256: str
    causal_due_transaction_sha256: str
    sham_due_transaction_sha256: str
    shared_causal_scrub_transaction_sha256: str
    shared_reacquisition_contract_sha256: str
    shared_learner_config_sha256: str
    shared_ledger_config_sha256: str
    shared_core_source_sha256: str
    shared_ledger_source_sha256: str
    shared_due_observation_sha256: str
    shared_due_targets_sha256: str
    shared_due_context_sha256: str
    shared_freeze_start_key_sha256: str
    shared_due_pre_key_sha256: str
    shared_freeze_end_pre_fresh_key_sha256: str
    shared_prefix_input_manifest_sha256: str
    shared_suffix_input_manifest_sha256: str
    exact_crn_input_parity: bool
    causal_arm_mode: str
    sham_arm_mode: str
    fresh_key_data_uint32: tuple[int, int]
    causal_endpoint_learner_update_calls_before_pairing: int
    sham_endpoint_learner_update_calls_before_pairing: int
    causal_endpoint_revalidation_calls_per_paired_build: int
    sham_endpoint_revalidation_calls_per_paired_build: int
    causal_endpoint_revalidation_calls_per_independent_paired_validation: int
    sham_endpoint_revalidation_calls_per_independent_paired_validation: int
    causal_total_learner_update_calls: int
    sham_total_learner_update_calls: int
    exact_learner_work_parity: bool
    matched_sham_scrub_kernel_work_executed: bool
    matched_sham_scrub_kernel_calls_before_pairing: int
    matched_sham_scrub_kernel_calls_per_paired_build: int
    matched_sham_scrub_kernel_calls_per_independent_paired_validation: int
    total_matched_sham_scrub_kernel_calls_for_validated_pair: int
    operation_accounting_scope: str
    sham_endpoint_state_discarded: bool
    causal_output_core_state_sha256: str
    causal_output_ledger_state_sha256: str
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
class GeneratedBirthIdentityPairedFreezeTransaction:
    """Both validated paths; only the causal endpoint pair is carried."""

    causal_endpoint: GeneratedBirthIdentityFreezeEndpointTransaction
    sham_endpoint: GeneratedBirthIdentityFreezeEndpointTransaction
    causal_output_core_state: CompositionalFeatureState
    causal_output_ledger_state: GeneratedBirthIdentityLedgerV4State
    audit: GeneratedBirthIdentityPairedFreezeAudit


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedBirthIdentityPairedFreezeValidation:
    """Independent paired-path validation with the sham result discarded."""

    valid: bool
    canonical_transaction_sha256: str
    supplied_transaction_sha256: str
    causal_path_strictly_validated: bool
    sham_path_strictly_validated: bool
    exact_learner_work_parity: bool
    matched_sham_work_actually_consumed: bool
    causal_total_learner_update_calls_accounted: int
    sham_total_learner_update_calls_accounted: int
    total_matched_sham_scrub_kernel_calls_accounted: int
    sham_endpoint_state_discarded: bool
    causal_output_ready_for_next_trace: bool
    development_only: bool
    execution_authorized: bool
    runner_authorized: bool
    campaign_authorized: bool
    artifact_writes_authorized: bool
    threshold_authorized: bool
    evidence_authorized: bool
    scientific_promotion_allowed: bool


def _matched_sham_start_payload(
    receipt: GeneratedBirthIdentityMatchedShamStart,
    *,
    include_transaction_sha256: bool,
) -> dict[str, object]:
    audit = dataclasses.asdict(receipt.audit)
    if not include_transaction_sha256:
        audit.pop("transaction_sha256")
    return {
        "start_core_state": _value_record(
            receipt.start_core_state,
            path="matched_sham.start_core_state",
        ),
        "start_ledger_state": _value_record(
            receipt.start_ledger_state,
            path="matched_sham.start_ledger_state",
        ),
        "epoch_plan": _value_record(
            receipt.epoch_plan,
            path="matched_sham.epoch_plan",
        ),
        "audit": audit,
    }


def generated_birth_identity_matched_sham_start_sha256(
    receipt: GeneratedBirthIdentityMatchedShamStart,
) -> str:
    return _sha256_json(
        _matched_sham_start_payload(receipt, include_transaction_sha256=False)
    )


def _build_matched_sham_start(
    causal_scrub: GeneratedBirthIdentityScrubEpochTransaction,
    scrub_inputs: GeneratedBirthIdentityScrubEpochInputs,
) -> GeneratedBirthIdentityMatchedShamStart:
    strict = validate_generated_birth_identity_scrub_epoch_transaction_from_inputs(
        causal_scrub,
        scrub_inputs,
    )
    _require(strict.valid, "causal scrub rejected while constructing matched sham")
    sham = scrub_compositional_feature_state(
        scrub_inputs.pre_core_state,
        scrub_inputs.lineage_plan.active_mask,
        scrub_inputs.lineage_plan.candidate_mask,
        jnp.asarray(False, dtype=jnp.bool_),
        config=scrub_inputs.scrub_config,
    )
    _require(
        _scalar_bool(sham.diagnostics.sham_noop, name="matched sham no-op")
        and not _scalar_bool(
            sham.diagnostics.commit_requested,
            name="matched sham commit_requested",
        )
        and not _scalar_bool(sham.diagnostics.committed, name="matched sham committed"),
        "matched sham scrub did not execute the canonical noncommitting path",
    )
    _same_value_bits(
        sham.state,
        scrub_inputs.pre_core_state,
        name="matched sham returned/pre core state",
    )
    _validate_core_ledger_descriptors(
        scrub_inputs.config,
        scrub_inputs.pre_ledger_state,
        scrub_inputs.pre_core_state,
    )
    causal_start = scrub_inputs.post_core_state
    sham_start = scrub_inputs.pre_core_state
    for field_name in ("step_count", "step_words", "replacement_phase"):
        _same_value_bits(
            getattr(causal_start, field_name),
            getattr(sham_start, field_name),
            name=f"causal/sham start coordinate {field_name}",
        )
    audit = GeneratedBirthIdentityMatchedShamStartAudit(
        schema=GENERATED_BIRTH_IDENTITY_FREEZE_SCHEMA,
        status=GENERATED_BIRTH_IDENTITY_FREEZE_STATUS,
        causal_scrub_transaction_sha256=causal_scrub.audit.transaction_sha256,
        reacquisition_contract_sha256=(
            causal_scrub.reacquisition_epoch_plan.contract.contract_sha256
        ),
        start_core_state_sha256=(
            generated_birth_identity_scrub_epoch_core_state_sha256(sham.state)
        ),
        start_ledger_state_sha256=scrub_inputs.pre_ledger_state.integrity_sha256,
        matched_sham_scrub_executed=True,
        matched_sham_scrub_commit_requested=False,
        matched_sham_scrub_noop_validated=True,
        causal_and_sham_start_coordinates_equal=True,
        sham_live_branch_untouched=True,
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
    receipt = GeneratedBirthIdentityMatchedShamStart(
        start_core_state=sham.state,
        start_ledger_state=scrub_inputs.pre_ledger_state,
        epoch_plan=causal_scrub.reacquisition_epoch_plan,
        audit=audit,
    )
    return dataclasses.replace(
        receipt,
        audit=dataclasses.replace(
            audit,
            transaction_sha256=generated_birth_identity_matched_sham_start_sha256(
                receipt
            ),
        ),
    )


def build_generated_birth_identity_matched_sham_start(
    causal_scrub: GeneratedBirthIdentityScrubEpochTransaction,
    scrub_inputs: GeneratedBirthIdentityScrubEpochInputs,
) -> GeneratedBirthIdentityMatchedShamStart:
    """Execute the matched noncommitting scrub and bind its unchanged start."""

    return _build_matched_sham_start(causal_scrub, scrub_inputs)


def validate_generated_birth_identity_matched_sham_start(
    receipt: GeneratedBirthIdentityMatchedShamStart,
    *,
    causal_scrub: GeneratedBirthIdentityScrubEpochTransaction,
    scrub_inputs: GeneratedBirthIdentityScrubEpochInputs,
) -> None:
    """Re-execute and recursively compare an untrusted matched-sham receipt."""

    if type(receipt) is not GeneratedBirthIdentityMatchedShamStart:
        raise TypeError("receipt must be an exact GeneratedBirthIdentityMatchedShamStart")
    canonical = _build_matched_sham_start(causal_scrub, scrub_inputs)
    _require(
        receipt.audit.transaction_sha256
        == generated_birth_identity_matched_sham_start_sha256(receipt),
        "matched sham start self-hash is stale",
    )
    _same_value_bits(
        receipt,
        canonical,
        name="supplied/canonical matched sham start",
    )


def _binding_record(binding: GeneratedBirthIdentityTraceBinding) -> dict[str, object]:
    return {
        field.name: getattr(binding, field.name)
        for field in dataclasses.fields(binding)
        if field.name not in {"event", "transaction", "ledger_validation"}
    } | {
        "event_sha256": binding.event.integrity_sha256,
        "transaction_sha256": binding.transaction.audit.transaction_sha256,
        "post_ledger_state_sha256": binding.transaction.post_state.integrity_sha256,
        "ledger_validation": dataclasses.asdict(binding.ledger_validation),
    }


def _transaction_payload(
    transaction: GeneratedBirthIdentityFreezeTransaction,
    *,
    include_transaction_sha256: bool,
) -> dict[str, object]:
    audit = dataclasses.asdict(transaction.audit)
    if not include_transaction_sha256:
        audit.pop("transaction_sha256")
    return {
        "attempted_result_sha256": _value_sha256(
            transaction.attempted_result,
            path="attempted_result",
        ),
        "shadow_pre_state_sha256": generated_birth_identity_scrub_epoch_core_state_sha256(
            transaction.shadow_pre_state
        ),
        "shadow_result_sha256": _value_sha256(
            transaction.shadow_result,
            path="shadow_result",
        ),
        "selective_rollback_state_sha256": (
            generated_birth_identity_scrub_epoch_core_state_sha256(
                transaction.selective_rollback_state
            )
        ),
        "attempted_binding": _binding_record(transaction.attempted_binding),
        "carried_binding": _binding_record(transaction.carried_binding),
        "attempted_abandoned_ledger_state_sha256": (
            transaction.attempted_abandoned_ledger_state.integrity_sha256
        ),
        "carried_ledger_state_sha256": transaction.carried_ledger_state.integrity_sha256,
        "committed_core_state_sha256": (
            generated_birth_identity_scrub_epoch_core_state_sha256(
                transaction.committed_core_state
            )
        ),
        "audit": audit,
    }


def generated_birth_identity_freeze_transaction_sha256(
    transaction: GeneratedBirthIdentityFreezeTransaction,
) -> str:
    """Hash every returned branch/state reference except the audit self-hash."""

    return _sha256_json(_transaction_payload(transaction, include_transaction_sha256=False))


def _build_transaction(
    learner: CompositionalFeatureLearner,
    config: GeneratedBirthIdentityLedgerV4Config,
    ledger_pre_state: GeneratedBirthIdentityLedgerV4State,
    learner_pre_state: CompositionalFeatureState,
    observation: Array,
    targets: Array,
    scrub_rollover: GeneratedBirthIdentityScrubEpochTransaction,
    scrub_inputs: GeneratedBirthIdentityScrubEpochInputs,
    prefix_steps: tuple[GeneratedBirthIdentityFreezeOrdinaryStep, ...],
    matched_sham_start: GeneratedBirthIdentityMatchedShamStart | None,
    *,
    context_id: Array | int,
) -> GeneratedBirthIdentityFreezeTransaction:
    if type(learner) is not CompositionalFeatureLearner:
        raise TypeError("learner must be an exact CompositionalFeatureLearner")
    if type(config) is not GeneratedBirthIdentityLedgerV4Config:
        raise TypeError("config must be an exact GeneratedBirthIdentityLedgerV4Config")
    if type(learner_pre_state) is not CompositionalFeatureState:
        raise TypeError("learner_pre_state must be an exact CompositionalFeatureState")
    _require(
        learner.to_config().get("learn_generator_resources") is False
        and config.learn_generator_resources is False,
        "freeze adapter only supports the canonical fixed-phase learner",
    )
    plan = _validate_scrub_rollover_receipt(scrub_rollover, scrub_inputs, config)
    if matched_sham_start is None:
        arm_mode = CAUSAL_FREEZE_ARM
        freeze_start_core_state = scrub_inputs.post_core_state
        freeze_start_ledger_state = scrub_rollover.post_ledger_state
        matched_sham_start_sha256: str | None = None
        matched_sham_scrub_work_executed = False
    else:
        validate_generated_birth_identity_matched_sham_start(
            matched_sham_start,
            causal_scrub=scrub_rollover,
            scrub_inputs=scrub_inputs,
        )
        arm_mode = MATCHED_SHAM_FREEZE_ARM
        freeze_start_core_state = matched_sham_start.start_core_state
        freeze_start_ledger_state = matched_sham_start.start_ledger_state
        matched_sham_start_sha256 = matched_sham_start.audit.transaction_sha256
        matched_sham_scrub_work_executed = True
        _same_value_bits(
            matched_sham_start.epoch_plan,
            plan,
            name="matched sham/causal epoch plan",
        )
    start, end, due_pre_step, due_post_step, updates_until_due = (
        _validate_freeze_position(
            plan,
            freeze_start_core_state,
            freeze_start_ledger_state,
            ledger_pre_state,
            learner_pre_state,
            learner,
        )
    )
    prefix_chain_sha256 = _validate_ordinary_no_event_chain(
        learner,
        config,
        prefix_steps,
        start_core_state=freeze_start_core_state,
        start_ledger_state=freeze_start_ledger_state,
        expected_end_core_state=learner_pre_state,
        expected_end_ledger_state=ledger_pre_state,
        expected_count=updates_until_due - 1,
        name="freeze_prefix",
    )

    base_config = learner.to_config()
    _require(type(base_config) is dict, "learner.to_config() did not return a dictionary")
    shadow_config = dict(base_config)
    shadow_config["replacement_interval"] = 0
    differing_config_fields = {
        name
        for name in base_config
        if _value_record(base_config[name], path=f"base_config.{name}")
        != _value_record(shadow_config[name], path=f"shadow_config.{name}")
    }
    _require(
        differing_config_fields == {"replacement_interval"},
        "base and shadow configs do not differ only in replacement_interval",
    )
    shadow_learner = CompositionalFeatureLearner.from_config(shadow_config)
    _require(
        type(shadow_learner) is CompositionalFeatureLearner,
        "shadow reconstruction did not return the exact base learner class",
    )
    _same_value_bits(
        shadow_learner.to_config(),
        shadow_config,
        name="realized/intended shadow learner config",
    )

    attempted = learner.update(
        learner_pre_state,
        observation,
        targets,
        context_id=context_id,
    )
    attempted.state.step_words.block_until_ready()
    attempted_binding = authenticate_generated_birth_identity_trace_by_source_replay(
        learner,
        config,
        ledger_pre_state,
        learner_pre_state=learner_pre_state,
        learner_post_state=attempted.state,
        supplied_update_result=attempted,
        observation=observation,
        targets=targets,
        context_id=context_id,
    )
    _require(
        attempted_binding.source_replay_authenticated
        and _scalar_bool(attempted.curation_trace.should_try_replace, name="attempted due")
        and _scalar_bool(attempted.curation_trace.has_event, name="attempted event")
        and _scalar_bool(attempted.curation_trace.proposal_formed, name="attempted proposal"),
        "production due branch did not form an authenticated curation event",
    )

    shadow_pre = cast(
        CompositionalFeatureState,
        learner_pre_state.replace(  # type: ignore[attr-defined]
            replacement_phase=jnp.asarray(0, dtype=jnp.int32)
        ),
    )
    for field in dataclasses.fields(CompositionalFeatureState):  # type: ignore[arg-type]
        if field.name == "replacement_phase":
            continue
        _same_value_bits(
            getattr(learner_pre_state, field.name),
            getattr(shadow_pre, field.name),
            name=f"actual/shadow pre {field.name}",
        )
    _require(
        _scalar_i32(shadow_pre.replacement_phase, name="shadow pre phase") == 0,
        "shadow pre phase normalization failed",
    )
    shadow = shadow_learner.update(
        shadow_pre,
        observation,
        targets,
        context_id=context_id,
    )
    shadow.state.step_words.block_until_ready()
    carried_binding = authenticate_generated_birth_identity_trace_by_source_replay(
        shadow_learner,
        config,
        ledger_pre_state,
        learner_pre_state=shadow_pre,
        learner_post_state=shadow.state,
        supplied_update_result=shadow,
        observation=observation,
        targets=targets,
        context_id=context_id,
    )
    _require(
        carried_binding.source_replay_authenticated
        and not _scalar_bool(shadow.curation_trace.should_try_replace, name="shadow due")
        and not _scalar_bool(shadow.curation_trace.has_event, name="shadow event"),
        "shadow branch is not an authenticated no-event transition",
    )
    _require(
        attempted_binding.core_module_sha256 == carried_binding.core_module_sha256
        and attempted_binding.ledger_module_sha256
        == carried_binding.ledger_module_sha256,
        "attempted and shadow branches do not share source bytes",
    )

    reconstructed, active_mask, candidate_mask = _selective_rollback_state(
        attempted.state,
        shadow.state,
        attempted.curation_trace,
        active_slots=config.active_slots,
        candidate_slots=config.candidate_slots,
    )
    _same_value_bits(
        attempted.predictions,
        shadow.predictions,
        name="attempted/shadow pre-update predictions",
    )
    _same_value_bits(
        attempted.errors,
        shadow.errors,
        name="attempted/shadow pre-update errors",
    )
    _require(
        _scalar_i32(shadow.state.step_count, name="freeze due post step_count")
        == due_post_step,
        "shadow committed state did not reach the phase-derived due post coordinate",
    )
    post_words = _step_words(shadow.state.step_words, name="freeze due post step_words")
    pre_words = _step_words(learner_pre_state.step_words, name="freeze pre step_words")
    _require(
        _words_integer(post_words) - _words_integer(pre_words) == 1,
        "freeze due transition did not advance lifetime words exactly once",
    )

    observation_sha256 = _value_sha256(observation, path="observation")
    targets_sha256 = _value_sha256(targets, path="targets")
    context = context_id if type(context_id) is int else _scalar_i32(context_id, name="context_id")
    prefix_count = len(prefix_steps)
    calls_per_build = 4 + prefix_count
    work = GeneratedBirthIdentityFreezeWorkAccounting(
        production_attempt_calls_per_build=1,
        production_attempt_source_replay_calls_per_build=1,
        shadow_commit_calls_per_build=1,
        shadow_commit_source_replay_calls_per_build=1,
        supplied_prefix_update_results_per_build=prefix_count,
        prefix_receipt_direct_calls=prefix_count,
        prefix_receipt_source_replay_calls=prefix_count,
        prefix_source_replay_calls_per_build=prefix_count,
        learner_update_calls_per_build=calls_per_build,
        learner_update_calls_per_independent_validation=calls_per_build,
        total_learner_update_calls_for_validated_transaction=(
            2 * prefix_count + 2 * calls_per_build
        ),
        trace_bindings_validated_per_build=2 + prefix_count,
        selective_rollback_reconstructions_per_build=1,
        matched_sham_scrub_kernel_calls_before_due=(
            1 if matched_sham_scrub_work_executed else 0
        ),
        matched_sham_scrub_kernel_calls_per_build=(
            1 if matched_sham_scrub_work_executed else 0
        ),
        matched_sham_scrub_kernel_calls_per_independent_validation=(
            1 if matched_sham_scrub_work_executed else 0
        ),
        total_matched_sham_scrub_kernel_calls=(
            3 if matched_sham_scrub_work_executed else 0
        ),
        matched_sham_required_total_learner_update_calls=(
            2 * prefix_count + 2 * calls_per_build
        ),
        matched_sham_work_parity_required=True,
        operation_accounting_scope=(
            "authenticated prefix results plus one canonical dual due builder and one "
            "strict independent validator; matched-sham arms must supply and replay the "
            "same number of ordinary results and execute both due branches"
        ),
    )
    fresh_words = cast(
        list[int],
        _typed_key_record(plan.fresh_learner_key, name="fresh learner key")[
            "words_uint32"
        ],
    )
    audit = GeneratedBirthIdentityFreezeAudit(
        schema=GENERATED_BIRTH_IDENTITY_FREEZE_SCHEMA,
        status=GENERATED_BIRTH_IDENTITY_FREEZE_STATUS,
        arm_mode=arm_mode,
        matched_sham_start_sha256=matched_sham_start_sha256,
        matched_sham_scrub_work_executed=matched_sham_scrub_work_executed,
        scrub_rollover_transaction_sha256=scrub_rollover.audit.transaction_sha256,
        reacquisition_contract_sha256=plan.contract.contract_sha256,
        base_config_sha256=_value_sha256(base_config, path="base_config"),
        shadow_config_sha256=_value_sha256(shadow_config, path="shadow_config"),
        configs_differ_only_in_replacement_interval=True,
        shared_observation_sha256=observation_sha256,
        shared_targets_sha256=targets_sha256,
        shared_context_id=context,
        shared_core_source_sha256=attempted_binding.core_module_sha256,
        shared_ledger_source_sha256=attempted_binding.ledger_module_sha256,
        freeze_start_step=start,
        freeze_end_step=end,
        phase_derived_due_pre_step=due_pre_step,
        phase_derived_due_post_step=due_post_step,
        prefix_update_count=updates_until_due - 1,
        suffix_update_count=end - due_post_step,
        scheduled_curation_slots=1,
        prefix_chain_sha256=prefix_chain_sha256,
        prefix_every_core_and_ledger_bit_source_replayed=True,
        base_pre_state_sha256=generated_birth_identity_scrub_epoch_core_state_sha256(
            learner_pre_state
        ),
        shadow_pre_state_sha256=generated_birth_identity_scrub_epoch_core_state_sha256(
            shadow_pre
        ),
        phase_normalization_only_pre_state_change=True,
        attempted_result_sha256=_value_sha256(attempted, path="attempted_result"),
        shadow_result_sha256=_value_sha256(shadow, path="shadow_result"),
        selective_rollback_state_sha256=(
            generated_birth_identity_scrub_epoch_core_state_sha256(reconstructed)
        ),
        committed_due_core_state_sha256=(
            generated_birth_identity_scrub_epoch_core_state_sha256(shadow.state)
        ),
        planned_fresh_key_data_uint32=(fresh_words[0], fresh_words[1]),
        attempted_ledger_transaction_sha256=(
            attempted_binding.transaction.audit.transaction_sha256
        ),
        carried_ledger_transaction_sha256=(
            carried_binding.transaction.audit.transaction_sha256
        ),
        attempted_branch_abandoned=True,
        shadow_no_event_branch_carried=True,
        attempted_event_authenticated=True,
        shadow_no_event_authenticated=True,
        active_rollback_count=int(np.count_nonzero(active_mask)),
        candidate_rollback_count=int(np.count_nonzero(candidate_mask)),
        ordinary_learning_preserved_bit_exactly=True,
        work=work,
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
    transaction = GeneratedBirthIdentityFreezeTransaction(
        attempted_result=attempted,
        shadow_pre_state=shadow_pre,
        shadow_result=shadow,
        selective_rollback_state=reconstructed,
        attempted_binding=attempted_binding,
        carried_binding=carried_binding,
        attempted_abandoned_ledger_state=attempted_binding.transaction.post_state,
        carried_ledger_state=carried_binding.transaction.post_state,
        committed_core_state=shadow.state,
        audit=audit,
    )
    transaction = dataclasses.replace(
        transaction,
        audit=dataclasses.replace(
            audit,
            transaction_sha256=generated_birth_identity_freeze_transaction_sha256(
                transaction
            ),
        ),
    )
    return transaction


def build_generated_birth_identity_freeze_transaction(
    learner: CompositionalFeatureLearner,
    config: GeneratedBirthIdentityLedgerV4Config,
    ledger_pre_state: GeneratedBirthIdentityLedgerV4State,
    learner_pre_state: CompositionalFeatureState,
    observation: Array,
    targets: Array,
    scrub_rollover: GeneratedBirthIdentityScrubEpochTransaction,
    scrub_inputs: GeneratedBirthIdentityScrubEpochInputs,
    prefix_steps: tuple[GeneratedBirthIdentityFreezeOrdinaryStep, ...],
    *,
    matched_sham_start: GeneratedBirthIdentityMatchedShamStart | None = None,
    context_id: Array | int = 0,
) -> GeneratedBirthIdentityFreezeTransaction:
    """Execute and authenticate both branches of the unique freeze due slot."""

    return _build_transaction(
        learner,
        config,
        ledger_pre_state,
        learner_pre_state,
        observation,
        targets,
        scrub_rollover,
        scrub_inputs,
        prefix_steps,
        matched_sham_start,
        context_id=context_id,
    )


def validate_generated_birth_identity_freeze_transaction(
    transaction: GeneratedBirthIdentityFreezeTransaction,
    *,
    learner: CompositionalFeatureLearner,
    config: GeneratedBirthIdentityLedgerV4Config,
    ledger_pre_state: GeneratedBirthIdentityLedgerV4State,
    learner_pre_state: CompositionalFeatureState,
    observation: Array,
    targets: Array,
    scrub_rollover: GeneratedBirthIdentityScrubEpochTransaction,
    scrub_inputs: GeneratedBirthIdentityScrubEpochInputs,
    prefix_steps: tuple[GeneratedBirthIdentityFreezeOrdinaryStep, ...],
    matched_sham_start: GeneratedBirthIdentityMatchedShamStart | None = None,
    context_id: Array | int = 0,
) -> GeneratedBirthIdentityFreezeValidation:
    """Replay both calls again and compare every returned branch/state hash."""

    if type(transaction) is not GeneratedBirthIdentityFreezeTransaction:
        raise TypeError(
            "transaction must be an exact GeneratedBirthIdentityFreezeTransaction"
        )
    canonical = _build_transaction(
        learner,
        config,
        ledger_pre_state,
        learner_pre_state,
        observation,
        targets,
        scrub_rollover,
        scrub_inputs,
        prefix_steps,
        matched_sham_start,
        context_id=context_id,
    )
    for name, binding in (
        ("attempted", transaction.attempted_binding),
        ("carried", transaction.carried_binding),
    ):
        _require(
            type(binding) is GeneratedBirthIdentityTraceBinding
            and binding.source_replay_authenticated,
            f"supplied {name} binding is not exact and source authenticated",
        )
        validation = validate_generated_birth_identity_transaction_v4(
            binding.transaction,
            config=config,
            pre_state=ledger_pre_state,
            event=binding.event,
        )
        _require(validation.valid, f"supplied {name} ledger transaction rejected")
    supplied_sha256 = generated_birth_identity_freeze_transaction_sha256(transaction)
    _require(
        transaction.audit.transaction_sha256 == supplied_sha256,
        "supplied freeze transaction self-hash is stale",
    )
    _require(
        _canonical_json_bytes(
            _transaction_payload(transaction, include_transaction_sha256=True)
        )
        == _canonical_json_bytes(
            _transaction_payload(canonical, include_transaction_sha256=True)
        ),
        "freeze transaction differs from the strict independent canonical rebuild",
    )
    _same_value_bits(
        transaction,
        canonical,
        name="supplied/canonical complete freeze transaction",
    )
    _same_value_bits(
        transaction.committed_core_state,
        transaction.selective_rollback_state,
        name="supplied selective rollback/committed due state",
    )
    _same_value_bits(
        transaction.carried_ledger_state,
        transaction.carried_binding.transaction.post_state,
        name="supplied carried ledger/authenticated shadow branch",
    )
    _same_value_bits(
        transaction.attempted_abandoned_ledger_state,
        transaction.attempted_binding.transaction.post_state,
        name="supplied abandoned ledger/authenticated attempted branch",
    )
    return GeneratedBirthIdentityFreezeValidation(
        valid=True,
        canonical_transaction_sha256=canonical.audit.transaction_sha256,
        supplied_transaction_sha256=transaction.audit.transaction_sha256,
        attempted_event_branch_validated_and_abandoned=True,
        shadow_no_event_branch_validated_and_carried=True,
        selective_rollback_bit_exact=True,
        output_core_ledger_pair_ready_for_suffix=True,
        total_learner_update_calls_accounted=(
            transaction.audit.work.total_learner_update_calls_for_validated_transaction
        ),
        total_matched_sham_scrub_kernel_calls_accounted=(
            transaction.audit.work.total_matched_sham_scrub_kernel_calls
        ),
        matched_sham_work_parity_required=True,
        development_only=True,
        execution_authorized=False,
        runner_authorized=False,
        campaign_authorized=False,
        artifact_writes_authorized=False,
        threshold_authorized=False,
        evidence_authorized=False,
        scientific_promotion_allowed=False,
    )


def _validate_due_transaction_from_inputs(
    transaction: GeneratedBirthIdentityFreezeTransaction,
    inputs: GeneratedBirthIdentityFreezeDueInputs,
) -> GeneratedBirthIdentityFreezeValidation:
    if type(inputs) is not GeneratedBirthIdentityFreezeDueInputs:
        raise TypeError("due_inputs must be exact GeneratedBirthIdentityFreezeDueInputs")
    return validate_generated_birth_identity_freeze_transaction(
        transaction,
        learner=inputs.learner,
        config=inputs.config,
        ledger_pre_state=inputs.ledger_pre_state,
        learner_pre_state=inputs.learner_pre_state,
        observation=inputs.observation,
        targets=inputs.targets,
        scrub_rollover=inputs.scrub_rollover,
        scrub_inputs=inputs.scrub_inputs,
        prefix_steps=inputs.prefix_steps,
        matched_sham_start=inputs.matched_sham_start,
        context_id=inputs.context_id,
    )


def _endpoint_payload(
    transaction: GeneratedBirthIdentityFreezeEndpointTransaction,
    *,
    include_transaction_sha256: bool,
) -> dict[str, object]:
    audit = dataclasses.asdict(transaction.audit)
    if not include_transaction_sha256:
        audit.pop("transaction_sha256")
    return {
        "freeze_end_core_state": _value_record(
            transaction.freeze_end_core_state,
            path="freeze_end_core_state",
        ),
        "carried_ledger_state": _value_record(
            transaction.carried_ledger_state,
            path="carried_ledger_state",
        ),
        "fresh_key_applied_core_state": _value_record(
            transaction.fresh_key_applied_core_state,
            path="fresh_key_applied_core_state",
        ),
        "audit": audit,
    }


def generated_birth_identity_freeze_endpoint_transaction_sha256(
    transaction: GeneratedBirthIdentityFreezeEndpointTransaction,
) -> str:
    """Hash the complete endpoint states and audit excluding its self-hash."""

    return _sha256_json(_endpoint_payload(transaction, include_transaction_sha256=False))


def _build_endpoint_transaction(
    due_transaction: GeneratedBirthIdentityFreezeTransaction,
    due_inputs: GeneratedBirthIdentityFreezeDueInputs,
    freeze_end_core_state: CompositionalFeatureState,
    freeze_end_ledger_state: GeneratedBirthIdentityLedgerV4State,
    suffix_steps: tuple[GeneratedBirthIdentityFreezeOrdinaryStep, ...],
) -> GeneratedBirthIdentityFreezeEndpointTransaction:
    if type(freeze_end_core_state) is not CompositionalFeatureState:
        raise TypeError(
            "freeze_end_core_state must be an exact CompositionalFeatureState"
        )
    due_validation = _validate_due_transaction_from_inputs(due_transaction, due_inputs)
    _require(due_validation.valid, "freeze due transaction strict validation rejected")
    config = due_inputs.config
    scrub_rollover = due_inputs.scrub_rollover
    scrub_inputs = due_inputs.scrub_inputs
    plan = _validate_scrub_rollover_receipt(scrub_rollover, scrub_inputs, config)
    if due_inputs.matched_sham_start is None:
        freeze_start_core_state = scrub_inputs.post_core_state
        freeze_start_ledger_state = scrub_rollover.post_ledger_state
    else:
        # The strict due revalidation above already rebuilt and compared this
        # exact receipt from the same immutable inputs. Do not execute the sham
        # scrub kernel a redundant second time in one endpoint build.
        freeze_start_core_state = due_inputs.matched_sham_start.start_core_state
        freeze_start_ledger_state = due_inputs.matched_sham_start.start_ledger_state
    _validate_core_ledger_descriptors(config, freeze_end_ledger_state, freeze_end_core_state)
    end = plan.contract.generation_write_freeze_end_state_step_count
    end_step = _scalar_i32(freeze_end_core_state.step_count, name="freeze end step_count")
    _require(end_step == end, "endpoint core state is not at the declared freeze endpoint")
    start_words = _step_words(
        freeze_start_ledger_state.step_words,
        name="freeze-start ledger step_words",
    )
    end_words = _step_words(
        freeze_end_ledger_state.step_words,
        name="freeze end ledger step_words",
    )
    _require(
        _words_integer(end_words) - _words_integer(start_words)
        == plan.contract.generation_write_freeze_updates,
        "freeze endpoint words do not cover the exact fixed window",
    )
    start_phase = _scalar_i32(
        freeze_start_core_state.replacement_phase,
        name="freeze-start replacement_phase",
    )
    _require(
        _scalar_i32(
            freeze_end_core_state.replacement_phase,
            name="freeze end replacement_phase",
        )
        == start_phase,
        "full-interval freeze endpoint did not return to its start phase",
    )
    for bank in ("active_identity", "candidate_identity"):
        _require(
            _array_bits_equal(
                getattr(freeze_end_ledger_state, bank),
                getattr(due_transaction.carried_ledger_state, bank),
            ),
            f"freeze suffix changed {bank}",
        )
    for name in (
        "active_parent_a",
        "active_parent_b",
        "active_ops",
        "active_depth",
        "active_generator_policy",
        "candidate_parent_a",
        "candidate_parent_b",
        "candidate_ops",
        "candidate_depth",
        "candidate_generator_policy",
    ):
        _require(
            _array_bits_equal(
                getattr(freeze_end_ledger_state, name),
                getattr(due_transaction.carried_ledger_state, name),
            ),
            f"freeze suffix changed descriptor bank {name}",
        )
    due_post_step = due_transaction.audit.phase_derived_due_post_step
    suffix_count = end - due_post_step
    suffix_chain_sha256 = _validate_ordinary_no_event_chain(
        due_inputs.learner,
        config,
        suffix_steps,
        start_core_state=due_transaction.committed_core_state,
        start_ledger_state=due_transaction.carried_ledger_state,
        expected_end_core_state=freeze_end_core_state,
        expected_end_ledger_state=freeze_end_ledger_state,
        expected_count=suffix_count,
        name="freeze_suffix",
    )
    due_words = _step_words(
        due_transaction.carried_ledger_state.step_words,
        name="due carried ledger step_words",
    )
    _require(
        _words_integer(end_words) - _words_integer(due_words) == suffix_count,
        "freeze suffix word count is stale",
    )

    _require(
        _typed_key_record(
            freeze_end_core_state.key,
            name="freeze end learner key",
        )
        != _typed_key_record(plan.fresh_learner_key, name="planned fresh learner key"),
        "freeze endpoint already carries the planned fresh key",
    )
    fresh_state = cast(
        CompositionalFeatureState,
        freeze_end_core_state.replace(  # type: ignore[attr-defined]
            key=plan.fresh_learner_key
        ),
    )
    for field in dataclasses.fields(CompositionalFeatureState):  # type: ignore[arg-type]
        if field.name == "key":
            continue
        _same_value_bits(
            getattr(freeze_end_core_state, field.name),
            getattr(fresh_state, field.name),
            name=f"fresh endpoint {field.name}",
        )
    _same_value_bits(
        fresh_state.key,
        plan.fresh_learner_key,
        name="fresh endpoint learner key",
    )
    fresh_words = cast(
        list[int],
        _typed_key_record(plan.fresh_learner_key, name="fresh endpoint key")[
            "words_uint32"
        ],
    )
    due_revalidation_calls = (
        due_transaction.audit.work.learner_update_calls_per_build
    )
    endpoint_build_calls = due_revalidation_calls + suffix_count
    prior_due_calls = (
        due_transaction.audit.work.total_learner_update_calls_for_validated_transaction
    )
    prior_due_sham_scrub_calls = (
        due_transaction.audit.work.total_matched_sham_scrub_kernel_calls
    )
    due_revalidation_sham_scrub_calls = (
        1 if due_transaction.audit.arm_mode == MATCHED_SHAM_FREEZE_ARM else 0
    )
    total_endpoint_calls = (
        prior_due_calls
        + 2 * suffix_count
        + 2 * endpoint_build_calls
    )
    total_endpoint_sham_scrub_calls = (
        prior_due_sham_scrub_calls + 2 * due_revalidation_sham_scrub_calls
    )
    audit = GeneratedBirthIdentityFreezeEndpointAudit(
        schema=GENERATED_BIRTH_IDENTITY_FREEZE_SCHEMA,
        status=GENERATED_BIRTH_IDENTITY_FREEZE_STATUS,
        due_transaction_sha256=due_transaction.audit.transaction_sha256,
        scrub_rollover_transaction_sha256=scrub_rollover.audit.transaction_sha256,
        reacquisition_contract_sha256=plan.contract.contract_sha256,
        freeze_start_step=plan.contract.generation_write_freeze_start_state_step_count,
        due_post_step=due_post_step,
        freeze_end_step=end,
        suffix_update_count=suffix_count,
        freeze_end_core_state_sha256=(
            generated_birth_identity_scrub_epoch_core_state_sha256(
                freeze_end_core_state
            )
        ),
        freeze_end_ledger_state_sha256=freeze_end_ledger_state.integrity_sha256,
        fresh_key_applied_core_state_sha256=(
            generated_birth_identity_scrub_epoch_core_state_sha256(fresh_state)
        ),
        fresh_key_data_uint32=(fresh_words[0], fresh_words[1]),
        due_transaction_strictly_revalidated=True,
        suffix_chain_sha256=suffix_chain_sha256,
        suffix_every_core_and_ledger_bit_source_replayed=True,
        suffix_identities_unchanged=True,
        suffix_descriptors_unchanged=True,
        suffix_words_exact=True,
        fresh_key_is_only_endpoint_state_change=True,
        endpoint_core_ledger_pair_ready_for_next_trace=True,
        previously_validated_due_transaction_learner_update_calls=prior_due_calls,
        previously_validated_due_matched_sham_scrub_kernel_calls=(
            prior_due_sham_scrub_calls
        ),
        learner_update_calls_for_due_revalidation_per_build=due_revalidation_calls,
        matched_sham_scrub_kernel_calls_for_due_revalidation_per_build=(
            due_revalidation_sham_scrub_calls
        ),
        supplied_suffix_update_results_per_build=suffix_count,
        suffix_receipt_direct_calls=suffix_count,
        suffix_receipt_source_replay_calls=suffix_count,
        suffix_source_replay_calls_per_build=suffix_count,
        learner_update_calls_per_endpoint_build=endpoint_build_calls,
        learner_update_calls_per_independent_endpoint_validation=(
            endpoint_build_calls
        ),
        total_learner_update_calls_for_validated_endpoint=total_endpoint_calls,
        matched_sham_scrub_kernel_calls_per_independent_endpoint_validation=(
            due_revalidation_sham_scrub_calls
        ),
        total_matched_sham_scrub_kernel_calls_for_validated_endpoint=(
            total_endpoint_sham_scrub_calls
        ),
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
    transaction = GeneratedBirthIdentityFreezeEndpointTransaction(
        freeze_end_core_state=freeze_end_core_state,
        carried_ledger_state=freeze_end_ledger_state,
        fresh_key_applied_core_state=fresh_state,
        audit=audit,
    )
    return dataclasses.replace(
        transaction,
        audit=dataclasses.replace(
            audit,
            transaction_sha256=(
                generated_birth_identity_freeze_endpoint_transaction_sha256(
                    transaction
                )
            ),
        ),
    )


def build_generated_birth_identity_freeze_endpoint_transaction(
    due_transaction: GeneratedBirthIdentityFreezeTransaction,
    due_inputs: GeneratedBirthIdentityFreezeDueInputs,
    freeze_end_core_state: CompositionalFeatureState,
    freeze_end_ledger_state: GeneratedBirthIdentityLedgerV4State,
    suffix_steps: tuple[GeneratedBirthIdentityFreezeOrdinaryStep, ...],
) -> GeneratedBirthIdentityFreezeEndpointTransaction:
    """Strictly revalidate the due event, suffix coordinate, and fresh key."""

    return _build_endpoint_transaction(
        due_transaction,
        due_inputs,
        freeze_end_core_state,
        freeze_end_ledger_state,
        suffix_steps,
    )


def validate_generated_birth_identity_freeze_endpoint_transaction(
    transaction: GeneratedBirthIdentityFreezeEndpointTransaction,
    *,
    due_transaction: GeneratedBirthIdentityFreezeTransaction,
    due_inputs: GeneratedBirthIdentityFreezeDueInputs,
    freeze_end_core_state: CompositionalFeatureState,
    freeze_end_ledger_state: GeneratedBirthIdentityLedgerV4State,
    suffix_steps: tuple[GeneratedBirthIdentityFreezeOrdinaryStep, ...],
) -> GeneratedBirthIdentityFreezeEndpointValidation:
    """Rebuild and recursively bit-compare one untrusted endpoint receipt."""

    if type(transaction) is not GeneratedBirthIdentityFreezeEndpointTransaction:
        raise TypeError(
            "transaction must be an exact GeneratedBirthIdentityFreezeEndpointTransaction"
        )
    canonical = _build_endpoint_transaction(
        due_transaction,
        due_inputs,
        freeze_end_core_state,
        freeze_end_ledger_state,
        suffix_steps,
    )
    supplied_sha256 = generated_birth_identity_freeze_endpoint_transaction_sha256(
        transaction
    )
    _require(
        transaction.audit.transaction_sha256 == supplied_sha256,
        "supplied endpoint transaction self-hash is stale",
    )
    _same_value_bits(
        transaction,
        canonical,
        name="supplied/canonical complete endpoint transaction",
    )
    return GeneratedBirthIdentityFreezeEndpointValidation(
        valid=True,
        canonical_transaction_sha256=canonical.audit.transaction_sha256,
        supplied_transaction_sha256=transaction.audit.transaction_sha256,
        due_transaction_strictly_revalidated=True,
        suffix_chain_structurally_continuous=True,
        fresh_key_application_bit_exact=True,
        output_core_ledger_pair_ready_for_next_trace=True,
        total_learner_update_calls_accounted=(
            transaction.audit.total_learner_update_calls_for_validated_endpoint
        ),
        total_matched_sham_scrub_kernel_calls_accounted=(
            transaction.audit.total_matched_sham_scrub_kernel_calls_for_validated_endpoint
        ),
        development_only=True,
        execution_authorized=False,
        runner_authorized=False,
        campaign_authorized=False,
        artifact_writes_authorized=False,
        threshold_authorized=False,
        evidence_authorized=False,
        scientific_promotion_allowed=False,
    )


def _validate_endpoint_from_inputs(
    transaction: GeneratedBirthIdentityFreezeEndpointTransaction,
    inputs: GeneratedBirthIdentityFreezeEndpointInputs,
) -> GeneratedBirthIdentityFreezeEndpointValidation:
    if type(inputs) is not GeneratedBirthIdentityFreezeEndpointInputs:
        raise TypeError(
            "endpoint inputs must be exact GeneratedBirthIdentityFreezeEndpointInputs"
        )
    return validate_generated_birth_identity_freeze_endpoint_transaction(
        transaction,
        due_transaction=inputs.due_transaction,
        due_inputs=inputs.due_inputs,
        freeze_end_core_state=inputs.freeze_end_core_state,
        freeze_end_ledger_state=inputs.freeze_end_ledger_state,
        suffix_steps=inputs.suffix_steps,
    )


def _paired_payload(
    transaction: GeneratedBirthIdentityPairedFreezeTransaction,
    *,
    include_transaction_sha256: bool,
) -> dict[str, object]:
    audit = dataclasses.asdict(transaction.audit)
    if not include_transaction_sha256:
        audit.pop("transaction_sha256")
    return {
        "causal_endpoint": _value_record(
            transaction.causal_endpoint,
            path="paired.causal_endpoint",
        ),
        "sham_endpoint": _value_record(
            transaction.sham_endpoint,
            path="paired.sham_endpoint",
        ),
        "causal_output_core_state": _value_record(
            transaction.causal_output_core_state,
            path="paired.causal_output_core_state",
        ),
        "causal_output_ledger_state": _value_record(
            transaction.causal_output_ledger_state,
            path="paired.causal_output_ledger_state",
        ),
        "audit": audit,
    }


def generated_birth_identity_paired_freeze_transaction_sha256(
    transaction: GeneratedBirthIdentityPairedFreezeTransaction,
) -> str:
    return _sha256_json(_paired_payload(transaction, include_transaction_sha256=False))


def _ordinary_input_manifest(
    steps: tuple[GeneratedBirthIdentityFreezeOrdinaryStep, ...],
    *,
    name: str,
) -> tuple[tuple[object, object, object, object], ...]:
    return tuple(
        (
            _value_record(step.observation, path=f"{name}[{index}].observation"),
            _value_record(step.targets, path=f"{name}[{index}].targets"),
            _value_record(step.context_id, path=f"{name}[{index}].context_id"),
            _typed_key_record(
                step.learner_pre_state.key,
                name=f"{name}[{index}].learner_pre_state.key",
            ),
        )
        for index, step in enumerate(steps)
    )


def _build_paired_transaction(
    causal_endpoint: GeneratedBirthIdentityFreezeEndpointTransaction,
    causal_inputs: GeneratedBirthIdentityFreezeEndpointInputs,
    sham_endpoint: GeneratedBirthIdentityFreezeEndpointTransaction,
    sham_inputs: GeneratedBirthIdentityFreezeEndpointInputs,
) -> GeneratedBirthIdentityPairedFreezeTransaction:
    causal_validation = _validate_endpoint_from_inputs(causal_endpoint, causal_inputs)
    sham_validation = _validate_endpoint_from_inputs(sham_endpoint, sham_inputs)
    _require(
        causal_validation.valid and sham_validation.valid,
        "one or both paired freeze endpoints failed strict validation",
    )
    causal_due = causal_inputs.due_transaction
    sham_due = sham_inputs.due_transaction
    _same_value_bits(
        causal_inputs.due_inputs.scrub_rollover,
        sham_inputs.due_inputs.scrub_rollover,
        name="paired causal/sham scrub rollover",
    )
    _same_value_bits(
        causal_inputs.due_inputs.scrub_inputs,
        sham_inputs.due_inputs.scrub_inputs,
        name="paired causal/sham strict scrub inputs",
    )
    _same_value_bits(
        causal_inputs.due_inputs.config,
        sham_inputs.due_inputs.config,
        name="paired causal/sham ledger config",
    )
    _same_value_bits(
        causal_inputs.due_inputs.learner.to_config(),
        sham_inputs.due_inputs.learner.to_config(),
        name="paired causal/sham learner config",
    )
    for field_name in ("observation", "targets", "context_id"):
        _same_value_bits(
            getattr(causal_inputs.due_inputs, field_name),
            getattr(sham_inputs.due_inputs, field_name),
            name=f"paired causal/sham due {field_name}",
        )
    causal_prefix_manifest = _ordinary_input_manifest(
        causal_inputs.due_inputs.prefix_steps,
        name="causal_prefix",
    )
    sham_prefix_manifest = _ordinary_input_manifest(
        sham_inputs.due_inputs.prefix_steps,
        name="sham_prefix",
    )
    _same_value_bits(
        causal_prefix_manifest,
        sham_prefix_manifest,
        name="paired causal/sham prefix CRN inputs",
    )
    causal_suffix_manifest = _ordinary_input_manifest(
        causal_inputs.suffix_steps,
        name="causal_suffix",
    )
    sham_suffix_manifest = _ordinary_input_manifest(
        sham_inputs.suffix_steps,
        name="sham_suffix",
    )
    _same_value_bits(
        causal_suffix_manifest,
        sham_suffix_manifest,
        name="paired causal/sham suffix CRN inputs",
    )
    _require(
        causal_due.audit.arm_mode == CAUSAL_FREEZE_ARM,
        "causal paired path is not marked causal",
    )
    _require(
        sham_due.audit.arm_mode == MATCHED_SHAM_FREEZE_ARM
        and sham_due.audit.matched_sham_scrub_work_executed
        and sham_due.audit.work.total_matched_sham_scrub_kernel_calls > 0,
        "matched-sham path did not execute and validate sham scrub work",
    )
    sham_start_value = sham_inputs.due_inputs.matched_sham_start
    _require(
        sham_start_value is not None,
        "matched-sham due inputs omit the sham start receipt",
    )
    sham_start = cast(GeneratedBirthIdentityMatchedShamStart, sham_start_value)
    _require(
        sham_start.audit.causal_scrub_transaction_sha256
        == causal_due.audit.scrub_rollover_transaction_sha256
        == sham_due.audit.scrub_rollover_transaction_sha256,
        "matched-sham start is not bound to this exact causal scrub",
    )
    causal_start_core_state = causal_inputs.due_inputs.scrub_inputs.post_core_state
    _same_value_bits(
        causal_start_core_state.key,
        sham_start.start_core_state.key,
        name="paired causal/sham freeze-start learner key",
    )
    _same_value_bits(
        causal_inputs.due_inputs.learner_pre_state.key,
        sham_inputs.due_inputs.learner_pre_state.key,
        name="paired causal/sham due pre-state learner key",
    )
    _same_value_bits(
        causal_inputs.freeze_end_core_state.key,
        sham_inputs.freeze_end_core_state.key,
        name="paired causal/sham freeze-end pre-fresh learner key",
    )
    _require(
        causal_due.audit.reacquisition_contract_sha256
        == sham_due.audit.reacquisition_contract_sha256,
        "causal and sham reacquisition contracts differ",
    )
    _require(
        causal_due.audit.shared_core_source_sha256
        == sham_due.audit.shared_core_source_sha256
        and causal_due.audit.shared_ledger_source_sha256
        == sham_due.audit.shared_ledger_source_sha256,
        "causal and sham source-byte bindings differ",
    )
    _require(
        (
            causal_due.audit.freeze_start_step,
            causal_due.audit.freeze_end_step,
            causal_due.audit.phase_derived_due_pre_step,
            causal_due.audit.phase_derived_due_post_step,
        )
        == (
            sham_due.audit.freeze_start_step,
            sham_due.audit.freeze_end_step,
            sham_due.audit.phase_derived_due_pre_step,
            sham_due.audit.phase_derived_due_post_step,
        ),
        "causal and sham freeze coordinates differ",
    )
    causal_endpoint_calls = (
        causal_endpoint.audit.total_learner_update_calls_for_validated_endpoint
    )
    sham_endpoint_calls = (
        sham_endpoint.audit.total_learner_update_calls_for_validated_endpoint
    )
    causal_pair_revalidation_calls = (
        causal_endpoint.audit.learner_update_calls_per_endpoint_build
    )
    sham_pair_revalidation_calls = (
        sham_endpoint.audit.learner_update_calls_per_endpoint_build
    )
    causal_calls = causal_endpoint_calls + 2 * causal_pair_revalidation_calls
    sham_calls = sham_endpoint_calls + 2 * sham_pair_revalidation_calls
    _require(causal_calls == sham_calls, "causal and sham learner work differs")
    sham_scrub_calls_before_pairing = (
        sham_endpoint.audit.total_matched_sham_scrub_kernel_calls_for_validated_endpoint
    )
    sham_scrub_calls_per_pair_revalidation = (
        sham_endpoint.audit.matched_sham_scrub_kernel_calls_per_independent_endpoint_validation
    )
    total_sham_scrub_calls = (
        sham_scrub_calls_before_pairing + 2 * sham_scrub_calls_per_pair_revalidation
    )
    _require(
        causal_due.audit.prefix_update_count == sham_due.audit.prefix_update_count
        and causal_due.audit.suffix_update_count == sham_due.audit.suffix_update_count
        and causal_due.audit.work.learner_update_calls_per_build
        == sham_due.audit.work.learner_update_calls_per_build,
        "causal and sham step/call manifests differ",
    )
    _require(
        causal_endpoint.audit.fresh_key_data_uint32
        == sham_endpoint.audit.fresh_key_data_uint32,
        "causal and sham endpoint keys differ",
    )
    _same_value_bits(
        causal_endpoint.fresh_key_applied_core_state,
        causal_inputs.freeze_end_core_state.replace(  # type: ignore[attr-defined]
            key=causal_endpoint.fresh_key_applied_core_state.key
        ),
        name="causal endpoint carried core state",
    )
    audit = GeneratedBirthIdentityPairedFreezeAudit(
        schema=GENERATED_BIRTH_IDENTITY_FREEZE_SCHEMA,
        status=GENERATED_BIRTH_IDENTITY_FREEZE_STATUS,
        causal_endpoint_transaction_sha256=causal_endpoint.audit.transaction_sha256,
        sham_endpoint_transaction_sha256=sham_endpoint.audit.transaction_sha256,
        causal_due_transaction_sha256=causal_due.audit.transaction_sha256,
        sham_due_transaction_sha256=sham_due.audit.transaction_sha256,
        shared_causal_scrub_transaction_sha256=(
            causal_due.audit.scrub_rollover_transaction_sha256
        ),
        shared_reacquisition_contract_sha256=(
            causal_due.audit.reacquisition_contract_sha256
        ),
        shared_learner_config_sha256=_value_sha256(
            causal_inputs.due_inputs.learner.to_config(),
            path="paired_learner_config",
        ),
        shared_ledger_config_sha256=_value_sha256(
            causal_inputs.due_inputs.config,
            path="paired_ledger_config",
        ),
        shared_core_source_sha256=causal_due.audit.shared_core_source_sha256,
        shared_ledger_source_sha256=causal_due.audit.shared_ledger_source_sha256,
        shared_due_observation_sha256=causal_due.audit.shared_observation_sha256,
        shared_due_targets_sha256=causal_due.audit.shared_targets_sha256,
        shared_due_context_sha256=_value_sha256(
            causal_inputs.due_inputs.context_id,
            path="paired_due_context",
        ),
        shared_freeze_start_key_sha256=_value_sha256(
            causal_start_core_state.key,
            path="paired_freeze_start_key",
        ),
        shared_due_pre_key_sha256=_value_sha256(
            causal_inputs.due_inputs.learner_pre_state.key,
            path="paired_due_pre_key",
        ),
        shared_freeze_end_pre_fresh_key_sha256=_value_sha256(
            causal_inputs.freeze_end_core_state.key,
            path="paired_freeze_end_pre_fresh_key",
        ),
        shared_prefix_input_manifest_sha256=_value_sha256(
            causal_prefix_manifest,
            path="paired_prefix_manifest",
        ),
        shared_suffix_input_manifest_sha256=_value_sha256(
            causal_suffix_manifest,
            path="paired_suffix_manifest",
        ),
        exact_crn_input_parity=True,
        causal_arm_mode=CAUSAL_FREEZE_ARM,
        sham_arm_mode=MATCHED_SHAM_FREEZE_ARM,
        fresh_key_data_uint32=causal_endpoint.audit.fresh_key_data_uint32,
        causal_endpoint_learner_update_calls_before_pairing=causal_endpoint_calls,
        sham_endpoint_learner_update_calls_before_pairing=sham_endpoint_calls,
        causal_endpoint_revalidation_calls_per_paired_build=(
            causal_pair_revalidation_calls
        ),
        sham_endpoint_revalidation_calls_per_paired_build=(
            sham_pair_revalidation_calls
        ),
        causal_endpoint_revalidation_calls_per_independent_paired_validation=(
            causal_pair_revalidation_calls
        ),
        sham_endpoint_revalidation_calls_per_independent_paired_validation=(
            sham_pair_revalidation_calls
        ),
        causal_total_learner_update_calls=causal_calls,
        sham_total_learner_update_calls=sham_calls,
        exact_learner_work_parity=True,
        matched_sham_scrub_kernel_work_executed=True,
        matched_sham_scrub_kernel_calls_before_pairing=(
            sham_scrub_calls_before_pairing
        ),
        matched_sham_scrub_kernel_calls_per_paired_build=(
            sham_scrub_calls_per_pair_revalidation
        ),
        matched_sham_scrub_kernel_calls_per_independent_paired_validation=(
            sham_scrub_calls_per_pair_revalidation
        ),
        total_matched_sham_scrub_kernel_calls_for_validated_pair=(
            total_sham_scrub_calls
        ),
        operation_accounting_scope=(
            "complete causal and matched-sham receipt construction through one paired "
            "builder plus one strict independent paired validator; every supplied "
            "ordinary result, source replay, due dual replay, endpoint revalidation, "
            "and matched-sham scrub kernel execution is counted"
        ),
        sham_endpoint_state_discarded=True,
        causal_output_core_state_sha256=(
            causal_endpoint.audit.fresh_key_applied_core_state_sha256
        ),
        causal_output_ledger_state_sha256=(
            causal_endpoint.carried_ledger_state.integrity_sha256
        ),
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
    transaction = GeneratedBirthIdentityPairedFreezeTransaction(
        causal_endpoint=causal_endpoint,
        sham_endpoint=sham_endpoint,
        causal_output_core_state=causal_endpoint.fresh_key_applied_core_state,
        causal_output_ledger_state=causal_endpoint.carried_ledger_state,
        audit=audit,
    )
    return dataclasses.replace(
        transaction,
        audit=dataclasses.replace(
            audit,
            transaction_sha256=generated_birth_identity_paired_freeze_transaction_sha256(
                transaction
            ),
        ),
    )


def build_generated_birth_identity_paired_freeze_transaction(
    causal_endpoint: GeneratedBirthIdentityFreezeEndpointTransaction,
    causal_inputs: GeneratedBirthIdentityFreezeEndpointInputs,
    sham_endpoint: GeneratedBirthIdentityFreezeEndpointTransaction,
    sham_inputs: GeneratedBirthIdentityFreezeEndpointInputs,
) -> GeneratedBirthIdentityPairedFreezeTransaction:
    """Validate both actual paths and return only the causal output pair."""

    return _build_paired_transaction(
        causal_endpoint,
        causal_inputs,
        sham_endpoint,
        sham_inputs,
    )


def validate_generated_birth_identity_paired_freeze_transaction(
    transaction: GeneratedBirthIdentityPairedFreezeTransaction,
    *,
    causal_endpoint: GeneratedBirthIdentityFreezeEndpointTransaction,
    causal_inputs: GeneratedBirthIdentityFreezeEndpointInputs,
    sham_endpoint: GeneratedBirthIdentityFreezeEndpointTransaction,
    sham_inputs: GeneratedBirthIdentityFreezeEndpointInputs,
) -> GeneratedBirthIdentityPairedFreezeValidation:
    """Revalidate both paths and recursively compare the paired transaction."""

    if type(transaction) is not GeneratedBirthIdentityPairedFreezeTransaction:
        raise TypeError(
            "transaction must be an exact GeneratedBirthIdentityPairedFreezeTransaction"
        )
    canonical = _build_paired_transaction(
        causal_endpoint,
        causal_inputs,
        sham_endpoint,
        sham_inputs,
    )
    _require(
        transaction.audit.transaction_sha256
        == generated_birth_identity_paired_freeze_transaction_sha256(transaction),
        "supplied paired freeze transaction self-hash is stale",
    )
    _same_value_bits(
        transaction,
        canonical,
        name="supplied/canonical complete paired freeze transaction",
    )
    return GeneratedBirthIdentityPairedFreezeValidation(
        valid=True,
        canonical_transaction_sha256=canonical.audit.transaction_sha256,
        supplied_transaction_sha256=transaction.audit.transaction_sha256,
        causal_path_strictly_validated=True,
        sham_path_strictly_validated=True,
        exact_learner_work_parity=True,
        matched_sham_work_actually_consumed=True,
        causal_total_learner_update_calls_accounted=(
            transaction.audit.causal_total_learner_update_calls
        ),
        sham_total_learner_update_calls_accounted=(
            transaction.audit.sham_total_learner_update_calls
        ),
        total_matched_sham_scrub_kernel_calls_accounted=(
            transaction.audit.total_matched_sham_scrub_kernel_calls_for_validated_pair
        ),
        sham_endpoint_state_discarded=True,
        causal_output_ready_for_next_trace=True,
        development_only=True,
        execution_authorized=False,
        runner_authorized=False,
        campaign_authorized=False,
        artifact_writes_authorized=False,
        threshold_authorized=False,
        evidence_authorized=False,
        scientific_promotion_allowed=False,
    )


__all__ = [
    "CAUSAL_FREEZE_ARM",
    "GENERATED_BIRTH_IDENTITY_FREEZE_SCHEMA",
    "GENERATED_BIRTH_IDENTITY_FREEZE_STATUS",
    "MATCHED_SHAM_FREEZE_ARM",
    "GeneratedBirthIdentityFreezeAudit",
    "GeneratedBirthIdentityFreezeDueInputs",
    "GeneratedBirthIdentityFreezeEndpointAudit",
    "GeneratedBirthIdentityFreezeEndpointInputs",
    "GeneratedBirthIdentityFreezeEndpointTransaction",
    "GeneratedBirthIdentityFreezeEndpointValidation",
    "GeneratedBirthIdentityFreezeError",
    "GeneratedBirthIdentityFreezeOrdinaryStep",
    "GeneratedBirthIdentityFreezeTransaction",
    "GeneratedBirthIdentityFreezeValidation",
    "GeneratedBirthIdentityFreezeWorkAccounting",
    "GeneratedBirthIdentityMatchedShamStart",
    "GeneratedBirthIdentityMatchedShamStartAudit",
    "GeneratedBirthIdentityPairedFreezeAudit",
    "GeneratedBirthIdentityPairedFreezeTransaction",
    "GeneratedBirthIdentityPairedFreezeValidation",
    "build_generated_birth_identity_freeze_transaction",
    "build_generated_birth_identity_freeze_endpoint_transaction",
    "build_generated_birth_identity_matched_sham_start",
    "build_generated_birth_identity_paired_freeze_transaction",
    "generated_birth_identity_freeze_endpoint_transaction_sha256",
    "generated_birth_identity_freeze_transaction_sha256",
    "generated_birth_identity_matched_sham_start_sha256",
    "generated_birth_identity_paired_freeze_transaction_sha256",
    "validate_generated_birth_identity_freeze_transaction",
    "validate_generated_birth_identity_freeze_endpoint_transaction",
    "validate_generated_birth_identity_matched_sham_start",
    "validate_generated_birth_identity_paired_freeze_transaction",
]
