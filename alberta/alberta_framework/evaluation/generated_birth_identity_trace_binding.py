"""Development-only authenticated binding of core curation traces to a birth ledger.

The structural entry point in this module consumes a complete public
``CompositionalCurationTrace`` and derives every ledger event index, mask, and
descriptor.  It deliberately remains unauthenticated: a caller can construct
an internally consistent trace and state pair.

The source-replay entry point is stricter.  It accepts only the exact production
``CompositionalFeatureLearner`` class from a byte-pinned core module, replays one
update from the supplied pre-state and inputs, and bit-compares the complete
result tree before constructing the same structural binding.  Typed PRNG keys
are compared by implementation and raw key data; floating arrays and Python
floats are compared by raw bytes, including NaN payloads.

This is a host-side development adapter.  Neither a structural binding nor an
authenticated replay grants execution, runner, artifact-write, evidence, or
scientific-promotion authority.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import struct
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from numpy.typing import NDArray

from alberta_framework.core import compositional_features as core
from alberta_framework.core.compositional_features import (
    COMPOSITIONAL_CURATION_OVERDEPTH_REGENERATION_CHANNEL,
    CURATION_DESTINATION_ACTIVE,
    CURATION_DESTINATION_CANDIDATE,
    CURATION_DESTINATION_NONE,
    FIXED_GENERATOR_POLICY_PLACEHOLDER,
    OP_RAW,
    CompositionalCurationTrace,
    CompositionalFeatureLearner,
    CompositionalFeatureState,
    CompositionalFeatureUpdateResult,
    compositional_curation_keys,
)
from alberta_framework.evaluation import generated_birth_identity_ledger as ledger_module
from alberta_framework.evaluation.generated_birth_identity_ledger import (
    GENERATED_BIRTH_IDENTITY_LEDGER_V4_SCHEMA,
    GeneratedBirthIdentityLedgerV4Config,
    GeneratedBirthIdentityLedgerV4Event,
    GeneratedBirthIdentityLedgerV4State,
    GeneratedBirthIdentityLedgerV4Transaction,
    GeneratedBirthIdentityLedgerV4Validation,
    build_generated_birth_identity_event_v4,
    build_generated_birth_identity_transaction_v4,
    initialize_generated_birth_identity_ledger_v4,
    validate_generated_birth_identity_transaction_v4,
)

GENERATED_BIRTH_IDENTITY_TRACE_BINDING_SCHEMA = (
    "alberta.generated-birth-identity-trace-binding.development.v1"
)
GENERATED_BIRTH_IDENTITY_TRACE_BINDING_STATUS = (
    "DEVELOPMENT_TRACE_BOUND_SOURCE_REPLAY_OPTIONAL_NO_AUTHORITY"
)

PINNED_COMPOSITIONAL_FEATURES_MODULE_SHA256 = (
    "767f054bb3413b2408e664a17bcb8690a9f83018f638d6acfcfde2e9debf5b5a"
)
PINNED_GENERATED_BIRTH_IDENTITY_LEDGER_MODULE_SHA256 = (
    "1008dc090d54d4a776e2681bbbaf8f20c01999839b1d7879137e00f728e85cdb"
)

PINNED_CURATION_DESTINATION_TAGS = (
    CURATION_DESTINATION_NONE,
    CURATION_DESTINATION_ACTIVE,
    CURATION_DESTINATION_CANDIDATE,
)
PINNED_CURATION_KEY_CHANNELS = (
    core.COMPOSITIONAL_CURATION_PROPOSAL_CHANNEL,
    core.COMPOSITIONAL_CURATION_CASCADE_CHANNEL,
    COMPOSITIONAL_CURATION_OVERDEPTH_REGENERATION_CHANNEL,
)

PINNED_COMPOSITIONAL_FEATURE_STATE_FIELD_MANIFEST = (
    "key",
    "ops",
    "parent_a",
    "parent_b",
    "theta",
    "depth",
    "output_weights",
    "output_bias",
    "utilities",
    "utility_contribution_trace",
    "utility_error_trace",
    "utility_feature_trace",
    "utility_feature_energy_trace",
    "utility_signal_second_moment",
    "feature_score_residual_trace",
    "feature_score_energy_trace",
    "retention_slow_utilities",
    "task_activity_ema",
    "ages",
    "candidate_ops",
    "candidate_parent_a",
    "candidate_parent_b",
    "candidate_theta",
    "candidate_depth",
    "candidate_output_weights",
    "candidate_utilities",
    "candidate_utility_contribution_trace",
    "candidate_utility_feature_trace",
    "candidate_utility_feature_energy_trace",
    "candidate_utility_signal_second_moment",
    "candidate_score_residual_trace",
    "candidate_score_energy_trace",
    "candidate_retention_slow_utilities",
    "candidate_active_correlation_trace",
    "candidate_ages",
    "candidate_selector_log_weights",
    "candidate_selector_cumulative_loss",
    "candidate_selector_action_counts",
    "feature_generator_policy",
    "candidate_generator_policy",
    "generator_resource_state",
    "replacement_accumulator",
    "step_count",
    "step_words",
    "replacement_phase",
    "birth_timestamp",
    "uptime_s",
)
PINNED_COMPOSITIONAL_CURATION_TRACE_FIELD_MANIFEST = (
    "pre_step",
    "post_step",
    "pre_step_words",
    "post_step_words",
    "pre_replacement_phase",
    "post_replacement_phase",
    "lifetime_counter_valid",
    "lifetime_capacity_available",
    "decision_key",
    "curation_key",
    "proposal_key",
    "cascade_key",
    "candidate_overdepth_regeneration_key",
    "should_try_replace",
    "has_event",
    "generator_policy_sampled",
    "generator_policy_id",
    "decision_update_available",
    "decision_commit_available",
    "decision_active_ops",
    "decision_active_parent_a",
    "decision_active_parent_b",
    "decision_active_theta",
    "decision_active_depth",
    "decision_active_generator_policy",
    "decision_active_ages",
    "decision_active_fast_utilities",
    "decision_active_slow_utilities",
    "decision_active_direct_scores",
    "decision_active_backed_scores",
    "decision_active_eligible",
    "decision_active_selection_scores",
    "decision_worst_active",
    "decision_has_active_slot",
    "decision_candidate_ops",
    "decision_candidate_parent_a",
    "decision_candidate_parent_b",
    "decision_candidate_theta",
    "decision_candidate_depth",
    "decision_candidate_generator_policy",
    "decision_candidate_ages",
    "decision_candidate_fast_utilities",
    "decision_candidate_slow_utilities",
    "decision_candidate_direct_scores",
    "decision_candidate_novelty_scores",
    "decision_candidate_augmented_scores",
    "decision_candidate_mature",
    "decision_candidate_recomputed_depth",
    "decision_candidate_topology_compatible",
    "decision_candidate_depth_compatible",
    "decision_candidate_headroom_compatible",
    "decision_candidate_margin_eligible",
    "decision_candidate_destination_compatible",
    "decision_candidate_has_destination",
    "decision_candidate_ranking_scores",
    "decision_candidate_refresh_utilities",
    "decision_selected_candidate",
    "decision_has_candidate",
    "decision_selected_destination",
    "decision_selected_refresh_candidate",
    "decision_has_refresh_candidate",
    "decision_left_pack_destinations_enabled",
    "decision_left_pack_destination_available",
    "decision_effective_promotion_margin",
    "decision_selected_candidate_score",
    "decision_selected_destination_backed_score",
    "decision_margin_rhs",
    "decision_margin_passed",
    "decision_selected_topology_ok",
    "decision_selected_depth_ok",
    "decision_selected_headroom_ok",
    "decision_selected_can_promote",
    "decision_should_promote",
    "decision_should_refresh",
    "proposal_formed",
    "proposal_destination_bank",
    "proposal_destination_slot",
    "proposal_op",
    "proposal_parent_a",
    "proposal_parent_b",
    "proposal_theta",
    "proposal_depth",
    "proposal_generator_policy",
    "root_change_mask",
    "root_change_applied",
    "post_root_pre_cascade_slot",
    "post_root_pre_cascade_op",
    "post_root_pre_cascade_parent_a",
    "post_root_pre_cascade_parent_b",
    "post_root_pre_cascade_theta",
    "post_root_pre_cascade_depth",
    "post_root_pre_cascade_generator_policy",
    "promotion_applied",
    "promotion_source_candidate",
    "promotion_destination_active",
    "promoted_pre_refresh_op",
    "promoted_pre_refresh_parent_a",
    "promoted_pre_refresh_parent_b",
    "promoted_pre_refresh_theta",
    "promoted_pre_refresh_depth",
    "promoted_pre_refresh_generator_policy",
    "cascade_refill_mask",
    "cascade_final_ops",
    "cascade_final_parent_a",
    "cascade_final_parent_b",
    "cascade_final_theta",
    "cascade_final_depth",
    "cascade_final_generator_policy",
    "active_change_mask",
    "ordinary_candidate_refresh_mask",
    "post_promotion_candidate_refresh_mask",
    "candidate_refresh_mask",
    "candidate_rebound_mask",
    "candidate_overdepth_regeneration_mask",
    "candidate_final_ops",
    "candidate_final_parent_a",
    "candidate_final_parent_b",
    "candidate_final_theta",
    "candidate_final_depth",
    "candidate_final_generator_policy",
    "proposal_count",
    "root_change_count",
    "promotion_count",
    "cascade_refill_count",
    "ordinary_candidate_refresh_count",
    "post_promotion_candidate_refresh_count",
    "candidate_refresh_count",
    "candidate_rebound_count",
    "candidate_overdepth_regeneration_count",
    "logical_event_count",
)
PINNED_COMPOSITIONAL_UPDATE_RESULT_FIELD_MANIFEST = (
    "state",
    "predictions",
    "errors",
    "metrics",
    "replaced_slot",
    "promoted_candidate",
    "curation_trace",
)

Int32Array = NDArray[np.int32]
UInt32Array = NDArray[np.uint32]
BoolArray = NDArray[np.bool_]


class GeneratedBirthIdentityTraceBindingError(ValueError):
    """A trace, state, replay, pin, or ledger-binding check failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GeneratedBirthIdentityTraceBindingError(message)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _module_source_path(module: ModuleType, *, name: str) -> Path:
    source = inspect.getsourcefile(module)
    _require(source is not None, f"{name} source path is unavailable")
    path = Path(cast(str, source)).resolve()
    _require(path.is_file(), f"{name} source path is not a file")
    return path


def _live_manifest(cls: Any) -> tuple[str, ...]:
    return tuple(field.name for field in dataclasses.fields(cls))


def _validate_source_manifests() -> None:
    _require(
        bool(PINNED_COMPOSITIONAL_FEATURE_STATE_FIELD_MANIFEST),
        "state field manifest pin is pending",
    )
    _require(
        bool(PINNED_COMPOSITIONAL_CURATION_TRACE_FIELD_MANIFEST),
        "trace field manifest pin is pending",
    )
    _require(
        bool(PINNED_COMPOSITIONAL_UPDATE_RESULT_FIELD_MANIFEST),
        "result field manifest pin is pending",
    )
    _require(
        _live_manifest(CompositionalFeatureState)
        == PINNED_COMPOSITIONAL_FEATURE_STATE_FIELD_MANIFEST,
        "CompositionalFeatureState field manifest drifted",
    )
    _require(
        _live_manifest(CompositionalCurationTrace)
        == PINNED_COMPOSITIONAL_CURATION_TRACE_FIELD_MANIFEST,
        "CompositionalCurationTrace field manifest drifted",
    )
    _require(
        _live_manifest(CompositionalFeatureUpdateResult)
        == PINNED_COMPOSITIONAL_UPDATE_RESULT_FIELD_MANIFEST,
        "CompositionalFeatureUpdateResult field manifest drifted",
    )
    _require(
        (
            CURATION_DESTINATION_NONE,
            CURATION_DESTINATION_ACTIVE,
            CURATION_DESTINATION_CANDIDATE,
        )
        == PINNED_CURATION_DESTINATION_TAGS,
        "curation destination tags drifted",
    )
    _require(
        (
            core.COMPOSITIONAL_CURATION_PROPOSAL_CHANNEL,
            core.COMPOSITIONAL_CURATION_CASCADE_CHANNEL,
            core.COMPOSITIONAL_CURATION_OVERDEPTH_REGENERATION_CHANNEL,
        )
        == PINNED_CURATION_KEY_CHANNELS,
        "curation key channels drifted",
    )


def _jax_array(value: object, *, name: str) -> Array:
    _require(isinstance(value, Array), f"{name} must be a JAX array")
    return cast(Array, value)


def _array_bytes(value: Array | np.ndarray[Any, Any]) -> bytes:
    return np.ascontiguousarray(np.asarray(value)).tobytes(order="C")


def _exact_jax_array(
    value: object,
    *,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray[Any, Any]:
    array = _jax_array(value, name=name)
    _require(array.shape == shape, f"{name} must have shape {shape}")
    _require(np.dtype(array.dtype) == dtype, f"{name} must have dtype {dtype.name}")
    return np.asarray(array).copy(order="C")


def _i32_array(value: object, *, shape: tuple[int, ...], name: str) -> Int32Array:
    return cast(
        Int32Array,
        _exact_jax_array(value, dtype=np.dtype(np.int32), shape=shape, name=name),
    )


def _u32_array(value: object, *, shape: tuple[int, ...], name: str) -> UInt32Array:
    return cast(
        UInt32Array,
        _exact_jax_array(value, dtype=np.dtype(np.uint32), shape=shape, name=name),
    )


def _bool_array(value: object, *, shape: tuple[int, ...], name: str) -> BoolArray:
    return cast(
        BoolArray,
        _exact_jax_array(value, dtype=np.dtype(np.bool_), shape=shape, name=name),
    )


def _f32_array(value: object, *, shape: tuple[int, ...], name: str) -> NDArray[np.float32]:
    return cast(
        NDArray[np.float32],
        _exact_jax_array(value, dtype=np.dtype(np.float32), shape=shape, name=name),
    )


def _i32_scalar(value: object, *, name: str) -> int:
    return int(_i32_array(value, shape=(), name=name))


def _bool_scalar(value: object, *, name: str) -> bool:
    return bool(_bool_array(value, shape=(), name=name))


def _same_array_bits(left: object, right: object, *, name: str) -> None:
    left_array = _jax_array(left, name=f"{name} left")
    right_array = _jax_array(right, name=f"{name} right")
    _require(left_array.shape == right_array.shape, f"{name} shape mismatch")
    _require(left_array.dtype == right_array.dtype, f"{name} dtype mismatch")
    left_is_key = jax.dtypes.issubdtype(  # type: ignore[attr-defined]
        left_array.dtype, jax.dtypes.prng_key
    )
    right_is_key = jax.dtypes.issubdtype(  # type: ignore[attr-defined]
        right_array.dtype, jax.dtypes.prng_key
    )
    _require(left_is_key == right_is_key, f"{name} PRNG-key kind mismatch")
    if left_is_key:
        _require(
            type(jr.key_impl(left_array)) is type(jr.key_impl(right_array)),
            f"{name} PRNG implementation type mismatch",
        )
        _require(
            str(jr.key_impl(left_array)) == str(jr.key_impl(right_array)),
            f"{name} PRNG implementation mismatch",
        )
        left_data = jr.key_data(left_array)
        right_data = jr.key_data(right_array)
        _require(left_data.shape == right_data.shape, f"{name} key-data shape mismatch")
        _require(left_data.dtype == right_data.dtype, f"{name} key-data dtype mismatch")
        _require(
            _array_bytes(left_data) == _array_bytes(right_data),
            f"{name} key-data bits mismatch",
        )
        return
    _require(_array_bytes(left_array) == _array_bytes(right_array), f"{name} bits mismatch")


def _bit_compare_complete(left: object, right: object, *, path: str) -> None:
    """Recursively compare a complete replay result without value coercion."""

    _require(type(left) is type(right), f"{path} exact type mismatch")
    if isinstance(left, Array):
        _same_array_bits(left, right, name=path)
        return
    if type(left) is np.ndarray:
        left_array = left
        right_array = cast(np.ndarray[Any, Any], right)
        _require(left_array.shape == right_array.shape, f"{path} shape mismatch")
        _require(left_array.dtype == right_array.dtype, f"{path} dtype mismatch")
        _require(
            np.ascontiguousarray(left_array).tobytes(order="C")
            == np.ascontiguousarray(right_array).tobytes(order="C"),
            f"{path} bits mismatch",
        )
        return
    if isinstance(left, np.generic):
        _require(left.dtype == cast(np.generic, right).dtype, f"{path} dtype mismatch")
        _require(left.tobytes() == cast(np.generic, right).tobytes(), f"{path} bits mismatch")
        return
    if dataclasses.is_dataclass(left) and not isinstance(left, type):
        _require(
            _live_manifest(type(left)) == _live_manifest(type(right)),
            f"{path} dataclass manifest mismatch",
        )
        for field in dataclasses.fields(left):
            _bit_compare_complete(
                getattr(left, field.name),
                getattr(right, field.name),
                path=f"{path}.{field.name}",
            )
        return
    if type(left) is tuple or type(left) is list:
        right_sequence = cast(tuple[object, ...] | list[object], right)
        _require(len(left) == len(right_sequence), f"{path} length mismatch")
        for index, (left_item, right_item) in enumerate(zip(left, right_sequence, strict=True)):
            _bit_compare_complete(left_item, right_item, path=f"{path}[{index}]")
        return
    if type(left) is dict:
        left_dict = cast(dict[object, object], left)
        right_dict = cast(dict[object, object], right)
        _require(tuple(left_dict) == tuple(right_dict), f"{path} dictionary keys mismatch")
        for key in left_dict:
            _bit_compare_complete(left_dict[key], right_dict[key], path=f"{path}[{key!r}]")
        return
    if type(left) is float:
        _require(
            struct.pack("=d", left) == struct.pack("=d", cast(float, right)),
            f"{path} float bits mismatch",
        )
        return
    if type(left) in {bool, int, str, bytes, type(None)}:
        _require(left == right, f"{path} scalar mismatch")
        return
    raise GeneratedBirthIdentityTraceBindingError(
        f"{path} has unsupported replay leaf type {type(left).__qualname__}"
    )


def _singleton_slot(mask: BoolArray, *, name: str, required: bool = False) -> int:
    indices = np.flatnonzero(mask)
    _require(indices.size <= 1, f"{name} must contain at most one selected slot")
    if required:
        _require(indices.size == 1, f"{name} must contain exactly one selected slot")
    return -1 if indices.size == 0 else int(indices[0])


def _same_np_bits(left: np.ndarray[Any, Any], right: np.ndarray[Any, Any], *, name: str) -> None:
    _require(left.shape == right.shape, f"{name} shape mismatch")
    _require(left.dtype == right.dtype, f"{name} dtype mismatch")
    _require(
        np.ascontiguousarray(left).tobytes(order="C")
        == np.ascontiguousarray(right).tobytes(order="C"),
        f"{name} bits mismatch",
    )


def _sentinel_descriptor(trace: CompositionalCurationTrace, *, prefix: str) -> None:
    for suffix in ("slot", "op", "parent_a", "parent_b", "depth", "generator_policy"):
        field_name = f"{prefix}_{suffix}"
        _require(
            _i32_scalar(getattr(trace, field_name), name=field_name) == -1,
            f"{field_name} is not sentinel -1",
        )
    theta = _f32_array(getattr(trace, f"{prefix}_theta"), shape=(2,), name=f"{prefix}_theta")
    _require(
        theta.tobytes(order="C") == np.zeros((2,), dtype=np.float32).tobytes(),
        f"{prefix}_theta is not the exact zero sentinel",
    )


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedBirthIdentityTraceBinding:
    """One structural trace binding and its optional source-replay authentication."""

    schema: str
    status: str
    core_module_sha256: str
    ledger_module_sha256: str
    ledger_schema: str
    state_field_manifest: tuple[str, ...]
    trace_field_manifest: tuple[str, ...]
    result_field_manifest: tuple[str, ...]
    structural_trace_validated: bool
    complete_result_bit_compared: bool
    typed_prng_implementation_and_key_data_compared: bool
    float_raw_bytes_compared_including_nan_payloads: bool
    source_replay_authenticated: bool
    development_only: bool
    execution_authorized: bool
    runner_authorized: bool
    artifact_writes_authorized: bool
    evidence_authorized: bool
    scientific_promotion_allowed: bool
    event: GeneratedBirthIdentityLedgerV4Event
    transaction: GeneratedBirthIdentityLedgerV4Transaction
    ledger_validation: GeneratedBirthIdentityLedgerV4Validation

    def __post_init__(self) -> None:
        _require(
            self.schema == GENERATED_BIRTH_IDENTITY_TRACE_BINDING_SCHEMA, "binding schema is stale"
        )
        _require(
            self.status == GENERATED_BIRTH_IDENTITY_TRACE_BINDING_STATUS, "binding status is stale"
        )
        for name in (
            "structural_trace_validated",
            "complete_result_bit_compared",
            "typed_prng_implementation_and_key_data_compared",
            "float_raw_bytes_compared_including_nan_payloads",
            "source_replay_authenticated",
            "development_only",
            "execution_authorized",
            "runner_authorized",
            "artifact_writes_authorized",
            "evidence_authorized",
            "scientific_promotion_allowed",
        ):
            _require(type(getattr(self, name)) is bool, f"binding {name} must be an exact boolean")
        _require(self.structural_trace_validated, "binding must be structurally validated")
        _require(self.development_only, "binding must remain development-only")
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
            "trace binding cannot grant authority",
        )
        replay_disclosures = (
            self.complete_result_bit_compared,
            self.typed_prng_implementation_and_key_data_compared,
            self.float_raw_bytes_compared_including_nan_payloads,
        )
        _require(
            all(replay_disclosures) == self.source_replay_authenticated,
            "replay disclosures must exactly match source authentication",
        )


def _make_binding(
    *,
    core_sha256: str,
    ledger_sha256: str,
    source_replay_authenticated: bool,
    event: GeneratedBirthIdentityLedgerV4Event,
    transaction: GeneratedBirthIdentityLedgerV4Transaction,
    validation: GeneratedBirthIdentityLedgerV4Validation,
) -> GeneratedBirthIdentityTraceBinding:
    return GeneratedBirthIdentityTraceBinding(
        schema=GENERATED_BIRTH_IDENTITY_TRACE_BINDING_SCHEMA,
        status=GENERATED_BIRTH_IDENTITY_TRACE_BINDING_STATUS,
        core_module_sha256=core_sha256,
        ledger_module_sha256=ledger_sha256,
        ledger_schema=GENERATED_BIRTH_IDENTITY_LEDGER_V4_SCHEMA,
        state_field_manifest=PINNED_COMPOSITIONAL_FEATURE_STATE_FIELD_MANIFEST,
        trace_field_manifest=PINNED_COMPOSITIONAL_CURATION_TRACE_FIELD_MANIFEST,
        result_field_manifest=PINNED_COMPOSITIONAL_UPDATE_RESULT_FIELD_MANIFEST,
        structural_trace_validated=True,
        complete_result_bit_compared=source_replay_authenticated,
        typed_prng_implementation_and_key_data_compared=source_replay_authenticated,
        float_raw_bytes_compared_including_nan_payloads=source_replay_authenticated,
        source_replay_authenticated=source_replay_authenticated,
        development_only=True,
        execution_authorized=False,
        runner_authorized=False,
        artifact_writes_authorized=False,
        evidence_authorized=False,
        scientific_promotion_allowed=False,
        event=event,
        transaction=transaction,
        ledger_validation=validation,
    )


def _state_descriptors(
    state: CompositionalFeatureState,
    *,
    active_slots: int,
    candidate_slots: int,
    prefix: str,
) -> dict[str, np.ndarray[Any, Any]]:
    return {
        "active_parent_a": _i32_array(
            state.parent_a, shape=(active_slots,), name=f"{prefix}.parent_a"
        ),
        "active_parent_b": _i32_array(
            state.parent_b, shape=(active_slots,), name=f"{prefix}.parent_b"
        ),
        "active_ops": _i32_array(state.ops, shape=(active_slots,), name=f"{prefix}.ops"),
        "active_theta": _f32_array(state.theta, shape=(active_slots, 2), name=f"{prefix}.theta"),
        "active_depth": _i32_array(state.depth, shape=(active_slots,), name=f"{prefix}.depth"),
        "active_generator_policy": _i32_array(
            state.feature_generator_policy,
            shape=(active_slots,),
            name=f"{prefix}.feature_generator_policy",
        ),
        "candidate_parent_a": _i32_array(
            state.candidate_parent_a, shape=(candidate_slots,), name=f"{prefix}.candidate_parent_a"
        ),
        "candidate_parent_b": _i32_array(
            state.candidate_parent_b, shape=(candidate_slots,), name=f"{prefix}.candidate_parent_b"
        ),
        "candidate_ops": _i32_array(
            state.candidate_ops, shape=(candidate_slots,), name=f"{prefix}.candidate_ops"
        ),
        "candidate_theta": _f32_array(
            state.candidate_theta, shape=(candidate_slots, 2), name=f"{prefix}.candidate_theta"
        ),
        "candidate_depth": _i32_array(
            state.candidate_depth, shape=(candidate_slots,), name=f"{prefix}.candidate_depth"
        ),
        "candidate_generator_policy": _i32_array(
            state.candidate_generator_policy,
            shape=(candidate_slots,),
            name=f"{prefix}.candidate_generator_policy",
        ),
    }


def _validate_config_and_pre_ledger(
    config: GeneratedBirthIdentityLedgerV4Config,
    ledger_pre_state: GeneratedBirthIdentityLedgerV4State,
    learner_pre_state: CompositionalFeatureState,
) -> dict[str, np.ndarray[Any, Any]]:
    _require(type(config) is GeneratedBirthIdentityLedgerV4Config, "ledger config type is invalid")
    _require(
        type(ledger_pre_state) is GeneratedBirthIdentityLedgerV4State,
        "ledger pre-state type is invalid",
    )
    _require(
        type(learner_pre_state) is CompositionalFeatureState, "learner pre-state type is invalid"
    )
    descriptors = _state_descriptors(
        learner_pre_state,
        active_slots=config.active_slots,
        candidate_slots=config.candidate_slots,
        prefix="learner_pre_state",
    )
    for ledger_name, descriptor_name in (
        ("active_parent_a", "active_parent_a"),
        ("active_parent_b", "active_parent_b"),
        ("active_ops", "active_ops"),
        ("active_depth", "active_depth"),
        ("active_generator_policy", "active_generator_policy"),
        ("candidate_parent_a", "candidate_parent_a"),
        ("candidate_parent_b", "candidate_parent_b"),
        ("candidate_ops", "candidate_ops"),
        ("candidate_depth", "candidate_depth"),
        ("candidate_generator_policy", "candidate_generator_policy"),
    ):
        _same_np_bits(
            np.asarray(getattr(ledger_pre_state, ledger_name)),
            descriptors[descriptor_name],
            name=f"ledger_pre_state.{ledger_name}",
        )
    _require(
        np.array_equal(
            descriptors["active_ops"][: config.raw_feature_slots],
            np.full((config.raw_feature_slots,), OP_RAW, dtype=np.int32),
        ),
        "configured raw prefix does not match learner pre-state",
    )
    return descriptors


def attach_generated_birth_identity_ledger_at_core_genesis(
    config: GeneratedBirthIdentityLedgerV4Config,
    *,
    learner_pre_state: CompositionalFeatureState,
    paired_development_life_seed: int,
) -> GeneratedBirthIdentityLedgerV4State:
    """Attach a sidecar only to an exact learner genesis at canonical step zero."""

    _validate_source_manifests()
    _require(type(config) is GeneratedBirthIdentityLedgerV4Config, "ledger config type is invalid")
    _require(
        type(learner_pre_state) is CompositionalFeatureState, "learner genesis type is invalid"
    )
    _require(
        _i32_scalar(learner_pre_state.step_count, name="genesis.step_count") == 0,
        "ledger attach is only valid at core step 0",
    )
    genesis_words = _u32_array(
        learner_pre_state.step_words,
        shape=(2,),
        name="genesis.step_words",
    )
    _require(not np.any(genesis_words), "ledger attach requires exact genesis words [0, 0]")
    _require(
        _i32_scalar(
            learner_pre_state.replacement_phase,
            name="genesis.replacement_phase",
        )
        == 0,
        "ledger attach requires replacement phase zero",
    )
    descriptors = _state_descriptors(
        learner_pre_state,
        active_slots=config.active_slots,
        candidate_slots=config.candidate_slots,
        prefix="learner_genesis",
    )
    _require(
        not np.any(descriptors["active_generator_policy"]),
        "genesis active provenance must use fixed placeholders",
    )
    _require(
        not np.any(descriptors["candidate_generator_policy"]),
        "genesis candidate provenance must use fixed placeholders",
    )
    return initialize_generated_birth_identity_ledger_v4(
        config,
        paired_development_life_seed=paired_development_life_seed,
        active_parent_a=cast(Int32Array, descriptors["active_parent_a"]),
        active_parent_b=cast(Int32Array, descriptors["active_parent_b"]),
        active_ops=cast(Int32Array, descriptors["active_ops"]),
        active_depth=cast(Int32Array, descriptors["active_depth"]),
        active_generator_policy=cast(Int32Array, descriptors["active_generator_policy"]),
        candidate_parent_a=cast(Int32Array, descriptors["candidate_parent_a"]),
        candidate_parent_b=cast(Int32Array, descriptors["candidate_parent_b"]),
        candidate_ops=cast(Int32Array, descriptors["candidate_ops"]),
        candidate_depth=cast(Int32Array, descriptors["candidate_depth"]),
        candidate_generator_policy=cast(Int32Array, descriptors["candidate_generator_policy"]),
    )


def _validate_trace_keys(
    learner_pre_state: CompositionalFeatureState,
    learner_post_state: CompositionalFeatureState,
    trace: CompositionalCurationTrace,
) -> None:
    expected_post_key, expected_decision_key, expected_curation_key = jr.split(
        learner_pre_state.key,
        3,
    )
    expected_proposal_key, expected_cascade_key = compositional_curation_keys(expected_curation_key)
    expected_overdepth_key = jr.fold_in(
        expected_curation_key,
        jnp.uint32(COMPOSITIONAL_CURATION_OVERDEPTH_REGENERATION_CHANNEL),
    )
    for actual, expected, name in (
        (learner_post_state.key, expected_post_key, "post state key"),
        (trace.decision_key, expected_decision_key, "trace decision key"),
        (trace.curation_key, expected_curation_key, "trace curation key"),
        (trace.proposal_key, expected_proposal_key, "trace proposal key"),
        (trace.cascade_key, expected_cascade_key, "trace cascade key"),
        (
            trace.candidate_overdepth_regeneration_key,
            expected_overdepth_key,
            "trace overdepth-regeneration key",
        ),
    ):
        _same_array_bits(actual, expected, name=name)
    domains = {
        (
            str(jr.key_impl(key)),
            np.asarray(jr.key_data(key)).dtype.str,
            np.asarray(jr.key_data(key)).shape,
            _array_bytes(jr.key_data(key)),
        )
        for key in (expected_proposal_key, expected_cascade_key, expected_overdepth_key)
    }
    _require(len(domains) == 3, "curation subdomains collide for this root key")


def _validate_trace_steps(
    ledger_pre_state: GeneratedBirthIdentityLedgerV4State,
    learner_pre_state: CompositionalFeatureState,
    learner_post_state: CompositionalFeatureState,
    trace: CompositionalCurationTrace,
) -> UInt32Array:
    """Authenticate canonical words; scalar counters remain telemetry only."""

    pre_step = _i32_scalar(learner_pre_state.step_count, name="pre step_count")
    post_step = _i32_scalar(learner_post_state.step_count, name="post step_count")
    pre_words = _u32_array(
        learner_pre_state.step_words,
        shape=(2,),
        name="pre step_words",
    )
    post_words = _u32_array(
        learner_post_state.step_words,
        shape=(2,),
        name="post step_words",
    )
    trace_pre_words = _u32_array(
        trace.pre_step_words,
        shape=(2,),
        name="trace.pre_step_words",
    )
    trace_post_words = _u32_array(
        trace.post_step_words,
        shape=(2,),
        name="trace.post_step_words",
    )
    _same_np_bits(trace_pre_words, pre_words, name="trace pre-step words")
    _same_np_bits(trace_post_words, post_words, name="trace post-step words")
    _same_np_bits(
        np.asarray(ledger_pre_state.step_words),
        pre_words,
        name="ledger/learner pre-step words",
    )
    _require(
        _bool_scalar(trace.lifetime_counter_valid, name="trace.lifetime_counter_valid"),
        "core rejected the supplied lifetime counter state",
    )
    _require(
        _bool_scalar(
            trace.lifetime_capacity_available,
            name="trace.lifetime_capacity_available",
        ),
        "core lifetime identity capacity is exhausted",
    )
    high = int(pre_words[0])
    low = int(pre_words[1])
    _require(
        not (high == 2**32 - 1 and low == 2**32 - 1),
        "pre-state lifetime identity capacity is exhausted",
    )
    next_low = (low + 1) & (2**32 - 1)
    next_high = (high + int(next_low == 0)) & (2**32 - 1)
    expected_post_words = np.asarray((next_high, next_low), dtype=np.uint32)
    _same_np_bits(post_words, expected_post_words, name="checked step-word increment")

    int32_max = 2**31 - 1

    def telemetry(words: UInt32Array) -> int:
        word_high = int(words[0])
        word_low = int(words[1])
        return word_low if word_high == 0 and word_low <= int32_max else int32_max

    _require(pre_step == telemetry(pre_words), "pre scalar step telemetry is stale")
    _require(post_step == telemetry(post_words), "post scalar step telemetry is stale")
    _require(
        _i32_scalar(trace.pre_step, name="trace.pre_step") == pre_step,
        "trace pre-step telemetry does not match the pre-state",
    )
    _require(
        _i32_scalar(trace.post_step, name="trace.post_step") == post_step,
        "trace post-step telemetry does not match the post-state",
    )
    pre_phase = _i32_scalar(
        learner_pre_state.replacement_phase,
        name="pre replacement_phase",
    )
    post_phase = _i32_scalar(
        learner_post_state.replacement_phase,
        name="post replacement_phase",
    )
    _require(
        _i32_scalar(
            trace.pre_replacement_phase,
            name="trace.pre_replacement_phase",
        )
        == pre_phase,
        "trace pre replacement phase is stale",
    )
    _require(
        _i32_scalar(
            trace.post_replacement_phase,
            name="trace.post_replacement_phase",
        )
        == post_phase,
        "trace post replacement phase is stale",
    )
    return post_words


def _validate_counts(
    trace: CompositionalCurationTrace,
    *,
    root_mask: BoolArray,
    cascade_mask: BoolArray,
    ordinary_mask: BoolArray,
    post_promotion_mask: BoolArray,
    refresh_mask: BoolArray,
    rebound_mask: BoolArray,
    overdepth_mask: BoolArray,
) -> None:
    proposal_formed = _bool_scalar(trace.proposal_formed, name="trace.proposal_formed")
    promotion_applied = _bool_scalar(trace.promotion_applied, name="trace.promotion_applied")
    counts = {
        "proposal_count": int(proposal_formed),
        "root_change_count": int(np.count_nonzero(root_mask)),
        "promotion_count": int(promotion_applied),
        "cascade_refill_count": int(np.count_nonzero(cascade_mask)),
        "ordinary_candidate_refresh_count": int(np.count_nonzero(ordinary_mask)),
        "post_promotion_candidate_refresh_count": int(np.count_nonzero(post_promotion_mask)),
        "candidate_refresh_count": int(np.count_nonzero(refresh_mask)),
        "candidate_rebound_count": int(np.count_nonzero(rebound_mask)),
        "candidate_overdepth_regeneration_count": int(np.count_nonzero(overdepth_mask)),
    }
    for field_name, expected in counts.items():
        _require(
            _i32_scalar(getattr(trace, field_name), name=f"trace.{field_name}") == expected,
            f"trace {field_name} is stale",
        )
    logical = (
        counts["root_change_count"]
        + counts["cascade_refill_count"]
        + counts["candidate_refresh_count"]
        + counts["candidate_rebound_count"]
        + counts["candidate_overdepth_regeneration_count"]
    )
    _require(
        _i32_scalar(trace.logical_event_count, name="trace.logical_event_count") == logical,
        "trace logical-event count is stale",
    )
    _require(
        _bool_scalar(trace.has_event, name="trace.has_event") == (logical > 0),
        "trace has-event flag is stale",
    )


def _validate_trace_final_snapshots(
    trace: CompositionalCurationTrace,
    post: dict[str, np.ndarray[Any, Any]],
    *,
    active_slots: int,
    candidate_slots: int,
) -> dict[str, np.ndarray[Any, Any]]:
    trace_final: dict[str, np.ndarray[Any, Any]] = {
        "active_ops": _i32_array(
            trace.cascade_final_ops,
            shape=(active_slots,),
            name="trace.cascade_final_ops",
        ),
        "active_parent_a": _i32_array(
            trace.cascade_final_parent_a,
            shape=(active_slots,),
            name="trace.cascade_final_parent_a",
        ),
        "active_parent_b": _i32_array(
            trace.cascade_final_parent_b,
            shape=(active_slots,),
            name="trace.cascade_final_parent_b",
        ),
        "active_theta": _f32_array(
            trace.cascade_final_theta,
            shape=(active_slots, 2),
            name="trace.cascade_final_theta",
        ),
        "active_depth": _i32_array(
            trace.cascade_final_depth,
            shape=(active_slots,),
            name="trace.cascade_final_depth",
        ),
        "active_generator_policy": _i32_array(
            trace.cascade_final_generator_policy,
            shape=(active_slots,),
            name="trace.cascade_final_generator_policy",
        ),
        "candidate_ops": _i32_array(
            trace.candidate_final_ops,
            shape=(candidate_slots,),
            name="trace.candidate_final_ops",
        ),
        "candidate_parent_a": _i32_array(
            trace.candidate_final_parent_a,
            shape=(candidate_slots,),
            name="trace.candidate_final_parent_a",
        ),
        "candidate_parent_b": _i32_array(
            trace.candidate_final_parent_b,
            shape=(candidate_slots,),
            name="trace.candidate_final_parent_b",
        ),
        "candidate_theta": _f32_array(
            trace.candidate_final_theta,
            shape=(candidate_slots, 2),
            name="trace.candidate_final_theta",
        ),
        "candidate_depth": _i32_array(
            trace.candidate_final_depth,
            shape=(candidate_slots,),
            name="trace.candidate_final_depth",
        ),
        "candidate_generator_policy": _i32_array(
            trace.candidate_final_generator_policy,
            shape=(candidate_slots,),
            name="trace.candidate_final_generator_policy",
        ),
    }
    for name, actual in trace_final.items():
        _same_np_bits(actual, post[name], name=f"trace final {name}")
    return trace_final


def _validate_proposal_and_root_algebra(
    *,
    config: GeneratedBirthIdentityLedgerV4Config,
    trace: CompositionalCurationTrace,
    update_result: CompositionalFeatureUpdateResult,
    pre: dict[str, np.ndarray[Any, Any]],
    final: dict[str, np.ndarray[Any, Any]],
    root_mask: BoolArray,
    ordinary_mask: BoolArray,
    post_promotion_mask: BoolArray,
) -> tuple[int, int, int, int, dict[str, np.ndarray[Any, Any]]]:
    proposal = _bool_scalar(trace.proposal_formed, name="trace.proposal_formed")
    promotion = _bool_scalar(trace.promotion_applied, name="trace.promotion_applied")
    root_applied = _bool_scalar(trace.root_change_applied, name="trace.root_change_applied")
    should_try = _bool_scalar(trace.should_try_replace, name="trace.should_try_replace")
    _require(not proposal or should_try, "a proposal cannot form without a replacement attempt")

    root_slot = _singleton_slot(root_mask, name="trace.root_change_mask")
    ordinary_slot = _singleton_slot(
        ordinary_mask,
        name="trace.ordinary_candidate_refresh_mask",
    )
    post_promotion_slot = _singleton_slot(
        post_promotion_mask,
        name="trace.post_promotion_candidate_refresh_mask",
    )
    _require(root_applied == (root_slot >= 0), "root applied flag does not match root mask")
    _require(
        not np.any(ordinary_mask & post_promotion_mask),
        "ordinary and post-promotion refresh masks overlap",
    )

    promotion_active_slot = -1
    promotion_candidate_slot = -1
    direct_active_slot = -1
    if promotion:
        promotion_active_slot = _i32_scalar(
            trace.promotion_destination_active,
            name="trace.promotion_destination_active",
        )
        promotion_candidate_slot = _i32_scalar(
            trace.promotion_source_candidate,
            name="trace.promotion_source_candidate",
        )
        _require(root_slot == promotion_active_slot, "promotion destination does not match root")
        _require(
            post_promotion_slot == promotion_candidate_slot,
            "promotion source does not match post-promotion refresh",
        )
        _require(ordinary_slot == -1, "promotion and ordinary refresh are incompatible")
        _require(
            _i32_scalar(update_result.replaced_slot, name="result.replaced_slot")
            == promotion_active_slot,
            "result replacement index does not match promotion destination",
        )
        _require(
            _i32_scalar(update_result.promoted_candidate, name="result.promoted_candidate")
            == promotion_candidate_slot,
            "result promoted index does not match promotion source",
        )
        for trace_name, pre_name in (
            ("promoted_pre_refresh_op", "candidate_ops"),
            ("promoted_pre_refresh_parent_a", "candidate_parent_a"),
            ("promoted_pre_refresh_parent_b", "candidate_parent_b"),
            ("promoted_pre_refresh_depth", "candidate_depth"),
            ("promoted_pre_refresh_generator_policy", "candidate_generator_policy"),
        ):
            expected = int(pre[pre_name][promotion_candidate_slot])
            _require(
                _i32_scalar(getattr(trace, trace_name), name=f"trace.{trace_name}") == expected,
                f"trace {trace_name} does not match the promotion source",
            )
    else:
        _require(
            _i32_scalar(trace.promotion_source_candidate, name="trace.promotion_source_candidate")
            == -1,
            "non-promotion source index is not sentinel -1",
        )
        _require(
            _i32_scalar(
                trace.promotion_destination_active,
                name="trace.promotion_destination_active",
            )
            == -1,
            "non-promotion destination index is not sentinel -1",
        )
        for suffix in ("op", "parent_a", "parent_b", "depth", "generator_policy"):
            name = f"promoted_pre_refresh_{suffix}"
            _require(
                _i32_scalar(getattr(trace, name), name=f"trace.{name}") == -1,
                f"trace {name} is not sentinel -1",
            )
        promoted_theta = _f32_array(
            trace.promoted_pre_refresh_theta,
            shape=(2,),
            name="trace.promoted_pre_refresh_theta",
        )
        _require(
            promoted_theta.tobytes(order="C")
            == np.zeros((2,), dtype=np.float32).tobytes(order="C"),
            "non-promotion theta is not the exact zero sentinel",
        )
        _require(
            _i32_scalar(update_result.promoted_candidate, name="result.promoted_candidate") == -1,
            "non-promotion result promoted index is not sentinel -1",
        )
        if root_slot >= 0:
            direct_active_slot = root_slot
            _require(config.candidate_slots == 0, "direct active replacement requires C=0")
            _require(
                _i32_scalar(update_result.replaced_slot, name="result.replaced_slot") == root_slot,
                "direct result replacement index does not match root",
            )
        else:
            _require(
                _i32_scalar(update_result.replaced_slot, name="result.replaced_slot") == -1,
                "result replacement index exists without an active root",
            )

    if root_slot >= 0:
        _require(proposal, "an active root requires proposal formation")
        _require(
            _i32_scalar(trace.post_root_pre_cascade_slot, name="trace.post_root_pre_cascade_slot")
            == root_slot,
            "post-root slot does not match root mask",
        )
        root_values = {
            "op": int(final["active_ops"][root_slot]),
            "parent_a": int(final["active_parent_a"][root_slot]),
            "parent_b": int(final["active_parent_b"][root_slot]),
            "depth": int(final["active_depth"][root_slot]),
            "generator_policy": int(final["active_generator_policy"][root_slot]),
        }
        for suffix, expected in root_values.items():
            _require(
                _i32_scalar(
                    getattr(trace, f"post_root_pre_cascade_{suffix}"),
                    name=f"trace.post_root_pre_cascade_{suffix}",
                )
                == expected,
                f"post-root {suffix} does not match final root",
            )
        post_root_theta = _f32_array(
            trace.post_root_pre_cascade_theta,
            shape=(2,),
            name="trace.post_root_pre_cascade_theta",
        )
        _same_np_bits(
            post_root_theta,
            final["active_theta"][root_slot],
            name="post-root theta",
        )
        if promotion:
            for suffix in ("op", "parent_a", "parent_b", "depth", "generator_policy"):
                _require(
                    _i32_scalar(
                        getattr(trace, f"promoted_pre_refresh_{suffix}"),
                        name=f"trace.promoted_pre_refresh_{suffix}",
                    )
                    == root_values[suffix],
                    f"promoted source {suffix} does not match installed root",
                )
            promoted_theta = _f32_array(
                trace.promoted_pre_refresh_theta,
                shape=(2,),
                name="trace.promoted_pre_refresh_theta",
            )
            _same_np_bits(promoted_theta, post_root_theta, name="promoted root theta")
    else:
        _sentinel_descriptor(trace, prefix="post_root_pre_cascade")

    staged = {
        name: np.array(pre[name], copy=True)
        for name in (
            "candidate_ops",
            "candidate_parent_a",
            "candidate_parent_b",
            "candidate_theta",
            "candidate_depth",
            "candidate_generator_policy",
        )
    }
    destination_bank = _i32_scalar(
        trace.proposal_destination_bank,
        name="trace.proposal_destination_bank",
    )
    destination_slot = _i32_scalar(
        trace.proposal_destination_slot,
        name="trace.proposal_destination_slot",
    )
    if proposal:
        proposal_values = {
            "candidate_ops": _i32_scalar(trace.proposal_op, name="trace.proposal_op"),
            "candidate_parent_a": _i32_scalar(
                trace.proposal_parent_a,
                name="trace.proposal_parent_a",
            ),
            "candidate_parent_b": _i32_scalar(
                trace.proposal_parent_b,
                name="trace.proposal_parent_b",
            ),
            "candidate_depth": _i32_scalar(
                trace.proposal_depth,
                name="trace.proposal_depth",
            ),
            "candidate_generator_policy": _i32_scalar(
                trace.proposal_generator_policy,
                name="trace.proposal_generator_policy",
            ),
        }
        proposal_theta = _f32_array(
            trace.proposal_theta,
            shape=(2,),
            name="trace.proposal_theta",
        )
        if destination_bank == CURATION_DESTINATION_CANDIDATE:
            _require(
                0 <= destination_slot < config.candidate_slots,
                "candidate proposal destination is outside its bank",
            )
            _require(
                destination_slot == (post_promotion_slot if promotion else ordinary_slot),
                "proposal destination does not match candidate refresh mask",
            )
            for name, value in proposal_values.items():
                staged[name][destination_slot] = value
            staged["candidate_theta"][destination_slot] = proposal_theta
        elif destination_bank == CURATION_DESTINATION_ACTIVE:
            _require(not promotion, "promotion proposal destination must be candidate")
            _require(destination_slot == direct_active_slot, "active proposal destination mismatch")
            for suffix, final_name in (
                ("op", "active_ops"),
                ("parent_a", "active_parent_a"),
                ("parent_b", "active_parent_b"),
                ("depth", "active_depth"),
                ("generator_policy", "active_generator_policy"),
            ):
                _require(
                    _i32_scalar(
                        getattr(trace, f"proposal_{suffix}"), name=f"trace.proposal_{suffix}"
                    )
                    == int(final[final_name][destination_slot]),
                    f"active proposal {suffix} does not match final root",
                )
            _same_np_bits(
                proposal_theta,
                final["active_theta"][destination_slot],
                name="active proposal theta",
            )
        else:
            raise GeneratedBirthIdentityTraceBindingError(
                "formed proposal has an invalid destination bank"
            )
    else:
        _require(destination_bank == CURATION_DESTINATION_NONE, "absent proposal bank is stale")
        _require(destination_slot == -1, "absent proposal slot is not sentinel -1")
        for suffix in ("op", "parent_a", "parent_b", "depth", "generator_policy"):
            _require(
                _i32_scalar(getattr(trace, f"proposal_{suffix}"), name=f"trace.proposal_{suffix}")
                == -1,
                f"absent proposal {suffix} is not sentinel -1",
            )
        proposal_theta = _f32_array(
            trace.proposal_theta,
            shape=(2,),
            name="trace.proposal_theta",
        )
        _require(
            proposal_theta.tobytes(order="C")
            == np.zeros((2,), dtype=np.float32).tobytes(order="C"),
            "absent proposal theta is not the exact zero sentinel",
        )

    return (
        promotion_active_slot,
        promotion_candidate_slot,
        direct_active_slot,
        ordinary_slot,
        staged,
    )


def bind_generated_birth_identity_trace_structurally(
    config: GeneratedBirthIdentityLedgerV4Config,
    ledger_pre_state: GeneratedBirthIdentityLedgerV4State,
    *,
    learner_pre_state: CompositionalFeatureState,
    learner_post_state: CompositionalFeatureState,
    update_result: CompositionalFeatureUpdateResult,
) -> GeneratedBirthIdentityTraceBinding:
    """Derive and validate a ledger transition from caller-provided core outputs.

    This entry point does not execute the learner and always returns
    ``source_replay_authenticated=False``.  Use
    :func:`authenticate_generated_birth_identity_trace_by_source_replay` when
    source authentication is required.
    """

    _validate_source_manifests()
    _require(
        type(update_result) is CompositionalFeatureUpdateResult, "update result type is invalid"
    )
    _require(
        type(learner_post_state) is CompositionalFeatureState, "learner post-state type is invalid"
    )
    _require(
        type(update_result.curation_trace) is CompositionalCurationTrace,
        "curation trace type is invalid",
    )
    _bit_compare_complete(
        update_result.state,
        learner_post_state,
        path="supplied_result.state_vs_explicit_post_state",
    )
    trace = update_result.curation_trace
    pre = _validate_config_and_pre_ledger(
        config,
        ledger_pre_state,
        learner_pre_state,
    )
    post = _state_descriptors(
        learner_post_state,
        active_slots=config.active_slots,
        candidate_slots=config.candidate_slots,
        prefix="learner_post_state",
    )
    post_step_words = _validate_trace_steps(
        ledger_pre_state,
        learner_pre_state,
        learner_post_state,
        trace,
    )
    _validate_trace_keys(learner_pre_state, learner_post_state, trace)

    active_slots = config.active_slots
    candidate_slots = config.candidate_slots
    root_mask = _bool_array(
        trace.root_change_mask,
        shape=(active_slots,),
        name="trace.root_change_mask",
    )
    cascade_mask = _bool_array(
        trace.cascade_refill_mask,
        shape=(active_slots,),
        name="trace.cascade_refill_mask",
    )
    active_change_mask = _bool_array(
        trace.active_change_mask,
        shape=(active_slots,),
        name="trace.active_change_mask",
    )
    ordinary_mask = _bool_array(
        trace.ordinary_candidate_refresh_mask,
        shape=(candidate_slots,),
        name="trace.ordinary_candidate_refresh_mask",
    )
    post_promotion_mask = _bool_array(
        trace.post_promotion_candidate_refresh_mask,
        shape=(candidate_slots,),
        name="trace.post_promotion_candidate_refresh_mask",
    )
    refresh_mask = _bool_array(
        trace.candidate_refresh_mask,
        shape=(candidate_slots,),
        name="trace.candidate_refresh_mask",
    )
    rebound_mask = _bool_array(
        trace.candidate_rebound_mask,
        shape=(candidate_slots,),
        name="trace.candidate_rebound_mask",
    )
    overdepth_mask = _bool_array(
        trace.candidate_overdepth_regeneration_mask,
        shape=(candidate_slots,),
        name="trace.candidate_overdepth_regeneration_mask",
    )
    _require(
        np.array_equal(active_change_mask, root_mask | cascade_mask),
        "active-change mask is not root union cascade",
    )
    _require(
        not np.any(root_mask & cascade_mask),
        "root and cascade masks must be disjoint",
    )
    _require(
        np.array_equal(refresh_mask, ordinary_mask | post_promotion_mask),
        "candidate-refresh mask is not ordinary union post-promotion",
    )
    _require(
        not np.any(rebound_mask & overdepth_mask),
        "candidate rebound and overdepth masks overlap",
    )
    _validate_counts(
        trace,
        root_mask=root_mask,
        cascade_mask=cascade_mask,
        ordinary_mask=ordinary_mask,
        post_promotion_mask=post_promotion_mask,
        refresh_mask=refresh_mask,
        rebound_mask=rebound_mask,
        overdepth_mask=overdepth_mask,
    )
    final = _validate_trace_final_snapshots(
        trace,
        post,
        active_slots=active_slots,
        candidate_slots=candidate_slots,
    )

    sampled = _bool_scalar(
        trace.generator_policy_sampled,
        name="trace.generator_policy_sampled",
    )
    policy_id = _i32_scalar(
        trace.generator_policy_id,
        name="trace.generator_policy_id",
    )
    _require(
        sampled == config.learn_generator_resources,
        "trace sampled-policy mode does not match ledger config",
    )
    _require(
        0 <= policy_id < config.generator_policy_count,
        "trace generator-policy id is outside the pinned manifest",
    )
    if not sampled:
        _require(
            policy_id == FIXED_GENERATOR_POLICY_PLACEHOLDER,
            "unsampled trace policy is not the fixed placeholder",
        )

    (
        promotion_active_slot,
        promotion_candidate_slot,
        direct_active_slot,
        ordinary_slot,
        staged,
    ) = _validate_proposal_and_root_algebra(
        config=config,
        trace=trace,
        update_result=update_result,
        pre=pre,
        final=final,
        root_mask=root_mask,
        ordinary_mask=ordinary_mask,
        post_promotion_mask=post_promotion_mask,
    )

    event = build_generated_birth_identity_event_v4(
        config,
        ledger_pre_state,
        post_step_words=post_step_words,
        generator_policy_sampled=sampled,
        generator_policy_id=policy_id,
        active_parent_a=cast(Int32Array, final["active_parent_a"]),
        active_parent_b=cast(Int32Array, final["active_parent_b"]),
        active_ops=cast(Int32Array, final["active_ops"]),
        active_depth=cast(Int32Array, final["active_depth"]),
        active_generator_policy=cast(Int32Array, final["active_generator_policy"]),
        candidate_staged_parent_a=cast(Int32Array, staged["candidate_parent_a"]),
        candidate_staged_parent_b=cast(Int32Array, staged["candidate_parent_b"]),
        candidate_staged_ops=cast(Int32Array, staged["candidate_ops"]),
        candidate_staged_depth=cast(Int32Array, staged["candidate_depth"]),
        candidate_staged_generator_policy=cast(
            Int32Array,
            staged["candidate_generator_policy"],
        ),
        candidate_parent_a=cast(Int32Array, final["candidate_parent_a"]),
        candidate_parent_b=cast(Int32Array, final["candidate_parent_b"]),
        candidate_ops=cast(Int32Array, final["candidate_ops"]),
        candidate_depth=cast(Int32Array, final["candidate_depth"]),
        candidate_generator_policy=cast(
            Int32Array,
            final["candidate_generator_policy"],
        ),
        promotion_active_slot=promotion_active_slot,
        promotion_candidate_slot=promotion_candidate_slot,
        direct_active_replacement_slot=direct_active_slot,
        cascade_refill_mask=cascade_mask,
        ordinary_candidate_refresh_slot=ordinary_slot,
        post_promotion_candidate_refresh_slot=(
            promotion_candidate_slot if promotion_candidate_slot >= 0 else -1
        ),
    )
    _require(
        np.array_equal(event.cascade_refill_mask, cascade_mask),
        "ledger-derived cascade mask differs from trace",
    )
    _require(
        np.array_equal(event.candidate_rebound_mask, rebound_mask),
        "ledger-derived candidate rebound mask differs from trace",
    )
    _require(
        np.array_equal(event.candidate_overdepth_regeneration_mask, overdepth_mask),
        "ledger-derived candidate overdepth mask differs from trace",
    )
    transaction = build_generated_birth_identity_transaction_v4(
        config,
        ledger_pre_state,
        event,
    )
    validation = validate_generated_birth_identity_transaction_v4(
        transaction,
        config=config,
        pre_state=ledger_pre_state,
        event=event,
    )
    core_sha256 = _sha256_file(_module_source_path(core, name="compositional_features"))
    ledger_sha256 = _sha256_file(
        _module_source_path(ledger_module, name="generated_birth_identity_ledger")
    )
    return _make_binding(
        core_sha256=core_sha256,
        ledger_sha256=ledger_sha256,
        source_replay_authenticated=False,
        event=event,
        transaction=transaction,
        validation=validation,
    )


def _context_index(learner: CompositionalFeatureLearner, context_id: Array | int) -> int:
    if type(context_id) is int:
        context = context_id
    else:
        context = _i32_scalar(context_id, name="context_id")
    _require(
        0 <= context < learner._generator_resource_contexts,
        "context_id is outside the learner resource-context bank",
    )
    return context


def _validate_authenticated_configuration(
    learner: CompositionalFeatureLearner,
    config: GeneratedBirthIdentityLedgerV4Config,
    observation: Array,
    targets: Array,
    context_id: Array | int,
) -> None:
    for private_name in (
        "_n_features",
        "_n_tasks",
        "_candidate_count",
        "_max_depth",
        "_generator_resource_contexts",
        "_replacement_interval",
    ):
        _require(
            type(getattr(learner, private_name)) is int,
            f"learner {private_name} must be an exact Python integer",
        )
    _require(
        type(learner._learn_generator_resources) is bool,
        "learner resource mode must be an exact Python boolean",
    )
    _require(learner._n_features == config.active_slots, "active-slot config mismatch")
    _require(
        learner._candidate_count == config.candidate_slots,
        "candidate-slot config mismatch",
    )
    _require(learner._max_depth == config.max_depth, "max-depth config mismatch")
    _require(
        learner._learn_generator_resources == config.learn_generator_resources,
        "generator-resource mode mismatch",
    )
    observation_array = _jax_array(observation, name="observation")
    targets_array = _jax_array(targets, name="targets")
    _require(
        observation_array.ndim == 1 and observation_array.shape[0] == config.raw_feature_slots,
        "observation shape does not match configured raw-feature slots",
    )
    _require(
        targets_array.ndim == 1 and targets_array.shape[0] == learner._n_tasks,
        "target shape does not match learner task count",
    )
    _context_index(learner, context_id)


def _validate_authenticated_replacement_phase(
    learner: CompositionalFeatureLearner,
    pre_state: CompositionalFeatureState,
    result: CompositionalFeatureUpdateResult,
) -> None:
    interval = learner._replacement_interval
    pre_phase = _i32_scalar(pre_state.replacement_phase, name="pre replacement_phase")
    post_phase = _i32_scalar(
        result.state.replacement_phase,
        name="post replacement_phase",
    )
    if interval == 0:
        _require(pre_phase == 0 and post_phase == 0, "disabled replacement phase must stay zero")
        if not learner._learn_generator_resources:
            _require(
                not _bool_scalar(
                    result.curation_trace.should_try_replace,
                    name="trace.should_try_replace",
                ),
                "disabled fixed replacement cannot attempt curation",
            )
        return
    _require(0 <= pre_phase < interval, "pre replacement phase is outside its private bound")
    if learner._learn_generator_resources:
        _require(post_phase == pre_phase, "learned-resource update changed fixed replacement phase")
        return
    expected = 0 if pre_phase == interval - 1 else pre_phase + 1
    _require(post_phase == expected, "fixed replacement phase schedule is stale")
    _require(
        _bool_scalar(
            result.curation_trace.should_try_replace,
            name="trace.should_try_replace",
        )
        == (expected == 0),
        "fixed replacement attempt does not match the bounded phase boundary",
    )


def authenticate_generated_birth_identity_trace_by_source_replay(
    learner: CompositionalFeatureLearner,
    config: GeneratedBirthIdentityLedgerV4Config,
    ledger_pre_state: GeneratedBirthIdentityLedgerV4State,
    *,
    learner_pre_state: CompositionalFeatureState,
    learner_post_state: CompositionalFeatureState,
    supplied_update_result: CompositionalFeatureUpdateResult,
    observation: Array,
    targets: Array,
    context_id: Array | int = 0,
) -> GeneratedBirthIdentityTraceBinding:
    """Authenticate one trace by byte-pinned exact-base-class source replay."""

    _validate_source_manifests()
    _require(
        type(learner) is CompositionalFeatureLearner,
        "authenticated replay requires the exact CompositionalFeatureLearner class",
    )
    _require(
        type(learner_pre_state) is CompositionalFeatureState,
        "authenticated replay pre-state type is invalid",
    )
    _require(
        type(learner_post_state) is CompositionalFeatureState,
        "authenticated replay post-state type is invalid",
    )
    _require(
        type(supplied_update_result) is CompositionalFeatureUpdateResult,
        "authenticated replay result type is invalid",
    )
    _validate_authenticated_configuration(
        learner,
        config,
        observation,
        targets,
        context_id,
    )
    _require(
        len(PINNED_COMPOSITIONAL_FEATURES_MODULE_SHA256) == 64,
        "core module SHA pin is pending or malformed",
    )
    _require(
        len(PINNED_GENERATED_BIRTH_IDENTITY_LEDGER_MODULE_SHA256) == 64,
        "ledger module SHA pin is pending or malformed",
    )
    core_sha256 = _sha256_file(_module_source_path(core, name="compositional_features"))
    ledger_sha256 = _sha256_file(
        _module_source_path(ledger_module, name="generated_birth_identity_ledger")
    )
    _require(
        core_sha256 == PINNED_COMPOSITIONAL_FEATURES_MODULE_SHA256,
        "compositional_features module bytes do not match the authenticated pin",
    )
    _require(
        ledger_sha256 == PINNED_GENERATED_BIRTH_IDENTITY_LEDGER_MODULE_SHA256,
        "birth-identity ledger module bytes do not match the authenticated pin",
    )
    replayed_result = learner.update(
        learner_pre_state,
        observation,
        targets,
        context_id=context_id,
    )
    _require(
        type(replayed_result) is CompositionalFeatureUpdateResult,
        "replay did not return the exact production result class",
    )
    _bit_compare_complete(
        replayed_result,
        supplied_update_result,
        path="complete_update_result",
    )
    _validate_authenticated_replacement_phase(
        learner,
        learner_pre_state,
        supplied_update_result,
    )
    structural = bind_generated_birth_identity_trace_structurally(
        config,
        ledger_pre_state,
        learner_pre_state=learner_pre_state,
        learner_post_state=learner_post_state,
        update_result=supplied_update_result,
    )
    return _make_binding(
        core_sha256=core_sha256,
        ledger_sha256=ledger_sha256,
        source_replay_authenticated=True,
        event=structural.event,
        transaction=structural.transaction,
        validation=structural.ledger_validation,
    )


__all__ = [
    "GENERATED_BIRTH_IDENTITY_TRACE_BINDING_SCHEMA",
    "GENERATED_BIRTH_IDENTITY_TRACE_BINDING_STATUS",
    "PINNED_COMPOSITIONAL_FEATURES_MODULE_SHA256",
    "PINNED_GENERATED_BIRTH_IDENTITY_LEDGER_MODULE_SHA256",
    "PINNED_COMPOSITIONAL_FEATURE_STATE_FIELD_MANIFEST",
    "PINNED_COMPOSITIONAL_CURATION_TRACE_FIELD_MANIFEST",
    "PINNED_COMPOSITIONAL_UPDATE_RESULT_FIELD_MANIFEST",
    "GeneratedBirthIdentityTraceBinding",
    "GeneratedBirthIdentityTraceBindingError",
    "attach_generated_birth_identity_ledger_at_core_genesis",
    "bind_generated_birth_identity_trace_structurally",
    "authenticate_generated_birth_identity_trace_by_source_replay",
]
