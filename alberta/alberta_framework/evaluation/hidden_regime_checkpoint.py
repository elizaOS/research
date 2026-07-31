# mypy: disable-error-code="call-arg"
"""Exact development-only checkpoint/resume for hidden-regime runs.

This module persists only the continuing execution state and its static
binding.  It deliberately contains no performance summary, oracle label,
acceptance gate, artifact writer, or evidence-promotion path.  Chunked versus
one-shot equality is a same-implementation execution-equivalence check on one
JAX/XLA backend; it is not an independent correctness result.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.core.slot_signaling_agent import (
    N_SLOTS,
    SLOT_DURABLE,
    SLOT_VACANT,
    SLOT_VALUE_SHAPE,
    DurableWritePolicy,
    ReplacementTargetPolicy,
    SlotRoleState,
    SlotSignalingAgent,
    SlotSignalingConfig,
    SlotSignalingState,
    slot_signaling_keys,
    slot_signaling_resource_budget,
)
from alberta_framework.evaluation.hidden_regime_execution_governance import (
    hidden_regime_world_requires_managed_execution,
)
from alberta_framework.evaluation.hidden_regime_signaling_development import (
    ACCEPTANCE_STATUS,
    REPLAY_PORTABILITY_SCOPE,
    RESERVED_DEVELOPMENT_SEED_NAMESPACE,
    HiddenRegimeDevelopmentCondition,
    HiddenRegimeDevelopmentConfig,
    HiddenRegimePrimitiveTrace,
    HiddenRegimeSeedPair,
    _channel_code,
    _config_from_payload,
    _expected_trace_dtypes,
    _expected_trace_shapes,
    _make_trace,
    _scan_runner,
    _seed_pair_from_payload,
    condition_spec,
)
from alberta_framework.evaluation.slot_signaling_lifecycle_oracle import (
    SlotRoleOracleConfig,
    SlotRoleStateSnapshot,
    validate_slot_role_state_snapshot,
)
from alberta_framework.streams.hidden_regime_signaling import (
    HiddenRegimeChannel,
    HiddenRegimeSignalingWorld,
    HiddenRegimeWorldState,
    hidden_regime_world_keys,
)

HIDDEN_REGIME_CHECKPOINT_SCHEMA = "alberta.hidden-regime-signaling.checkpoint.v2"
HIDDEN_REGIME_TRACE_CHUNK_SCHEMA = "alberta.hidden-regime-signaling.trace-chunk.v2"
CHECKPOINT_EXECUTION_SCOPE = (
    "same-implementation bit-exact execution equivalence on one JAX/XLA backend; "
    "not independent correctness evidence and not eligible for scientific promotion"
)
CHECKPOINT_INTEGRITY_SCOPE = (
    "self-contained SHA-256 detects accidental mutation only; it is not a signature, "
    "does not authenticate provenance, and does not prove that nonzero learner values "
    "arose from the claimed history; value provenance requires a contiguous trace chain "
    "or replay"
)
DEVELOPMENT_ONLY = True
SCIENTIFIC_PROMOTION_ALLOWED = False
ARTIFACT_WRITTEN = False

_INT32_MAX = int(np.iinfo(np.int32).max)
_UINT32_MAX = int(np.iinfo(np.uint32).max)
_CHECKPOINT_FIELDS = {
    "schema_version",
    "development_only",
    "scientific_promotion_allowed",
    "artifact_written",
    "acceptance_status",
    "execution_equivalence_scope",
    "integrity_scope",
    "replay_portability_scope",
    "condition_binding",
    "config",
    "config_sha256",
    "seed_pair",
    "completed_steps",
    "world_state",
    "learner_state",
    "integrity_sha256",
}
_ROLE_STATE_FIELDS = {
    "values_float32_bits",
    "relevance_mean_float32_bits",
    "relevance_mass_float32_bits",
    "failed_leases",
    "idle_leases",
    "status",
    "generation",
    "active_slot",
    "lease_offset",
    "lease_reward_sum_float32_bits",
    "remaining_durable_tests",
    "search_cursor",
    "candidate_successful_leases",
    "next_generation",
    "policy_key_data",
}
_WORLD_STATE_FIELDS = {
    "cue_key_data",
    "channel_key_data",
    "cue",
    "step_count",
    "schedule_position",
}


class HiddenRegimeCheckpointError(ValueError):
    """Raised when checkpoint or chunk data does not validate exactly."""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise HiddenRegimeCheckpointError(f"{name} must be a mapping")
    if any(type(key) is not str for key in value):
        raise HiddenRegimeCheckpointError(f"{name} keys must be strings")
    return cast(Mapping[str, object], value)


def _require_fields(value: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise HiddenRegimeCheckpointError(
            f"{name} fields differ from schema (missing={missing}, extra={extra})"
        )


def _strict_int(
    value: object,
    name: str,
    *,
    minimum: int = -_INT32_MAX - 1,
    maximum: int = _INT32_MAX,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise HiddenRegimeCheckpointError(
            f"{name} must be an integer in [{minimum}, {maximum}]"
        )
    return value


def _strict_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise HiddenRegimeCheckpointError(f"{name} must be boolean")
    return value


def _strict_string(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise HiddenRegimeCheckpointError(f"{name} must be a nonempty string")
    return value


def _strict_sha256(value: object, name: str) -> str:
    digest = _strict_string(value, name)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise HiddenRegimeCheckpointError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _same_json_type_and_value(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _same_json_type_and_value(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _same_json_type_and_value(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _nested_uint32(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    def visit(current: object, dimensions: tuple[int, ...], path: str) -> object:
        if not dimensions:
            return _strict_int(current, path, minimum=0, maximum=_UINT32_MAX)
        if not isinstance(current, list) or len(current) != dimensions[0]:
            raise HiddenRegimeCheckpointError(
                f"{path} must be a JSON list with length {dimensions[0]}"
            )
        return [
            visit(item, dimensions[1:], f"{path}[{index}]")
            for index, item in enumerate(current)
        ]

    return np.asarray(visit(value, shape, name), dtype=np.uint32)


def _nested_int32(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    def visit(current: object, dimensions: tuple[int, ...], path: str) -> object:
        if not dimensions:
            return _strict_int(current, path)
        if not isinstance(current, list) or len(current) != dimensions[0]:
            raise HiddenRegimeCheckpointError(
                f"{path} must be a JSON list with length {dimensions[0]}"
            )
        return [
            visit(item, dimensions[1:], f"{path}[{index}]")
            for index, item in enumerate(current)
        ]

    return np.asarray(visit(value, shape, name), dtype=np.int32)


def _float32_bits(value: Array) -> object:
    array = np.asarray(value)
    if array.dtype != np.float32:
        raise HiddenRegimeCheckpointError("checkpoint float state must have dtype float32")
    if not np.all(np.isfinite(array)):
        raise HiddenRegimeCheckpointError("checkpoint float state must be finite")
    return array.view(np.uint32).tolist()


def _scalar_float32_bits(value: Array) -> int:
    array = np.asarray(value)
    if array.shape != () or array.dtype != np.float32 or not bool(np.isfinite(array)):
        raise HiddenRegimeCheckpointError("checkpoint scalar float state must be finite float32")
    return int(array.view(np.uint32))


def _key_data(key: Array, name: str) -> list[int]:
    data = np.asarray(jr.key_data(key))
    if data.shape != (2,) or data.dtype != np.uint32:
        raise HiddenRegimeCheckpointError(f"{name} must be a two-word uint32 PRNG key")
    return [int(item) for item in data]


def _wrap_key(data: np.ndarray) -> Array:
    return cast(
        Array,
        jr.wrap_key_data(jnp.asarray(data, dtype=jnp.uint32), impl="threefry2x32"),
    )


@dataclasses.dataclass(frozen=True)
class HiddenRegimeConditionBinding:
    """Resolved static intervention carried by every checkpoint."""

    condition: HiddenRegimeDevelopmentCondition
    channel: HiddenRegimeChannel
    channel_code: int
    helper_write_enabled: bool
    beneficiary_write_enabled: bool
    effective_durable_write_policy: DurableWritePolicy
    effective_replacement_target_policy: ReplacementTargetPolicy

    @classmethod
    def from_condition(
        cls, condition: HiddenRegimeDevelopmentCondition
    ) -> HiddenRegimeConditionBinding:
        spec = condition_spec(condition)
        return cls(
            condition=condition,
            channel=spec.channel,
            channel_code=_channel_code(spec.channel),
            helper_write_enabled=spec.helper_write,
            beneficiary_write_enabled=spec.beneficiary_write,
            effective_durable_write_policy=spec.durable_write_policy,
            effective_replacement_target_policy=spec.replacement_target_policy,
        )

    @classmethod
    def from_dict(cls, value: object) -> HiddenRegimeConditionBinding:
        raw = _mapping(value, "condition_binding")
        expected = {field.name for field in dataclasses.fields(cls)}
        _require_fields(raw, expected, "condition_binding")
        condition = cast(
            HiddenRegimeDevelopmentCondition,
            _strict_string(raw["condition"], "condition_binding.condition"),
        )
        result = cls(
            condition=condition,
            channel=cast(
                HiddenRegimeChannel,
                _strict_string(raw["channel"], "condition_binding.channel"),
            ),
            channel_code=_strict_int(
                raw["channel_code"], "condition_binding.channel_code", minimum=0, maximum=2
            ),
            helper_write_enabled=_strict_bool(
                raw["helper_write_enabled"], "condition_binding.helper_write_enabled"
            ),
            beneficiary_write_enabled=_strict_bool(
                raw["beneficiary_write_enabled"],
                "condition_binding.beneficiary_write_enabled",
            ),
            effective_durable_write_policy=cast(
                DurableWritePolicy,
                _strict_string(
                    raw["effective_durable_write_policy"],
                    "condition_binding.effective_durable_write_policy",
                ),
            ),
            effective_replacement_target_policy=cast(
                ReplacementTargetPolicy,
                _strict_string(
                    raw["effective_replacement_target_policy"],
                    "condition_binding.effective_replacement_target_policy",
                ),
            ),
        )
        try:
            expected_result = cls.from_condition(condition)
        except ValueError as error:
            raise HiddenRegimeCheckpointError(str(error)) from error
        if result != expected_result:
            raise HiddenRegimeCheckpointError("condition binding does not match its condition")
        return result

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def _role_state_to_dict(state: SlotRoleState) -> dict[str, object]:
    return {
        "values_float32_bits": _float32_bits(state.values),
        "relevance_mean_float32_bits": _float32_bits(state.relevance_mean),
        "relevance_mass_float32_bits": _float32_bits(state.relevance_mass),
        "failed_leases": np.asarray(state.failed_leases).tolist(),
        "idle_leases": np.asarray(state.idle_leases).tolist(),
        "status": np.asarray(state.status).tolist(),
        "generation": np.asarray(state.generation).tolist(),
        "active_slot": int(np.asarray(state.active_slot)),
        "lease_offset": int(np.asarray(state.lease_offset)),
        "lease_reward_sum_float32_bits": _scalar_float32_bits(state.lease_reward_sum),
        "remaining_durable_tests": int(np.asarray(state.remaining_durable_tests)),
        "search_cursor": int(np.asarray(state.search_cursor)),
        "candidate_successful_leases": int(
            np.asarray(state.candidate_successful_leases)
        ),
        "next_generation": int(np.asarray(state.next_generation)),
        "policy_key_data": _key_data(state.key, "role policy key"),
    }


def _role_state_from_dict(value: object, name: str) -> SlotRoleState:
    raw = _mapping(value, name)
    _require_fields(raw, _ROLE_STATE_FIELDS, name)
    values_bits = _nested_uint32(
        raw["values_float32_bits"], SLOT_VALUE_SHAPE, f"{name}.values_float32_bits"
    )
    relevance_mean_bits = _nested_uint32(
        raw["relevance_mean_float32_bits"], (N_SLOTS,), f"{name}.relevance_mean_float32_bits"
    )
    relevance_mass_bits = _nested_uint32(
        raw["relevance_mass_float32_bits"], (N_SLOTS,), f"{name}.relevance_mass_float32_bits"
    )
    lease_sum_bits = _strict_int(
        raw["lease_reward_sum_float32_bits"],
        f"{name}.lease_reward_sum_float32_bits",
        minimum=0,
        maximum=_UINT32_MAX,
    )
    key_words = _nested_uint32(raw["policy_key_data"], (2,), f"{name}.policy_key_data")
    result = SlotRoleState(
        values=jnp.asarray(values_bits.view(np.float32), dtype=jnp.float32),
        relevance_mean=jnp.asarray(relevance_mean_bits.view(np.float32), dtype=jnp.float32),
        relevance_mass=jnp.asarray(relevance_mass_bits.view(np.float32), dtype=jnp.float32),
        failed_leases=jnp.asarray(
            _nested_int32(raw["failed_leases"], (N_SLOTS,), f"{name}.failed_leases")
        ),
        idle_leases=jnp.asarray(
            _nested_int32(raw["idle_leases"], (N_SLOTS,), f"{name}.idle_leases")
        ),
        status=jnp.asarray(_nested_int32(raw["status"], (N_SLOTS,), f"{name}.status")),
        generation=jnp.asarray(
            _nested_int32(raw["generation"], (N_SLOTS,), f"{name}.generation")
        ),
        active_slot=jnp.asarray(
            _strict_int(raw["active_slot"], f"{name}.active_slot"), dtype=jnp.int32
        ),
        lease_offset=jnp.asarray(
            _strict_int(raw["lease_offset"], f"{name}.lease_offset"), dtype=jnp.int32
        ),
        lease_reward_sum=jax.lax.bitcast_convert_type(
            jnp.asarray(lease_sum_bits, dtype=jnp.uint32), jnp.float32
        ),
        remaining_durable_tests=jnp.asarray(
            _strict_int(
                raw["remaining_durable_tests"], f"{name}.remaining_durable_tests"
            ),
            dtype=jnp.int32,
        ),
        search_cursor=jnp.asarray(
            _strict_int(raw["search_cursor"], f"{name}.search_cursor"), dtype=jnp.int32
        ),
        candidate_successful_leases=jnp.asarray(
            _strict_int(
                raw["candidate_successful_leases"],
                f"{name}.candidate_successful_leases",
            ),
            dtype=jnp.int32,
        ),
        next_generation=jnp.asarray(
            _strict_int(raw["next_generation"], f"{name}.next_generation"),
            dtype=jnp.int32,
        ),
        key=_wrap_key(key_words),
    )
    for field_name in ("values", "relevance_mean", "relevance_mass", "lease_reward_sum"):
        if not np.all(np.isfinite(np.asarray(getattr(result, field_name)))):
            raise HiddenRegimeCheckpointError(f"{name}.{field_name} must be finite")
    return result


def _world_state_to_dict(state: HiddenRegimeWorldState) -> dict[str, object]:
    return {
        "cue_key_data": _key_data(state.cue_key, "world cue key"),
        "channel_key_data": _key_data(state.channel_key, "world channel key"),
        "cue": int(np.asarray(state.cue)),
        "step_count": int(np.asarray(state.step_count)),
        "schedule_position": int(np.asarray(state.schedule_position)),
    }


def _world_state_from_dict(value: object) -> HiddenRegimeWorldState:
    raw = _mapping(value, "world_state")
    _require_fields(raw, _WORLD_STATE_FIELDS, "world_state")
    return HiddenRegimeWorldState(
        cue_key=_wrap_key(
            _nested_uint32(raw["cue_key_data"], (2,), "world_state.cue_key_data")
        ),
        channel_key=_wrap_key(
            _nested_uint32(raw["channel_key_data"], (2,), "world_state.channel_key_data")
        ),
        cue=jnp.asarray(
            _strict_int(raw["cue"], "world_state.cue", minimum=0, maximum=2),
            dtype=jnp.int32,
        ),
        step_count=jnp.asarray(
            _strict_int(raw["step_count"], "world_state.step_count", minimum=0),
            dtype=jnp.int32,
        ),
        schedule_position=jnp.asarray(
            _strict_int(raw["schedule_position"], "world_state.schedule_position", minimum=0),
            dtype=jnp.int32,
        ),
    )


@dataclasses.dataclass(frozen=True)
class HiddenRegimeCheckpoint:
    """Versioned exact state for one finite nonrepeating development run."""

    binding: HiddenRegimeConditionBinding
    config: HiddenRegimeDevelopmentConfig
    config_sha256: str
    seed_pair: HiddenRegimeSeedPair
    completed_steps: int
    world_state: HiddenRegimeWorldState
    learner_state: SlotSignalingState
    integrity_sha256: str

    @property
    def remaining_steps(self) -> int:
        return self.config.num_steps - self.completed_steps

    @property
    def terminal(self) -> bool:
        return self.completed_steps == self.config.num_steps

    def _body_dict(self) -> dict[str, object]:
        return {
            "schema_version": HIDDEN_REGIME_CHECKPOINT_SCHEMA,
            "development_only": True,
            "scientific_promotion_allowed": False,
            "artifact_written": False,
            "acceptance_status": ACCEPTANCE_STATUS,
            "execution_equivalence_scope": CHECKPOINT_EXECUTION_SCOPE,
            "integrity_scope": CHECKPOINT_INTEGRITY_SCOPE,
            "replay_portability_scope": REPLAY_PORTABILITY_SCOPE,
            "condition_binding": self.binding.to_dict(),
            "config": self.config.to_dict(),
            "config_sha256": self.config_sha256,
            "seed_pair": self.seed_pair.to_dict(),
            "completed_steps": self.completed_steps,
            "world_state": _world_state_to_dict(self.world_state),
            "learner_state": {
                "helper": _role_state_to_dict(self.learner_state.helper),
                "beneficiary": _role_state_to_dict(self.learner_state.beneficiary),
            },
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._body_dict(), "integrity_sha256": self.integrity_sha256}

    def to_json(self) -> str:
        """Return canonical JSON suitable for an exact round trip."""

        errors = validate_hidden_regime_checkpoint(self)
        if errors:
            raise HiddenRegimeCheckpointError("; ".join(errors))
        return _canonical_json(self.to_dict())


@jax.jit
def _advance_named_streams(
    cue_key: Array,
    channel_key: Array,
    cue: Array,
    helper_key: Array,
    beneficiary_key: Array,
    steps: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    """Advance only the named RNG ownership streams for provenance checks."""

    def body(
        _: Array,
        carry: tuple[Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array, Array]:
        old_cue_key, old_channel_key, _old_cue, old_helper_key, old_beneficiary_key = carry
        cue_draw_key, next_cue_key = jr.split(old_cue_key)
        next_cue = jr.randint(cue_draw_key, (), 0, 3, dtype=jnp.int32)
        _, next_channel_key = jr.split(old_channel_key)
        next_helper_key = jr.split(old_helper_key, 4)[0]
        next_beneficiary_key = jr.split(old_beneficiary_key, 4)[0]
        return (
            next_cue_key,
            next_channel_key,
            next_cue,
            next_helper_key,
            next_beneficiary_key,
        )

    return cast(
        tuple[Array, Array, Array, Array, Array],
        jax.lax.fori_loop(
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(steps, dtype=jnp.int32),
            body,
            (cue_key, channel_key, cue, helper_key, beneficiary_key),
        ),
    )


def _effective_learner_config(checkpoint: HiddenRegimeCheckpoint) -> SlotSignalingConfig:
    spec = condition_spec(checkpoint.binding.condition)
    return dataclasses.replace(
        checkpoint.config.learner,
        writable_lru_ablation=False,
        durable_write_policy=spec.durable_write_policy,
        replacement_target_policy=spec.replacement_target_policy,
    )


def _protected_world(config: HiddenRegimeDevelopmentConfig) -> bool:
    """Return whether this schema has no checkpoint execution authority."""

    return hidden_regime_world_requires_managed_execution(config.world)


def _array_equal(left: object, right: object) -> bool:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
        return False
    if left_array.dtype == np.float32:
        return bool(
            np.array_equal(left_array.view(np.uint32), right_array.view(np.uint32))
        )
    return bool(np.array_equal(left_array, right_array))


def _state_equal(left: SlotSignalingState, right: SlotSignalingState) -> bool:
    for role_name in ("helper", "beneficiary"):
        left_role = getattr(left, role_name)
        right_role = getattr(right, role_name)
        for field in dataclasses.fields(left_role):
            left_value = getattr(left_role, field.name)
            right_value = getattr(right_role, field.name)
            if field.name == "key":
                if not _array_equal(jr.key_data(left_value), jr.key_data(right_value)):
                    return False
            elif not _array_equal(left_value, right_value):
                return False
    return True


def _world_state_equal(left: HiddenRegimeWorldState, right: HiddenRegimeWorldState) -> bool:
    return (
        _array_equal(jr.key_data(left.cue_key), jr.key_data(right.cue_key))
        and _array_equal(jr.key_data(left.channel_key), jr.key_data(right.channel_key))
        and _array_equal(left.cue, right.cue)
        and _array_equal(left.step_count, right.step_count)
        and _array_equal(left.schedule_position, right.schedule_position)
    )


def _expected_named_state(
    checkpoint: HiddenRegimeCheckpoint,
) -> tuple[HiddenRegimeWorldState, Array, Array]:
    world = HiddenRegimeSignalingWorld(checkpoint.config.world)
    initial_world = world.init(
        hidden_regime_world_keys(jr.key(checkpoint.seed_pair.world_seed))
    )
    initial_learner = SlotSignalingAgent(_effective_learner_config(checkpoint)).init(
        slot_signaling_keys(jr.key(checkpoint.seed_pair.learner_seed))
    )
    cue_key, channel_key, cue, helper_key, beneficiary_key = _advance_named_streams(
        initial_world.cue_key,
        initial_world.channel_key,
        initial_world.cue,
        initial_learner.helper.key,
        initial_learner.beneficiary.key,
        jnp.asarray(checkpoint.completed_steps, dtype=jnp.int32),
    )
    position = min(checkpoint.completed_steps, checkpoint.config.num_steps - 1)
    expected_world = HiddenRegimeWorldState(
        cue_key=cue_key,
        channel_key=channel_key,
        cue=cue,
        step_count=jnp.asarray(checkpoint.completed_steps, dtype=jnp.int32),
        schedule_position=jnp.asarray(position, dtype=jnp.int32),
    )
    return expected_world, helper_key, beneficiary_key


def validate_hidden_regime_checkpoint(checkpoint: object) -> tuple[str, ...]:
    """Validate structure, provenance, state invariants, and integrity."""

    if not isinstance(checkpoint, HiddenRegimeCheckpoint):
        return ("checkpoint must be a HiddenRegimeCheckpoint",)
    errors: list[str] = []
    if not isinstance(checkpoint.binding, HiddenRegimeConditionBinding):
        errors.append("binding is not a HiddenRegimeConditionBinding")
    else:
        try:
            expected_binding = HiddenRegimeConditionBinding.from_condition(
                checkpoint.binding.condition
            )
            if checkpoint.binding != expected_binding:
                errors.append("condition binding does not match its condition")
        except (TypeError, ValueError) as error:
            errors.append(f"condition binding: {error}")
    if not isinstance(checkpoint.config, HiddenRegimeDevelopmentConfig):
        errors.append("config is not a HiddenRegimeDevelopmentConfig")
    elif checkpoint.config.world.repeat_schedule:
        errors.append("checkpoint execution requires a finite nonrepeating world")
    elif _protected_world(checkpoint.config):
        errors.append(
            "managed calibration/protected worlds cannot be checkpoint-executed; "
            "this checkpoint schema carries no execution authorization"
        )
    if not isinstance(checkpoint.seed_pair, HiddenRegimeSeedPair):
        errors.append("seed_pair is not a HiddenRegimeSeedPair")
    elif checkpoint.seed_pair.namespace == RESERVED_DEVELOPMENT_SEED_NAMESPACE:
        errors.append("reserved development namespace is intentionally unexecuted")
    if (
        type(checkpoint.completed_steps) is not int
        or not isinstance(checkpoint.config, HiddenRegimeDevelopmentConfig)
        or not 0 <= checkpoint.completed_steps <= checkpoint.config.num_steps
    ):
        errors.append("completed_steps lies outside the finite run")
    if not isinstance(checkpoint.world_state, HiddenRegimeWorldState):
        errors.append("world_state is not a HiddenRegimeWorldState")
    if not isinstance(checkpoint.learner_state, SlotSignalingState):
        errors.append("learner_state is not a SlotSignalingState")
    if errors:
        return tuple(errors)
    try:
        expected_config_digest = _sha256(checkpoint.config.to_dict())
        if checkpoint.config_sha256 != expected_config_digest:
            errors.append("config_sha256 does not match the exact configuration")
    except (TypeError, ValueError) as error:
        errors.append(f"config digest: {error}")
    try:
        body = checkpoint._body_dict()
        expected_integrity = _sha256(body)
        if checkpoint.integrity_sha256 != expected_integrity:
            errors.append("integrity_sha256 does not match checkpoint content")
    except (
        AttributeError,
        OverflowError,
        TypeError,
        ValueError,
        HiddenRegimeCheckpointError,
    ) as error:
        errors.append(f"checkpoint serialization: {error}")
    if errors:
        return tuple(errors)

    effective_config = _effective_learner_config(checkpoint)
    oracle_config = SlotRoleOracleConfig.from_config(effective_config)
    for role_name in ("helper", "beneficiary"):
        role = getattr(checkpoint.learner_state, role_name)
        try:
            snapshot = SlotRoleStateSnapshot.from_state(role)
            validation = validate_slot_role_state_snapshot(snapshot, oracle_config)
            if not validation.valid:
                errors.extend(f"{role_name} {item}" for item in validation.mismatches)
            values = np.asarray(role.values)
            relevance_mean = np.asarray(role.relevance_mean)
            lease_sum = float(np.asarray(role.lease_reward_sum))
            if np.any(values < 0.0) or np.any(values > 1.0):
                errors.append(f"{role_name} values lie outside [0, 1]")
            if np.any(relevance_mean < 0.0) or np.any(relevance_mean > 1.0):
                errors.append(f"{role_name} relevance_mean lies outside [0, 1]")
            if not 0.0 <= lease_sum <= checkpoint.config.learner.lease_length:
                errors.append(f"{role_name} lease_reward_sum lies outside one lease")
            status = np.asarray(role.status, dtype=np.int32)
            generation = np.asarray(role.generation, dtype=np.int32)
            occupied_generations = generation[status == SLOT_DURABLE]
            if np.unique(occupied_generations).size != occupied_generations.size:
                errors.append(f"{role_name} durable generations are not unique")
            if np.any(occupied_generations >= int(np.asarray(role.next_generation))):
                errors.append(f"{role_name} durable generation reaches next_generation")
            for slot in range(1, N_SLOTS):
                if int(status[slot]) != SLOT_VACANT:
                    continue
                vacant_arrays = (
                    np.asarray(role.values)[slot],
                    np.asarray(role.relevance_mean)[slot],
                    np.asarray(role.relevance_mass)[slot],
                    np.asarray(role.failed_leases)[slot],
                    np.asarray(role.idle_leases)[slot],
                )
                if any(np.any(value != 0) for value in vacant_arrays):
                    errors.append(f"{role_name} vacant slot {slot} contains persistent data")
        except (AttributeError, IndexError, TypeError, ValueError) as error:
            errors.append(f"{role_name} state: {error}")
    try:
        budget = slot_signaling_resource_budget(checkpoint.learner_state)
        if budget.state_scalars != 138 or budget.state_bytes != 552:
            errors.append("learner state is not the exact 138-scalar/552-byte dyad")
    except (AttributeError, TypeError, ValueError) as error:
        errors.append(f"learner resource state: {error}")

    synchronized_fields = (
        "relevance_mean",
        "relevance_mass",
        "failed_leases",
        "idle_leases",
        "status",
        "generation",
        "active_slot",
        "lease_offset",
        "lease_reward_sum",
        "remaining_durable_tests",
        "search_cursor",
        "candidate_successful_leases",
        "next_generation",
    )
    for field_name in synchronized_fields:
        if not _array_equal(
            getattr(checkpoint.learner_state.helper, field_name),
            getattr(checkpoint.learner_state.beneficiary, field_name),
        ):
            errors.append(f"role lifecycle field {field_name} is not synchronized")

    try:
        expected_world, expected_helper_key, expected_beneficiary_key = _expected_named_state(
            checkpoint
        )
        if not _world_state_equal(checkpoint.world_state, expected_world):
            errors.append("world state does not match seed provenance and completed_steps")
        if not _array_equal(
            jr.key_data(checkpoint.learner_state.helper.key),
            jr.key_data(expected_helper_key),
        ):
            errors.append("helper policy key does not match seed provenance and completed_steps")
        if not _array_equal(
            jr.key_data(checkpoint.learner_state.beneficiary.key),
            jr.key_data(expected_beneficiary_key),
        ):
            errors.append(
                "beneficiary policy key does not match seed provenance and completed_steps"
            )
        if checkpoint.completed_steps == 0:
            initial_learner = SlotSignalingAgent(effective_config).init(
                slot_signaling_keys(jr.key(checkpoint.seed_pair.learner_seed))
            )
            if not _state_equal(checkpoint.learner_state, initial_learner):
                errors.append("initial learner state does not match seed-derived initialization")
    except (IndexError, TypeError, ValueError) as error:
        errors.append(f"seed provenance: {error}")
    return tuple(errors)


def _make_checkpoint(
    *,
    binding: HiddenRegimeConditionBinding,
    config: HiddenRegimeDevelopmentConfig,
    seed_pair: HiddenRegimeSeedPair,
    completed_steps: int,
    world_state: HiddenRegimeWorldState,
    learner_state: SlotSignalingState,
) -> HiddenRegimeCheckpoint:
    config_digest = _sha256(config.to_dict())
    provisional = HiddenRegimeCheckpoint(
        binding=binding,
        config=config,
        config_sha256=config_digest,
        seed_pair=seed_pair,
        completed_steps=completed_steps,
        world_state=world_state,
        learner_state=learner_state,
        integrity_sha256="0" * 64,
    )
    result = dataclasses.replace(provisional, integrity_sha256=_sha256(provisional._body_dict()))
    errors = validate_hidden_regime_checkpoint(result)
    if errors:
        raise HiddenRegimeCheckpointError("; ".join(errors))
    return result


def initialize_hidden_regime_checkpoint(
    condition: HiddenRegimeDevelopmentCondition,
    *,
    seed_pair: HiddenRegimeSeedPair,
    config: HiddenRegimeDevelopmentConfig | None = None,
) -> HiddenRegimeCheckpoint:
    """Create an exact seed-derived step-zero checkpoint without running a learner step."""

    if not isinstance(seed_pair, HiddenRegimeSeedPair):
        raise TypeError("seed_pair must be a HiddenRegimeSeedPair")
    if seed_pair.namespace == RESERVED_DEVELOPMENT_SEED_NAMESPACE:
        raise HiddenRegimeCheckpointError(
            "the reserved development namespace is intentionally unexecuted"
        )
    resolved_config = config or HiddenRegimeDevelopmentConfig()
    if _protected_world(resolved_config):
        raise HiddenRegimeCheckpointError(
            "managed calibration/protected worlds cannot be checkpoint-executed; "
            "this checkpoint schema carries no execution authorization"
        )
    binding = HiddenRegimeConditionBinding.from_condition(condition)
    effective_config = dataclasses.replace(
        resolved_config.learner,
        writable_lru_ablation=False,
        durable_write_policy=binding.effective_durable_write_policy,
        replacement_target_policy=binding.effective_replacement_target_policy,
    )
    world_state = HiddenRegimeSignalingWorld(resolved_config.world).init(
        hidden_regime_world_keys(jr.key(seed_pair.world_seed))
    )
    learner_state = SlotSignalingAgent(effective_config).init(
        slot_signaling_keys(jr.key(seed_pair.learner_seed))
    )
    return _make_checkpoint(
        binding=binding,
        config=resolved_config,
        seed_pair=seed_pair,
        completed_steps=0,
        world_state=world_state,
        learner_state=learner_state,
    )


def parse_hidden_regime_checkpoint(value: str | Mapping[str, object]) -> HiddenRegimeCheckpoint:
    """Parse strict checkpoint JSON/data and verify integrity and provenance."""

    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError as error:
            raise HiddenRegimeCheckpointError(f"invalid checkpoint JSON: {error}") from error
        raw = _mapping(loaded, "checkpoint")
    else:
        raw = _mapping(value, "checkpoint")
    _require_fields(raw, _CHECKPOINT_FIELDS, "checkpoint")
    if raw["schema_version"] != HIDDEN_REGIME_CHECKPOINT_SCHEMA:
        raise HiddenRegimeCheckpointError("unsupported checkpoint schema_version")
    if _strict_bool(raw["development_only"], "development_only") is not True:
        raise HiddenRegimeCheckpointError("checkpoint must remain development-only")
    if (
        _strict_bool(raw["scientific_promotion_allowed"], "scientific_promotion_allowed")
        is not False
    ):
        raise HiddenRegimeCheckpointError("checkpoint cannot authorize scientific promotion")
    if _strict_bool(raw["artifact_written"], "artifact_written") is not False:
        raise HiddenRegimeCheckpointError("checkpoint cannot claim an artifact write")
    if raw["acceptance_status"] != ACCEPTANCE_STATUS:
        raise HiddenRegimeCheckpointError("checkpoint acceptance status is unsupported")
    if raw["execution_equivalence_scope"] != CHECKPOINT_EXECUTION_SCOPE:
        raise HiddenRegimeCheckpointError("checkpoint execution-equivalence scope changed")
    if raw["integrity_scope"] != CHECKPOINT_INTEGRITY_SCOPE:
        raise HiddenRegimeCheckpointError("checkpoint integrity scope changed")
    if raw["replay_portability_scope"] != REPLAY_PORTABILITY_SCOPE:
        raise HiddenRegimeCheckpointError("checkpoint replay-portability scope changed")

    supplied_integrity = _strict_sha256(raw["integrity_sha256"], "integrity_sha256")
    body = {key: value for key, value in raw.items() if key != "integrity_sha256"}
    if _sha256(body) != supplied_integrity:
        raise HiddenRegimeCheckpointError("integrity_sha256 does not match checkpoint content")

    config_raw = _mapping(raw["config"], "config")
    try:
        config = _config_from_payload(config_raw)
    except (TypeError, ValueError) as error:
        raise HiddenRegimeCheckpointError(f"invalid config: {error}") from error
    if not _same_json_type_and_value(dict(config_raw), config.to_dict()):
        raise HiddenRegimeCheckpointError("config JSON types or values are noncanonical")
    supplied_config_digest = _strict_sha256(raw["config_sha256"], "config_sha256")
    if _sha256(config.to_dict()) != supplied_config_digest:
        raise HiddenRegimeCheckpointError("config_sha256 does not match the exact configuration")
    try:
        seed_pair = _seed_pair_from_payload(raw["seed_pair"])
    except (TypeError, ValueError) as error:
        raise HiddenRegimeCheckpointError(f"invalid seed_pair: {error}") from error
    binding = HiddenRegimeConditionBinding.from_dict(raw["condition_binding"])
    completed_steps = _strict_int(
        raw["completed_steps"],
        "completed_steps",
        minimum=0,
        maximum=config.num_steps,
    )
    learner_raw = _mapping(raw["learner_state"], "learner_state")
    _require_fields(learner_raw, {"helper", "beneficiary"}, "learner_state")
    checkpoint = HiddenRegimeCheckpoint(
        binding=binding,
        config=config,
        config_sha256=supplied_config_digest,
        seed_pair=seed_pair,
        completed_steps=completed_steps,
        world_state=_world_state_from_dict(raw["world_state"]),
        learner_state=SlotSignalingState(
            helper=_role_state_from_dict(learner_raw["helper"], "learner_state.helper"),
            beneficiary=_role_state_from_dict(
                learner_raw["beneficiary"], "learner_state.beneficiary"
            ),
        ),
        integrity_sha256=supplied_integrity,
    )
    errors = validate_hidden_regime_checkpoint(checkpoint)
    if errors:
        raise HiddenRegimeCheckpointError("; ".join(errors))
    if checkpoint.to_dict() != dict(raw):
        raise HiddenRegimeCheckpointError("checkpoint does not reconstruct canonically")
    return checkpoint


@dataclasses.dataclass(frozen=True)
class HiddenRegimeTraceChunk:
    """One positive trace interval bound to exact start and end checkpoints."""

    schema_version: str
    start_checkpoint: HiddenRegimeCheckpoint
    end_checkpoint: HiddenRegimeCheckpoint
    trace: HiddenRegimePrimitiveTrace
    trace_sha256: str

    @property
    def start_step(self) -> int:
        return self.start_checkpoint.completed_steps

    @property
    def end_step(self) -> int:
        return self.end_checkpoint.completed_steps

    @property
    def num_steps(self) -> int:
        return self.end_step - self.start_step


def _trace_digest(trace: HiddenRegimePrimitiveTrace) -> str:
    return _sha256(trace.to_dict())


_TRACE_STATE_NAMES = {
    "values": "value_bits",
    "relevance_mean": "relevance_mean",
    "relevance_mass": "relevance_mass",
    "failed_leases": "failed_leases",
    "idle_leases": "idle_leases",
    "status": "status",
    "generation": "generation",
    "active_slot": "active_slot",
    "lease_offset": "lease_offset",
    "lease_reward_sum": "lease_reward_sum",
    "remaining_durable_tests": "remaining_durable_tests",
    "search_cursor": "search_cursor",
    "candidate_successful_leases": "candidate_confirmations",
    "next_generation": "next_generation",
    "key": "policy_key_data",
}


def _role_trace_value(state: SlotRoleState, state_name: str) -> np.ndarray:
    value = getattr(state, state_name)
    if state_name == "values":
        return np.asarray(value).view(np.uint32)
    if state_name == "key":
        return np.asarray(jr.key_data(value), dtype=np.uint32)
    return np.asarray(value)


def _validate_trace_chunk(chunk: HiddenRegimeTraceChunk) -> tuple[str, ...]:
    errors: list[str] = []
    if chunk.schema_version != HIDDEN_REGIME_TRACE_CHUNK_SCHEMA:
        errors.append("trace chunk schema_version is unsupported")
    start_errors = validate_hidden_regime_checkpoint(chunk.start_checkpoint)
    end_errors = validate_hidden_regime_checkpoint(chunk.end_checkpoint)
    errors.extend(f"start checkpoint: {error}" for error in start_errors)
    errors.extend(f"end checkpoint: {error}" for error in end_errors)
    if errors:
        return tuple(errors)
    if chunk.num_steps <= 0:
        errors.append("trace chunk must contain a positive interval")
        return tuple(errors)
    static_start = (
        chunk.start_checkpoint.binding,
        chunk.start_checkpoint.config_sha256,
        chunk.start_checkpoint.seed_pair,
    )
    static_end = (
        chunk.end_checkpoint.binding,
        chunk.end_checkpoint.config_sha256,
        chunk.end_checkpoint.seed_pair,
    )
    if static_start != static_end:
        errors.append("trace chunk endpoints have different static bindings")
    if chunk.trace_sha256 != _trace_digest(chunk.trace):
        errors.append("trace_sha256 does not match primitive trace content")
    expected_shapes = _expected_trace_shapes(chunk.num_steps)
    expected_dtypes = _expected_trace_dtypes()
    for field in dataclasses.fields(HiddenRegimePrimitiveTrace):
        value = np.asarray(getattr(chunk.trace, field.name))
        if value.shape != expected_shapes[field.name]:
            errors.append(f"trace.{field.name} has the wrong shape")
        if value.dtype != expected_dtypes[field.name]:
            errors.append(f"trace.{field.name} has the wrong dtype")
    if errors:
        return tuple(errors)
    expected_steps = np.arange(chunk.start_step, chunk.end_step, dtype=np.int32)
    if not np.array_equal(np.asarray(chunk.trace.step_index), expected_steps):
        errors.append("trace step_index does not equal the checkpoint interval")
    if not np.array_equal(np.asarray(chunk.trace.world_step_count_pre), expected_steps):
        errors.append("trace world_step_count_pre does not equal the checkpoint interval")
    binding = chunk.start_checkpoint.binding
    if not np.all(
        np.asarray(chunk.trace.helper_write_enabled) == binding.helper_write_enabled
    ):
        errors.append("trace helper write permit differs from condition binding")
    if not np.all(
        np.asarray(chunk.trace.beneficiary_write_enabled)
        == binding.beneficiary_write_enabled
    ):
        errors.append("trace beneficiary write permit differs from condition binding")
    if binding.channel == "direct" and not np.array_equal(
        np.asarray(chunk.trace.helper_message), np.asarray(chunk.trace.delivered_message)
    ):
        errors.append("direct-channel trace changed the helper message")
    if binding.channel == "constant_0" and not np.all(
        np.asarray(chunk.trace.delivered_message) == 0
    ):
        errors.append("constant-channel trace delivered a nonzero symbol")
    if np.any(np.asarray(chunk.trace.world_terminated, dtype=np.bool_)):
        errors.append("trace world_terminated differs from continuing-world false")
    discount = np.asarray(chunk.trace.world_discount, dtype=np.float32)
    if not np.array_equal(
        discount.view(np.uint32),
        np.full(discount.shape, np.float32(1.0).view(np.uint32), dtype=np.uint32),
    ):
        errors.append("trace world_discount differs from continuing-world one")

    world_pairs = (
        ("cue", "world_cue"),
        ("step_count", "world_step_count"),
        ("schedule_position", "world_schedule_position"),
        ("cue_key", "world_cue_key_data"),
        ("channel_key", "world_channel_key_data"),
    )
    for state_name, trace_name in world_pairs:
        start_value = getattr(chunk.start_checkpoint.world_state, state_name)
        end_value = getattr(chunk.end_checkpoint.world_state, state_name)
        if state_name.endswith("key"):
            start_value = jr.key_data(start_value)
            end_value = jr.key_data(end_value)
        pre = np.asarray(getattr(chunk.trace, f"{trace_name}_pre"))
        post = np.asarray(getattr(chunk.trace, f"{trace_name}_post"))
        if not np.array_equal(pre[0], np.asarray(start_value)):
            errors.append(f"trace initial {trace_name} differs from start checkpoint")
        if not np.array_equal(post[-1], np.asarray(end_value)):
            errors.append(f"trace final {trace_name} differs from end checkpoint")
        if chunk.num_steps > 1 and not np.array_equal(post[:-1], pre[1:]):
            errors.append(f"trace {trace_name} is internally discontinuous")
    for role_name in ("helper", "beneficiary"):
        start_role = getattr(chunk.start_checkpoint.learner_state, role_name)
        end_role = getattr(chunk.end_checkpoint.learner_state, role_name)
        for state_name, trace_name in _TRACE_STATE_NAMES.items():
            pre = np.asarray(getattr(chunk.trace, f"{role_name}_{trace_name}_pre"))
            post = np.asarray(getattr(chunk.trace, f"{role_name}_{trace_name}_post"))
            if not np.array_equal(pre[0], _role_trace_value(start_role, state_name)):
                errors.append(
                    f"trace initial {role_name}.{state_name} differs from start checkpoint"
                )
            if not np.array_equal(post[-1], _role_trace_value(end_role, state_name)):
                errors.append(
                    f"trace final {role_name}.{state_name} differs from end checkpoint"
                )
            if chunk.num_steps > 1 and not np.array_equal(post[:-1], pre[1:]):
                errors.append(f"trace {role_name}.{state_name} is internally discontinuous")
    return tuple(errors)


def run_hidden_regime_chunk(
    checkpoint: HiddenRegimeCheckpoint,
    num_steps: int,
) -> HiddenRegimeTraceChunk:
    """Advance one exact positive chunk without crossing the finite run bound."""

    errors = validate_hidden_regime_checkpoint(checkpoint)
    if errors:
        raise HiddenRegimeCheckpointError("; ".join(errors))
    if type(num_steps) is not int or num_steps < 1:
        raise HiddenRegimeCheckpointError("num_steps must be a positive integer")
    if num_steps > checkpoint.remaining_steps:
        raise HiddenRegimeCheckpointError("chunk exceeds the remaining finite run")
    binding = checkpoint.binding
    (final_world, final_learner), outputs = _scan_runner(
        checkpoint.config,
        binding.effective_durable_write_policy,
        binding.effective_replacement_target_policy,
        num_steps,
    )(
        checkpoint.world_state,
        checkpoint.learner_state,
        jnp.asarray(binding.channel_code, dtype=jnp.int32),
        jnp.asarray(binding.helper_write_enabled),
        jnp.asarray(binding.beneficiary_write_enabled),
    )
    trace = _make_trace(outputs)
    end_checkpoint = _make_checkpoint(
        binding=binding,
        config=checkpoint.config,
        seed_pair=checkpoint.seed_pair,
        completed_steps=checkpoint.completed_steps + num_steps,
        world_state=final_world,
        learner_state=final_learner,
    )
    result = HiddenRegimeTraceChunk(
        schema_version=HIDDEN_REGIME_TRACE_CHUNK_SCHEMA,
        start_checkpoint=checkpoint,
        end_checkpoint=end_checkpoint,
        trace=trace,
        trace_sha256=_trace_digest(trace),
    )
    chunk_errors = _validate_trace_chunk(result)
    if chunk_errors:
        raise HiddenRegimeCheckpointError("; ".join(chunk_errors))
    return result


def concatenate_hidden_regime_trace_chunks(
    chunks: Sequence[HiddenRegimeTraceChunk],
) -> HiddenRegimePrimitiveTrace:
    """Concatenate one exact contiguous checkpoint chain, rejecting gaps/overlaps."""

    if not isinstance(chunks, Sequence) or isinstance(chunks, (str, bytes)) or not chunks:
        raise HiddenRegimeCheckpointError("chunks must be a nonempty sequence")
    validated = tuple(chunks)
    for index, chunk in enumerate(validated):
        if not isinstance(chunk, HiddenRegimeTraceChunk):
            raise HiddenRegimeCheckpointError(f"chunk {index} has an unsupported type")
        errors = _validate_trace_chunk(chunk)
        if errors:
            raise HiddenRegimeCheckpointError(
                f"chunk {index}: " + "; ".join(errors)
            )
        if index:
            previous = validated[index - 1]
            if previous.end_step != chunk.start_step:
                relation = "overlap" if previous.end_step > chunk.start_step else "gap"
                raise HiddenRegimeCheckpointError(f"chunk chain has a {relation} at index {index}")
            if previous.end_checkpoint.to_dict() != chunk.start_checkpoint.to_dict():
                raise HiddenRegimeCheckpointError(
                    f"chunk chain checkpoint state differs at index {index}"
                )
    outputs = {
        field.name: jnp.concatenate(
            [getattr(chunk.trace, field.name) for chunk in validated], axis=0
        )
        for field in dataclasses.fields(HiddenRegimePrimitiveTrace)
    }
    trace = _make_trace(outputs)
    expected_steps = np.arange(
        validated[0].start_step,
        validated[-1].end_step,
        dtype=np.int32,
    )
    if not np.array_equal(np.asarray(trace.step_index), expected_steps):
        raise HiddenRegimeCheckpointError("concatenated trace is not one contiguous interval")
    return trace


@dataclasses.dataclass(frozen=True)
class HiddenRegimeResumeResult:
    """The exact remainder trace and terminal checkpoint from bounded chunks."""

    chunks: tuple[HiddenRegimeTraceChunk, ...]
    trace: HiddenRegimePrimitiveTrace
    final_checkpoint: HiddenRegimeCheckpoint


def validate_hidden_regime_terminal_checkpoint(
    checkpoint: HiddenRegimeCheckpoint,
) -> tuple[str, ...]:
    """Require a valid checkpoint at the exact finite nonrepeating bound."""

    errors = list(validate_hidden_regime_checkpoint(checkpoint))
    if isinstance(checkpoint, HiddenRegimeCheckpoint) and not checkpoint.terminal:
        errors.append("checkpoint is not at the finite terminal execution bound")
    return tuple(errors)


def resume_hidden_regime_to_completion(
    checkpoint: HiddenRegimeCheckpoint,
    *,
    max_chunk_steps: int,
) -> HiddenRegimeResumeResult:
    """Resume to the finite bound using positive chunks no wider than the limit."""

    errors = validate_hidden_regime_checkpoint(checkpoint)
    if errors:
        raise HiddenRegimeCheckpointError("; ".join(errors))
    if type(max_chunk_steps) is not int or max_chunk_steps < 1:
        raise HiddenRegimeCheckpointError("max_chunk_steps must be a positive integer")
    if checkpoint.terminal:
        raise HiddenRegimeCheckpointError("cannot resume an already terminal checkpoint")
    chunks: list[HiddenRegimeTraceChunk] = []
    current = checkpoint
    while not current.terminal:
        chunk = run_hidden_regime_chunk(
            current,
            min(max_chunk_steps, current.remaining_steps),
        )
        chunks.append(chunk)
        current = chunk.end_checkpoint
    trace = concatenate_hidden_regime_trace_chunks(chunks)
    terminal_errors = validate_hidden_regime_terminal_checkpoint(current)
    if terminal_errors:
        raise HiddenRegimeCheckpointError("; ".join(terminal_errors))
    return HiddenRegimeResumeResult(tuple(chunks), trace, current)


__all__ = [
    "ARTIFACT_WRITTEN",
    "CHECKPOINT_EXECUTION_SCOPE",
    "CHECKPOINT_INTEGRITY_SCOPE",
    "DEVELOPMENT_ONLY",
    "HIDDEN_REGIME_CHECKPOINT_SCHEMA",
    "HIDDEN_REGIME_TRACE_CHUNK_SCHEMA",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "HiddenRegimeCheckpoint",
    "HiddenRegimeCheckpointError",
    "HiddenRegimeConditionBinding",
    "HiddenRegimeResumeResult",
    "HiddenRegimeTraceChunk",
    "concatenate_hidden_regime_trace_chunks",
    "initialize_hidden_regime_checkpoint",
    "parse_hidden_regime_checkpoint",
    "resume_hidden_regime_to_completion",
    "run_hidden_regime_chunk",
    "validate_hidden_regime_checkpoint",
    "validate_hidden_regime_terminal_checkpoint",
]
