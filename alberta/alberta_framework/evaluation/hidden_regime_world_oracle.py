"""Independent host oracle for hidden-regime world transitions.

The oracle consumes a strict JSON-compatible snapshot of one world transition
and reconstructs it with host NumPy plus the two named JAX PRNG streams.  It
does not call ``HiddenRegimeSignalingWorld.step``, ``step_with_delivery``, or
``deliver`` and does not replay an evaluator.

``pre_delivered`` records model the direct ``step_with_delivery`` contract:
the delivered ternary symbol is an explicit evaluator primitive.  Named
channel contracts additionally validate channel resolution.  Irrespective of
the delivery contract, both the cue key and channel key advance exactly once
per world step.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from numbers import Real
from typing import Literal, cast

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.streams.hidden_regime_signaling import (
    CONSTANT_ONE_TERNARY_CHANNEL,
    CONSTANT_TWO_TERNARY_CHANNEL,
    CONSTANT_ZERO_TERNARY_CHANNEL,
    DIRECT_TERNARY_CHANNEL,
    N_HIDDEN_REGIME_SYMBOLS,
    SHUFFLED_TERNARY_CHANNEL,
    HiddenRegimeTransition,
    HiddenRegimeWorldConfig,
    HiddenRegimeWorldState,
)

HIDDEN_REGIME_WORLD_ORACLE_SCHEMA = "alberta.hidden-regime.world-transition.v1"
PRE_DELIVERED: Literal["pre_delivered"] = "pre_delivered"

type DeliveryContract = Literal[
    "pre_delivered",
    "direct",
    "constant_0",
    "constant_1",
    "constant_2",
    "shuffled",
]

_DELIVERY_CONTRACTS: tuple[DeliveryContract, ...] = (
    PRE_DELIVERED,
    DIRECT_TERNARY_CHANNEL,
    CONSTANT_ZERO_TERNARY_CHANNEL,
    CONSTANT_ONE_TERNARY_CHANNEL,
    CONSTANT_TWO_TERNARY_CHANNEL,
    SHUFFLED_TERNARY_CHANNEL,
)
_INT32_MAX = int(np.iinfo(np.int32).max)

IntVector = tuple[int, ...]
IntMatrix = tuple[IntVector, ...]
KeyData = tuple[int, ...]


class HiddenRegimeWorldOracleInputError(ValueError):
    """A serialized world transition is malformed or outside the contract."""


def _require_keys(mapping: Mapping[str, object], expected: set[str], path: str) -> None:
    actual = set(mapping)
    if actual != expected:
        raise HiddenRegimeWorldOracleInputError(
            f"{path} has noncanonical keys; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise HiddenRegimeWorldOracleInputError(f"{path} must be a string-keyed object")
    return cast(Mapping[str, object], value)


def _list(value: object, path: str, length: int | None = None) -> list[object]:
    if type(value) is not list:
        raise HiddenRegimeWorldOracleInputError(f"{path} must be a JSON list")
    result = cast(list[object], value)
    if length is not None and len(result) != length:
        raise HiddenRegimeWorldOracleInputError(
            f"{path} must contain exactly {length} items"
        )
    return result


def _integer(
    value: object,
    path: str,
    *,
    minimum: int = -_INT32_MAX - 1,
    maximum: int = _INT32_MAX,
) -> int:
    if type(value) is not int:
        raise HiddenRegimeWorldOracleInputError(f"{path} must be an integer")
    if not minimum <= value <= maximum:
        raise HiddenRegimeWorldOracleInputError(
            f"{path} must lie in [{minimum}, {maximum}]"
        )
    return value


def _boolean(value: object, path: str) -> bool:
    if type(value) is not bool:
        raise HiddenRegimeWorldOracleInputError(f"{path} must be boolean")
    return value


def _float32(value: object, path: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise HiddenRegimeWorldOracleInputError(f"{path} must be numeric")
    result = float(value)
    result32 = np.float32(result)
    if not math.isfinite(result) or not np.isfinite(result32):
        raise HiddenRegimeWorldOracleInputError(f"{path} must be finite in float32")
    return float(result32)


def _string(value: object, path: str) -> str:
    if type(value) is not str:
        raise HiddenRegimeWorldOracleInputError(f"{path} must be a string")
    return value


def _int_vector(value: object, path: str, length: int | None = None) -> IntVector:
    items = _list(value, path, length)
    return tuple(_integer(item, f"{path}[{index}]") for index, item in enumerate(items))


def _int_matrix(value: object, path: str) -> IntMatrix:
    rows = _list(value, path)
    return tuple(
        _int_vector(row, f"{path}[{index}]", N_HIDDEN_REGIME_SYMBOLS)
        for index, row in enumerate(rows)
    )


def _scalar_int(value: object, path: str) -> int:
    host = np.asarray(value)
    if host.shape != () or host.dtype.kind not in "iu":
        raise HiddenRegimeWorldOracleInputError(f"{path} must be an integer scalar")
    return _integer(int(host), path)


def _scalar_bool(value: object, path: str) -> bool:
    host = np.asarray(value)
    if host.shape != () or host.dtype.kind != "b":
        raise HiddenRegimeWorldOracleInputError(f"{path} must be a boolean scalar")
    return bool(host)


def _scalar_float32(value: object, path: str) -> float:
    host = np.asarray(value, dtype=np.float32)
    if host.shape != () or not np.isfinite(host):
        raise HiddenRegimeWorldOracleInputError(f"{path} must be a finite scalar")
    return float(host)


def _key_data(key: Array, path: str) -> KeyData:
    try:
        host = np.asarray(jr.key_data(key), dtype=np.uint32)
    except (TypeError, ValueError) as error:
        raise HiddenRegimeWorldOracleInputError(f"{path} must be a JAX PRNG key") from error
    if host.shape != (2,):
        raise HiddenRegimeWorldOracleInputError(
            f"{path} must use two-word threefry key data"
        )
    return tuple(int(value) for value in host.tolist())


def _parse_key_data(value: object, path: str) -> KeyData:
    items = _list(value, path, 2)
    return tuple(
        _integer(item, f"{path}[{index}]", minimum=0, maximum=2**32 - 1)
        for index, item in enumerate(items)
    )


def _wrapped_key(data: KeyData) -> Array:
    return cast(
        Array,
        jr.wrap_key_data(jnp.asarray(data, dtype=jnp.uint32), impl="threefry2x32"),
    )


@dataclasses.dataclass(frozen=True)
class HiddenRegimeWorldOracleConfig:
    """Exact evaluator-owned schedule, regime sequence, and permutations."""

    segment_lengths: IntVector
    segment_regimes: IntVector
    regime_permutations: IntMatrix
    repeat_schedule: bool

    @classmethod
    def from_config(
        cls,
        config: HiddenRegimeWorldConfig,
    ) -> HiddenRegimeWorldOracleConfig:
        result = cls(
            segment_lengths=tuple(config.segment_lengths),
            segment_regimes=tuple(config.segment_regimes),
            regime_permutations=tuple(tuple(row) for row in config.regime_permutations),
            repeat_schedule=config.repeat_schedule,
        )
        result.validate()
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> HiddenRegimeWorldOracleConfig:
        expected = {
            "segment_lengths",
            "segment_regimes",
            "regime_permutations",
            "repeat_schedule",
        }
        _require_keys(raw, expected, "config")
        result = cls(
            segment_lengths=_int_vector(raw["segment_lengths"], "config.segment_lengths"),
            segment_regimes=_int_vector(raw["segment_regimes"], "config.segment_regimes"),
            regime_permutations=_int_matrix(
                raw["regime_permutations"], "config.regime_permutations"
            ),
            repeat_schedule=_boolean(raw["repeat_schedule"], "config.repeat_schedule"),
        )
        result.validate()
        return result

    @property
    def total_schedule_steps(self) -> int:
        return sum(self.segment_lengths)

    def validate(self) -> None:
        if type(self.segment_lengths) is not tuple or not self.segment_lengths:
            raise HiddenRegimeWorldOracleInputError(
                "config.segment_lengths must be a nonempty tuple"
            )
        if any(type(length) is not int or length < 1 for length in self.segment_lengths):
            raise HiddenRegimeWorldOracleInputError(
                "config.segment_lengths must contain positive integers"
            )
        if self.total_schedule_steps > _INT32_MAX:
            raise HiddenRegimeWorldOracleInputError("total schedule length must fit int32")
        if (
            type(self.segment_regimes) is not tuple
            or len(self.segment_regimes) != len(self.segment_lengths)
        ):
            raise HiddenRegimeWorldOracleInputError(
                "config.segment_regimes must match segment_lengths"
            )
        if type(self.regime_permutations) is not tuple or not self.regime_permutations:
            raise HiddenRegimeWorldOracleInputError(
                "config.regime_permutations must be a nonempty tuple"
            )
        expected_symbols = tuple(range(N_HIDDEN_REGIME_SYMBOLS))
        for permutation in self.regime_permutations:
            if (
                type(permutation) is not tuple
                or len(permutation) != N_HIDDEN_REGIME_SYMBOLS
                or any(type(symbol) is not int for symbol in permutation)
                or tuple(sorted(permutation)) != expected_symbols
            ):
                raise HiddenRegimeWorldOracleInputError(
                    "every regime permutation must contain exactly 0, 1, 2"
                )
        if any(
            type(regime) is not int
            or regime < 0
            or regime >= len(self.regime_permutations)
            for regime in self.segment_regimes
        ):
            raise HiddenRegimeWorldOracleInputError(
                "config.segment_regimes contains an invalid regime"
            )
        if type(self.repeat_schedule) is not bool:
            raise HiddenRegimeWorldOracleInputError("config.repeat_schedule must be boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "segment_lengths": list(self.segment_lengths),
            "segment_regimes": list(self.segment_regimes),
            "regime_permutations": [list(row) for row in self.regime_permutations],
            "repeat_schedule": self.repeat_schedule,
        }


@dataclasses.dataclass(frozen=True)
class HiddenRegimeWorldStateSnapshot:
    """Complete persistent world state as strict host primitives."""

    cue_key_data: KeyData
    channel_key_data: KeyData
    cue: int
    step_count: int
    schedule_position: int

    @classmethod
    def from_state(cls, state: HiddenRegimeWorldState) -> HiddenRegimeWorldStateSnapshot:
        return cls(
            cue_key_data=_key_data(state.cue_key, "state.cue_key"),
            channel_key_data=_key_data(state.channel_key, "state.channel_key"),
            cue=_scalar_int(state.cue, "state.cue"),
            step_count=_scalar_int(state.step_count, "state.step_count"),
            schedule_position=_scalar_int(
                state.schedule_position, "state.schedule_position"
            ),
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> HiddenRegimeWorldStateSnapshot:
        expected = {
            "cue_key_data",
            "channel_key_data",
            "cue",
            "step_count",
            "schedule_position",
        }
        _require_keys(raw, expected, "state")
        return cls(
            cue_key_data=_parse_key_data(raw["cue_key_data"], "state.cue_key_data"),
            channel_key_data=_parse_key_data(
                raw["channel_key_data"], "state.channel_key_data"
            ),
            cue=_integer(raw["cue"], "state.cue"),
            step_count=_integer(raw["step_count"], "state.step_count"),
            schedule_position=_integer(
                raw["schedule_position"], "state.schedule_position"
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "cue_key_data": list(self.cue_key_data),
            "channel_key_data": list(self.channel_key_data),
            "cue": self.cue,
            "step_count": self.step_count,
            "schedule_position": self.schedule_position,
        }


@dataclasses.dataclass(frozen=True)
class HiddenRegimeWorldDiagnosticsSnapshot:
    """Visible transition facts plus evaluator-only oracle diagnostics."""

    observation_helper_cue: int
    reward: float
    next_observation_helper_cue: int
    terminated: bool
    discount: float
    oracle_step_count: int
    oracle_segment_index: int
    oracle_segment_step: int
    oracle_regime_id: int
    oracle_target: int

    @classmethod
    def from_transition(
        cls,
        transition: HiddenRegimeTransition,
    ) -> HiddenRegimeWorldDiagnosticsSnapshot:
        return cls(
            observation_helper_cue=_scalar_int(
                transition.observation.helper_cue,
                "diagnostics.observation_helper_cue",
            ),
            reward=_scalar_float32(transition.reward, "diagnostics.reward"),
            next_observation_helper_cue=_scalar_int(
                transition.next_observation.helper_cue,
                "diagnostics.next_observation_helper_cue",
            ),
            terminated=_scalar_bool(transition.terminated, "diagnostics.terminated"),
            discount=_scalar_float32(transition.discount, "diagnostics.discount"),
            oracle_step_count=_scalar_int(
                transition.oracle.step_count, "diagnostics.oracle_step_count"
            ),
            oracle_segment_index=_scalar_int(
                transition.oracle.segment_index,
                "diagnostics.oracle_segment_index",
            ),
            oracle_segment_step=_scalar_int(
                transition.oracle.segment_step,
                "diagnostics.oracle_segment_step",
            ),
            oracle_regime_id=_scalar_int(
                transition.oracle.regime_id, "diagnostics.oracle_regime_id"
            ),
            oracle_target=_scalar_int(
                transition.oracle.target, "diagnostics.oracle_target"
            ),
        )

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
    ) -> HiddenRegimeWorldDiagnosticsSnapshot:
        expected = {field.name for field in dataclasses.fields(cls)}
        _require_keys(raw, expected, "diagnostics")
        return cls(
            observation_helper_cue=_integer(
                raw["observation_helper_cue"],
                "diagnostics.observation_helper_cue",
            ),
            reward=_float32(raw["reward"], "diagnostics.reward"),
            next_observation_helper_cue=_integer(
                raw["next_observation_helper_cue"],
                "diagnostics.next_observation_helper_cue",
            ),
            terminated=_boolean(raw["terminated"], "diagnostics.terminated"),
            discount=_float32(raw["discount"], "diagnostics.discount"),
            oracle_step_count=_integer(
                raw["oracle_step_count"], "diagnostics.oracle_step_count"
            ),
            oracle_segment_index=_integer(
                raw["oracle_segment_index"],
                "diagnostics.oracle_segment_index",
            ),
            oracle_segment_step=_integer(
                raw["oracle_segment_step"], "diagnostics.oracle_segment_step"
            ),
            oracle_regime_id=_integer(
                raw["oracle_regime_id"], "diagnostics.oracle_regime_id"
            ),
            oracle_target=_integer(raw["oracle_target"], "diagnostics.oracle_target"),
        )

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class HiddenRegimeWorldTransitionRecord:
    """Strict serializable observation of one complete world transition."""

    config: HiddenRegimeWorldOracleConfig
    old_state: HiddenRegimeWorldStateSnapshot
    delivery_contract: DeliveryContract
    helper_message: int
    delivered_message: int
    beneficiary_action: int
    diagnostics: HiddenRegimeWorldDiagnosticsSnapshot
    new_state: HiddenRegimeWorldStateSnapshot
    schema: str = HIDDEN_REGIME_WORLD_ORACLE_SCHEMA

    @classmethod
    def from_runtime(
        cls,
        *,
        config: HiddenRegimeWorldConfig,
        old_state: HiddenRegimeWorldState,
        delivery_contract: DeliveryContract,
        transition: HiddenRegimeTransition,
        new_state: HiddenRegimeWorldState,
    ) -> HiddenRegimeWorldTransitionRecord:
        return cls(
            config=HiddenRegimeWorldOracleConfig.from_config(config),
            old_state=HiddenRegimeWorldStateSnapshot.from_state(old_state),
            delivery_contract=delivery_contract,
            helper_message=_scalar_int(
                transition.helper_message, "transition.helper_message"
            ),
            delivered_message=_scalar_int(
                transition.delivered_message, "transition.delivered_message"
            ),
            beneficiary_action=_scalar_int(
                transition.beneficiary_action, "transition.beneficiary_action"
            ),
            diagnostics=HiddenRegimeWorldDiagnosticsSnapshot.from_transition(transition),
            new_state=HiddenRegimeWorldStateSnapshot.from_state(new_state),
        )

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
    ) -> HiddenRegimeWorldTransitionRecord:
        expected = {
            "schema",
            "config",
            "old_state",
            "delivery_contract",
            "helper_message",
            "delivered_message",
            "beneficiary_action",
            "diagnostics",
            "new_state",
        }
        _require_keys(raw, expected, "record")
        schema = _string(raw["schema"], "record.schema")
        if schema != HIDDEN_REGIME_WORLD_ORACLE_SCHEMA:
            raise HiddenRegimeWorldOracleInputError("record.schema is not supported")
        contract_raw = _string(raw["delivery_contract"], "record.delivery_contract")
        if contract_raw not in _DELIVERY_CONTRACTS:
            raise HiddenRegimeWorldOracleInputError(
                "record.delivery_contract is not supported"
            )
        return cls(
            schema=schema,
            config=HiddenRegimeWorldOracleConfig.from_dict(
                _mapping(raw["config"], "config")
            ),
            old_state=HiddenRegimeWorldStateSnapshot.from_dict(
                _mapping(raw["old_state"], "old_state")
            ),
            delivery_contract=contract_raw,
            helper_message=_integer(raw["helper_message"], "record.helper_message"),
            delivered_message=_integer(
                raw["delivered_message"], "record.delivered_message"
            ),
            beneficiary_action=_integer(
                raw["beneficiary_action"], "record.beneficiary_action"
            ),
            diagnostics=HiddenRegimeWorldDiagnosticsSnapshot.from_dict(
                _mapping(raw["diagnostics"], "diagnostics")
            ),
            new_state=HiddenRegimeWorldStateSnapshot.from_dict(
                _mapping(raw["new_state"], "new_state")
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "config": self.config.to_dict(),
            "old_state": self.old_state.to_dict(),
            "delivery_contract": self.delivery_contract,
            "helper_message": self.helper_message,
            "delivered_message": self.delivered_message,
            "beneficiary_action": self.beneficiary_action,
            "diagnostics": self.diagnostics.to_dict(),
            "new_state": self.new_state.to_dict(),
        }


@dataclasses.dataclass(frozen=True)
class HiddenRegimeWorldTransitionExpectation:
    """Host-reconstructed delivery, diagnostics, and complete new state."""

    delivered_message: int
    diagnostics: HiddenRegimeWorldDiagnosticsSnapshot
    new_state: HiddenRegimeWorldStateSnapshot


@dataclasses.dataclass(frozen=True)
class HiddenRegimeWorldOracleValidation:
    """Fail-closed transition validation with leaf field paths."""

    valid: bool
    mismatches: tuple[str, ...]
    expected: HiddenRegimeWorldTransitionExpectation | None


@dataclasses.dataclass(frozen=True)
class HiddenRegimeWorldStateValidation:
    """Standalone state or adjacent-state continuity validation."""

    valid: bool
    mismatches: tuple[str, ...]


def _validate_state(
    state: HiddenRegimeWorldStateSnapshot,
    config: HiddenRegimeWorldOracleConfig,
) -> None:
    if not 0 <= state.cue < N_HIDDEN_REGIME_SYMBOLS:
        raise HiddenRegimeWorldOracleInputError("state.cue is outside the ternary domain")
    if not 0 <= state.step_count <= _INT32_MAX:
        raise HiddenRegimeWorldOracleInputError("state.step_count is outside int32 count range")
    if not 0 <= state.schedule_position < config.total_schedule_steps:
        raise HiddenRegimeWorldOracleInputError(
            "state.schedule_position is outside the bounded schedule"
        )


def _expected_delivery(record: HiddenRegimeWorldTransitionRecord) -> int:
    contract = record.delivery_contract
    if contract == PRE_DELIVERED:
        return record.delivered_message
    if contract == DIRECT_TERNARY_CHANNEL:
        return record.helper_message
    if contract == CONSTANT_ZERO_TERNARY_CHANNEL:
        return 0
    if contract == CONSTANT_ONE_TERNARY_CHANNEL:
        return 1
    if contract == CONSTANT_TWO_TERNARY_CHANNEL:
        return 2
    if contract == SHUFFLED_TERNARY_CHANNEL:
        draw_key, _ = jr.split(_wrapped_key(record.old_state.channel_key_data))
        return int(
            np.asarray(
                jr.randint(
                    draw_key,
                    (),
                    0,
                    N_HIDDEN_REGIME_SYMBOLS,
                    dtype=jnp.int32,
                )
            )
        )
    raise HiddenRegimeWorldOracleInputError("delivery contract is not supported")


def reconstruct_hidden_regime_world_transition(
    record: HiddenRegimeWorldTransitionRecord,
) -> HiddenRegimeWorldTransitionExpectation:
    """Reconstruct one world transition without calling the world implementation."""

    if record.schema != HIDDEN_REGIME_WORLD_ORACLE_SCHEMA:
        raise HiddenRegimeWorldOracleInputError("record.schema is not supported")
    if record.delivery_contract not in _DELIVERY_CONTRACTS:
        raise HiddenRegimeWorldOracleInputError("delivery contract is not supported")
    record.config.validate()
    _validate_state(record.old_state, record.config)
    for name, symbol in (
        ("helper_message", record.helper_message),
        ("delivered_message", record.delivered_message),
        ("beneficiary_action", record.beneficiary_action),
    ):
        if type(symbol) is not int or not 0 <= symbol < N_HIDDEN_REGIME_SYMBOLS:
            raise HiddenRegimeWorldOracleInputError(f"{name} is outside the ternary domain")

    config = record.config
    old = record.old_state
    position = old.schedule_position
    segment_ends = np.cumsum(np.asarray(config.segment_lengths, dtype=np.int64))
    segment_index = int(np.sum(position >= segment_ends))
    previous_end = 0 if segment_index == 0 else int(segment_ends[segment_index - 1])
    segment_step = position - previous_end
    regime_id = config.segment_regimes[segment_index]
    target = config.regime_permutations[regime_id][old.cue]
    reward = float(np.float32(record.beneficiary_action == target))

    cue_draw_key, next_cue_key = jr.split(_wrapped_key(old.cue_key_data))
    next_cue = int(
        np.asarray(
            jr.randint(
                cue_draw_key,
                (),
                0,
                N_HIDDEN_REGIME_SYMBOLS,
                dtype=jnp.int32,
            )
        )
    )
    _, next_channel_key = jr.split(_wrapped_key(old.channel_key_data))
    next_step_count = old.step_count + 1 if old.step_count < _INT32_MAX else old.step_count
    final_position = config.total_schedule_steps - 1
    if config.repeat_schedule and position == final_position:
        next_schedule_position = 0
    elif position < final_position:
        next_schedule_position = position + 1
    else:
        next_schedule_position = position

    next_state = HiddenRegimeWorldStateSnapshot(
        cue_key_data=_key_data(next_cue_key, "expected.cue_key"),
        channel_key_data=_key_data(next_channel_key, "expected.channel_key"),
        cue=next_cue,
        step_count=next_step_count,
        schedule_position=next_schedule_position,
    )
    diagnostics = HiddenRegimeWorldDiagnosticsSnapshot(
        observation_helper_cue=old.cue,
        reward=reward,
        next_observation_helper_cue=next_cue,
        terminated=False,
        discount=1.0,
        oracle_step_count=old.step_count,
        oracle_segment_index=segment_index,
        oracle_segment_step=segment_step,
        oracle_regime_id=regime_id,
        oracle_target=target,
    )
    return HiddenRegimeWorldTransitionExpectation(
        delivered_message=_expected_delivery(record),
        diagnostics=diagnostics,
        new_state=next_state,
    )


def _float_equal(left: float, right: float) -> bool:
    left_bits = np.asarray(np.float32(left)).view(np.uint32)
    right_bits = np.asarray(np.float32(right)).view(np.uint32)
    return bool(left_bits == right_bits)


def _compare(
    expected: object,
    actual: object,
    path: str,
    mismatches: list[str],
) -> None:
    if dataclasses.is_dataclass(expected):
        if type(expected) is not type(actual):
            mismatches.append(path)
            return
        for field in dataclasses.fields(expected):
            _compare(
                getattr(expected, field.name),
                getattr(actual, field.name),
                f"{path}.{field.name}",
                mismatches,
            )
    elif isinstance(expected, tuple):
        if not isinstance(actual, tuple) or len(expected) != len(actual):
            mismatches.append(path)
            return
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=True)):
            _compare(expected_item, actual_item, f"{path}[{index}]", mismatches)
    elif isinstance(expected, float):
        if not isinstance(actual, Real) or isinstance(actual, bool) or not _float_equal(
            expected, float(actual)
        ):
            mismatches.append(path)
    elif expected != actual:
        mismatches.append(path)


def validate_hidden_regime_world_transition(
    record: HiddenRegimeWorldTransitionRecord,
) -> HiddenRegimeWorldOracleValidation:
    """Validate strict syntax and compare every reconstructed leaf."""

    try:
        canonical = HiddenRegimeWorldTransitionRecord.from_dict(record.to_dict())
        expected = reconstruct_hidden_regime_world_transition(canonical)
        _validate_state(canonical.new_state, canonical.config)
    except (
        IndexError,
        KeyError,
        OverflowError,
        HiddenRegimeWorldOracleInputError,
        TypeError,
        ValueError,
    ) as error:
        return HiddenRegimeWorldOracleValidation(
            valid=False,
            mismatches=(f"input: {error}",),
            expected=None,
        )
    mismatches: list[str] = []
    _compare(
        expected.delivered_message,
        canonical.delivered_message,
        "delivered_message",
        mismatches,
    )
    _compare(expected.diagnostics, canonical.diagnostics, "diagnostics", mismatches)
    _compare(expected.new_state, canonical.new_state, "new_state", mismatches)
    return HiddenRegimeWorldOracleValidation(
        valid=not mismatches,
        mismatches=tuple(mismatches),
        expected=expected,
    )


def validate_hidden_regime_world_state_snapshot(
    state: HiddenRegimeWorldStateSnapshot,
    config: HiddenRegimeWorldOracleConfig,
) -> HiddenRegimeWorldStateValidation:
    """Validate a standalone state against its exact bounded schedule."""

    try:
        canonical_config = HiddenRegimeWorldOracleConfig.from_dict(config.to_dict())
        canonical_state = HiddenRegimeWorldStateSnapshot.from_dict(state.to_dict())
        _validate_state(canonical_state, canonical_config)
    except (
        IndexError,
        KeyError,
        OverflowError,
        HiddenRegimeWorldOracleInputError,
        TypeError,
        ValueError,
    ) as error:
        return HiddenRegimeWorldStateValidation(
            valid=False,
            mismatches=(f"input: {error}",),
        )
    return HiddenRegimeWorldStateValidation(valid=True, mismatches=())


def validate_hidden_regime_world_state_continuity(
    previous_new_state: HiddenRegimeWorldStateSnapshot,
    current_old_state: HiddenRegimeWorldStateSnapshot,
) -> HiddenRegimeWorldStateValidation:
    """Require bit-exact continuity across adjacent world records."""

    mismatches: list[str] = []
    _compare(previous_new_state, current_old_state, "continuity", mismatches)
    return HiddenRegimeWorldStateValidation(
        valid=not mismatches,
        mismatches=tuple(mismatches),
    )
