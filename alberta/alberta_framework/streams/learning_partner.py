# mypy: disable-error-code="call-arg"
"""Minimal recurring Lewis game for genuine learning-partner experiments.

The world separates the two roles asymmetrically.  A helper privately observes
a fair binary cue ``x`` and emits one binary message.  A beneficiary observes
only that delivered message and a public recurring context, then emits one
binary action.  The common reward is one exactly when the beneficiary action
matches ``x XOR context``.

The target is evaluator-only oracle data.  It is never part of either role's
ordinary observation.  Cue and channel randomness use named independent keys,
so a shuffled-channel control cannot accidentally consume or perturb the cue
stream.  There are no episode resets: contexts alternate on a fixed schedule.

``step_words`` is the authoritative unsigned two-word lifetime identity.
``step_count`` and the scalar phase/cycle indices are saturating int32
telemetry only.  A valid life therefore keeps an exact schedule through
``2**64 - 1`` events; the all-ones identity is a terminal capacity sentinel
whose transaction is refused without advancing either random stream.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from typing import Any, Literal, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Bool, Float, Int, PRNGKeyArray, UInt

DIRECT_CHANNEL: Literal["direct"] = "direct"
CONSTANT_ZERO_CHANNEL: Literal["constant_0"] = "constant_0"
CONSTANT_ONE_CHANNEL: Literal["constant_1"] = "constant_1"
SHUFFLED_CHANNEL: Literal["shuffled"] = "shuffled"

type LearningPartnerChannel = Literal[
    "direct",
    "constant_0",
    "constant_1",
    "shuffled",
]

LEARNING_PARTNER_CHANNELS: tuple[LearningPartnerChannel, ...] = (
    DIRECT_CHANNEL,
    CONSTANT_ZERO_CHANNEL,
    CONSTANT_ONE_CHANNEL,
    SHUFFLED_CHANNEL,
)

LEARNING_PARTNER_WORLD_CONTRACT_VERSION = "learning-partner-world-v1"
LEARNING_PARTNER_WORLD_CONFIG_SCHEMA = "alberta.learning-partner-world.config.v2"
LEARNING_PARTNER_WORLD_STATE_SCHEMA = "alberta.learning-partner-world.state.v2"
LEARNING_PARTNER_WORLD_INPUT_SCHEMA = "alberta.learning-partner-world.input.v2"
LEARNING_PARTNER_WORLD_OUTPUT_SCHEMA = "alberta.learning-partner-world.output.v2"
LEARNING_PARTNER_WORLD_EXACT_IDENTITY_NBYTES = 8

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1

# Stable tags are part of the development-world contract.  Adding a random
# consumer must use a new tag rather than splitting an existing substream.
_CUE_RNG_TAG = 0x435545  # ASCII "CUE"
_CHANNEL_RNG_TAG = 0x43484E  # ASCII "CHN"


def _require_prng_key_contract(key: Any, *, name: str) -> None:
    """Require one scalar typed or legacy JAX PRNG key."""

    if not isinstance(key, Array):
        raise TypeError(f"{name} must be a scalar JAX PRNG key")
    try:
        words = jnp.asarray(jr.key_data(key))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a scalar JAX PRNG key") from error
    if words.shape != (2,) or words.dtype != jnp.dtype(jnp.uint32):
        raise TypeError(f"{name} must be a scalar JAX PRNG key")


@dataclasses.dataclass(frozen=True)
class LearningPartnerWorldConfig:
    """Strictly versioned schedule for the continuing binary signaling game."""

    # Keep phase_length first: pre-v2 callers used its positional constructor.
    phase_length: int = 512
    contract_version: str = LEARNING_PARTNER_WORLD_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if type(self.phase_length) is not int or not 1 <= self.phase_length <= _INT32_MAX:
            raise ValueError("phase_length must be an integer in [1, 2**31 - 1]")
        if self.contract_version != LEARNING_PARTNER_WORLD_CONTRACT_VERSION:
            raise ValueError(
                "contract_version must be "
                f"{LEARNING_PARTNER_WORLD_CONTRACT_VERSION!r}"
            )

    def to_config(self) -> dict[str, Any]:
        """Return the complete deterministic runtime contract."""

        return {
            "schema": LEARNING_PARTNER_WORLD_CONFIG_SCHEMA,
            "state_schema": LEARNING_PARTNER_WORLD_STATE_SCHEMA,
            "input_schema": LEARNING_PARTNER_WORLD_INPUT_SCHEMA,
            "output_schema": LEARNING_PARTNER_WORLD_OUTPUT_SCHEMA,
            "type": type(self).__name__,
            "contract_version": self.contract_version,
            "phase_length": self.phase_length,
        }

    def to_dict(self) -> dict[str, Any]:
        """Compatibility alias for the now-versioned configuration payload."""

        return self.to_config()

    def canonical_json(self) -> str:
        """Serialize with canonical ordering and no non-finite values."""

        return json.dumps(
            self.to_config(),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> LearningPartnerWorldConfig:
        """Strictly reconstruct one v2 configuration."""

        if not isinstance(config, Mapping):
            raise TypeError("learning-partner config must be a mapping")
        payload = dict(config)
        expected = {
            "schema",
            "state_schema",
            "input_schema",
            "output_schema",
            "type",
            "contract_version",
            "phase_length",
        }
        if set(payload) != expected:
            if "schema" not in payload:
                raise ValueError("legacy learning-partner config requires explicit migration")
            raise ValueError("config fields do not match the serialized schema")
        expected_values = {
            "schema": LEARNING_PARTNER_WORLD_CONFIG_SCHEMA,
            "state_schema": LEARNING_PARTNER_WORLD_STATE_SCHEMA,
            "input_schema": LEARNING_PARTNER_WORLD_INPUT_SCHEMA,
            "output_schema": LEARNING_PARTNER_WORLD_OUTPUT_SCHEMA,
            "type": cls.__name__,
        }
        for name, expected_value in expected_values.items():
            value = payload.pop(name)
            if value != expected_value:
                raise ValueError(f"unexpected {name}: {value!r}")
        return cls(**payload)


def migrate_legacy_learning_partner_world_config(
    legacy_config: Mapping[str, Any],
) -> LearningPartnerWorldConfig:
    """Stamp the exact schema-less pre-v2 ``phase_length`` payload."""

    if not isinstance(legacy_config, Mapping):
        raise TypeError("legacy learning-partner config must be a mapping")
    payload = dict(legacy_config)
    if set(payload) != {"phase_length"}:
        missing = sorted({"phase_length"} - set(payload))
        extra = sorted(set(payload) - {"phase_length"})
        raise ValueError(
            "legacy learning-partner config fields are not exact; "
            f"missing={missing}, extra={extra}"
        )
    return LearningPartnerWorldConfig(phase_length=payload["phase_length"])


@chex.dataclass(frozen=True)
class LearningPartnerWorldKeys:
    """Named, independent random streams owned by the world."""

    cue: PRNGKeyArray
    channel: PRNGKeyArray


def learning_partner_world_keys(root_key: Array) -> LearningPartnerWorldKeys:
    """Derive stable named world keys without positional split coupling."""

    _require_prng_key_contract(root_key, name="root_key")
    return LearningPartnerWorldKeys(
        cue=jr.fold_in(root_key, _CUE_RNG_TAG),
        channel=jr.fold_in(root_key, _CHANNEL_RNG_TAG),
    )


@chex.dataclass(frozen=True)
class LearningPartnerObservation:
    """Ordinary observations available before a message is sent.

    ``public_context`` is available to both roles.  ``helper_cue`` is private
    to the helper; an evaluator must not pass it to the beneficiary.
    """

    public_context: Int[Array, ""]
    helper_cue: Int[Array, ""]


@chex.dataclass(frozen=True)
class LearningPartnerWorldState:
    """Fixed-shape state with exact identity and saturating telemetry."""

    cue_key: PRNGKeyArray
    channel_key: PRNGKeyArray
    cue: Int[Array, ""]
    step_count: Int[Array, ""]
    step_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class LearningPartnerOracle:
    """Evaluator-only facts that must never enter either learner policy."""

    step_count: Int[Array, ""]
    phase_index: Int[Array, ""]
    context: Int[Array, ""]
    target: Int[Array, ""]
    step_words: UInt[Array, " 2"]
    phase_words: UInt[Array, " 2"]
    phase_step: Int[Array, ""]
    cycle_index: Int[Array, ""]
    cycle_words: UInt[Array, " 2"]
    next_step_words: UInt[Array, " 2"]
    next_phase_index: Int[Array, ""]
    next_phase_words: UInt[Array, " 2"]
    next_context: Int[Array, ""]
    next_cycle_index: Int[Array, ""]
    next_cycle_words: UInt[Array, " 2"]
    phase_switched: Bool[Array, ""]
    cycle_switched: Bool[Array, ""]


@chex.dataclass(frozen=True)
class LearningPartnerTransition:
    """One continuing signaling transition."""

    observation: LearningPartnerObservation
    helper_message: Int[Array, ""]
    delivered_message: Int[Array, ""]
    beneficiary_action: Int[Array, ""]
    reward: Float[Array, ""]
    next_observation: LearningPartnerObservation
    terminated: Bool[Array, ""]
    discount: Float[Array, ""]
    oracle: LearningPartnerOracle


@chex.dataclass(frozen=True)
class LearningPartnerWorldStepResult:
    """One fail-closed transaction and its exact lifetime audit."""

    transition: LearningPartnerTransition
    state: LearningPartnerWorldState
    pre_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    lifetime_counter_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    state_valid: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    candidate_state_finite: Bool[Array, ""]
    update_applied: Bool[Array, ""]


@dataclasses.dataclass(frozen=True)
class LearningPartnerWorldResourceBudget:
    """Exact logical persistent-state accounting for one world life."""

    state_schema: str
    input_schema: str
    output_schema: str
    observation_int32_scalars: int
    persistent_int32_scalars: int
    exact_identity_uint32_scalars: int
    exact_identity_nbytes: int
    lifetime_identity_bits: int
    telemetry_saturation: int
    rng_uint32_scalars: int
    persistent_state_scalars: int
    state_nbytes: int
    trainable_scalars: int
    replay_capacity: int

    def to_dict(self) -> dict[str, int | str]:
        """Return a JSON-compatible accounting record."""

        return dataclasses.asdict(self)


@chex.dataclass(frozen=True)
class _LearningPartnerSchedule:
    """Internal exact schedule projection for one lifetime identity."""

    phase_index: Int[Array, ""]
    phase_words: UInt[Array, " 2"]
    phase_step: Int[Array, ""]
    cycle_index: Int[Array, ""]
    cycle_words: UInt[Array, " 2"]
    context: Int[Array, ""]


def _array_has_contract(value: Any, shape: tuple[int, ...], dtype: Any) -> bool:
    """Return a nonthrowing exact array shape/dtype predicate."""

    return isinstance(value, Array) and value.shape == shape and value.dtype == dtype


def _checked_words_increment(words: Array) -> tuple[UInt[Array, " 2"], Bool[Array, ""]]:
    """Propose one exact lifetime increment without wrapping all-ones."""

    array = jnp.asarray(words)
    if array.shape != (2,):
        raise ValueError("learning-partner step_words must have shape (2,)")
    if array.dtype != jnp.dtype(jnp.uint32):
        raise TypeError("learning-partner step_words must have dtype uint32")
    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    capacity_available = ~jnp.all(array == maximum)
    low = array[1] + jnp.asarray(1, dtype=jnp.uint32)
    carry = (low == jnp.asarray(0, dtype=jnp.uint32)).astype(jnp.uint32)
    proposed = jnp.stack((array[0] + carry, low)).astype(jnp.uint32)
    return jnp.where(capacity_available, proposed, array), capacity_available


def _words_to_saturating_int32(words: Array) -> Int[Array, ""]:
    """Project exact unsigned words to non-negative int32 telemetry."""

    array = jnp.asarray(words)
    if array.shape != (2,):
        raise ValueError("learning-partner words must have shape (2,)")
    if array.dtype != jnp.dtype(jnp.uint32):
        raise TypeError("learning-partner words must have dtype uint32")
    below_saturation = (array[0] == jnp.asarray(0, dtype=jnp.uint32)) & (
        array[1] < jnp.asarray(_INT32_MAX, dtype=jnp.uint32)
    )
    return jnp.where(
        below_saturation,
        array[1].astype(jnp.int32),
        jnp.asarray(_INT32_MAX, dtype=jnp.int32),
    )


def _lifetime_counter_valid(words: Array, telemetry: Array) -> Bool[Array, ""]:
    """Authenticate saturating telemetry against exact lifetime words."""

    count = jnp.asarray(telemetry)
    if count.shape != ():
        raise ValueError("learning-partner step_count must be scalar")
    if count.dtype != jnp.dtype(jnp.int32):
        raise TypeError("learning-partner step_count must have dtype int32")
    return (count >= 0) & (count == _words_to_saturating_int32(words))


def _divmod_words_by_positive_int32(
    words: Array,
    divisor: Array,
) -> tuple[UInt[Array, " 2"], UInt[Array, ""]]:
    """Exactly divide unsigned two-word time by a positive int32 divisor."""

    array = jnp.asarray(words)
    if array.shape != (2,):
        raise ValueError("learning-partner schedule words must have shape (2,)")
    if array.dtype != jnp.dtype(jnp.uint32):
        raise TypeError("learning-partner schedule words must have dtype uint32")
    divisor_i = jnp.asarray(divisor)
    if divisor_i.shape != ():
        raise ValueError("learning-partner phase length must be scalar")
    if divisor_i.dtype != jnp.dtype(jnp.int32):
        raise TypeError("learning-partner phase length must have dtype int32")
    divisor_u = jnp.maximum(divisor_i, jnp.asarray(1, dtype=jnp.int32)).astype(jnp.uint32)
    zero = jnp.asarray(0, dtype=jnp.uint32)
    one = jnp.asarray(1, dtype=jnp.uint32)

    def divide_narrow(_: None) -> tuple[Array, Array]:
        quotient = jnp.stack((zero, array[1] // divisor_u)).astype(jnp.uint32)
        return quotient, (array[1] % divisor_u).astype(jnp.uint32)

    def divide_bit(
        index: int,
        carry: tuple[Array, Array, Array],
    ) -> tuple[Array, Array, Array]:
        quotient_high, quotient_low, remainder = carry
        bit_index = jnp.asarray(63 - index, dtype=jnp.int32)
        from_high = bit_index >= 32
        shift = jnp.where(from_high, bit_index - 32, bit_index)
        source = jnp.where(from_high, array[0], array[1])
        bit = (source >> shift.astype(jnp.uint32)) & one
        doubled = remainder + remainder + bit
        quotient_bit = doubled >= divisor_u
        next_remainder = jnp.where(quotient_bit, doubled - divisor_u, doubled)
        next_high = (quotient_high << one) | (quotient_low >> jnp.uint32(31))
        next_low = (quotient_low << one) | quotient_bit.astype(jnp.uint32)
        return next_high, next_low, next_remainder

    def divide_wide(_: None) -> tuple[Array, Array]:
        high, low, remainder = jax.lax.fori_loop(
            0,
            64,
            divide_bit,
            (zero, zero, zero),
        )
        return (
            jnp.stack((high, low)).astype(jnp.uint32),
            remainder.astype(jnp.uint32),
        )

    return cast(
        tuple[Array, Array],
        jax.lax.cond(
            array[0] == zero,
            divide_narrow,
            divide_wide,
            operand=None,
        ),
    )


def _words_shift_right_one(words: Array) -> UInt[Array, " 2"]:
    """Return an exact unsigned two-word right shift by one bit."""

    array = jnp.asarray(words)
    high = array[0] >> jnp.asarray(1, dtype=jnp.uint32)
    low = (array[1] >> jnp.asarray(1, dtype=jnp.uint32)) | (
        (array[0] & jnp.asarray(1, dtype=jnp.uint32))
        << jnp.asarray(31, dtype=jnp.uint32)
    )
    return jnp.stack((high, low)).astype(jnp.uint32)


def _floating_tree_is_finite(value: Any) -> Bool[Array, ""]:
    """Require every floating or complex runtime leaf to be finite."""

    valid = jnp.asarray(True, dtype=jnp.bool_)
    for leaf in jax.tree.leaves(value):
        if isinstance(leaf, Array) and jnp.issubdtype(leaf.dtype, jnp.inexact):
            valid = valid & jnp.all(jnp.isfinite(leaf))
    return valid


def _host_field_mapping(legacy_state: Any) -> dict[str, Any]:
    """Return a shallow host mapping for one legacy state record."""

    if isinstance(legacy_state, Mapping):
        return dict(legacy_state)
    if dataclasses.is_dataclass(legacy_state) and not isinstance(legacy_state, type):
        return {
            field.name: getattr(legacy_state, field.name)
            for field in dataclasses.fields(legacy_state)
        }
    raise TypeError("legacy learning-partner state must be a mapping or dataclass")


def migrate_legacy_learning_partner_world_state(
    legacy_state: Any,
) -> LearningPartnerWorldState:
    """Migrate exact pre-v2 state only while int32 time remains unambiguous."""

    fields = _host_field_mapping(legacy_state)
    current_names = {
        field.name for field in dataclasses.fields(cast(Any, LearningPartnerWorldState))
    }
    legacy_names = current_names - {"step_words"}
    if set(fields) != legacy_names:
        missing = sorted(legacy_names - set(fields))
        extra = sorted(set(fields) - legacy_names)
        raise ValueError(
            "legacy learning-partner state fields are not exact; "
            f"missing={missing}, extra={extra}"
        )
    step_array = jnp.asarray(fields["step_count"])
    if step_array.shape != () or step_array.dtype != jnp.dtype(jnp.int32):
        raise TypeError("legacy learning-partner step_count must be scalar int32")
    step = int(jax.device_get(step_array))
    if step < 0:
        raise ValueError("negative legacy learning-partner step_count indicates wrap")
    if step >= _INT32_MAX:
        raise ValueError("saturated legacy learning-partner step_count is ambiguous")
    fields["step_words"] = jnp.asarray((0, step), dtype=jnp.uint32)
    migrated = LearningPartnerWorldState(**fields)
    validator = LearningPartnerWorld()
    validator._require_state_contract(migrated)
    if not bool(jax.device_get(validator._state_values_valid(migrated))):
        raise ValueError("legacy learning-partner state values are inconsistent")
    return migrated


def measure_learning_partner_world_state_nbytes(state: LearningPartnerWorldState) -> int:
    """Measure every persistent JAX leaf in one world state."""

    validator = LearningPartnerWorld()
    validator._require_state_contract(state)
    return sum(int(leaf.nbytes) for leaf in jax.tree.leaves(state) if isinstance(leaf, Array))


class LearningPartnerWorld:
    """Pure-JAX recurring binary Lewis signaling world."""

    def __init__(self, config: LearningPartnerWorldConfig | None = None) -> None:
        self._config = LearningPartnerWorldConfig() if config is None else config
        if not isinstance(self._config, LearningPartnerWorldConfig):
            raise TypeError("config must be a LearningPartnerWorldConfig")
        self._phase_length = jnp.asarray(self._config.phase_length, dtype=jnp.int32)

    @property
    def config(self) -> LearningPartnerWorldConfig:
        """Static world configuration."""

        return self._config

    @property
    def resource_budget(self) -> LearningPartnerWorldResourceBudget:
        """Return exact persistent-state accounting."""

        int_scalars = 2  # cue and saturating lifetime telemetry
        exact_identity_scalars = 2
        rng_scalars = 4
        return LearningPartnerWorldResourceBudget(
            state_schema=LEARNING_PARTNER_WORLD_STATE_SCHEMA,
            input_schema=LEARNING_PARTNER_WORLD_INPUT_SCHEMA,
            output_schema=LEARNING_PARTNER_WORLD_OUTPUT_SCHEMA,
            observation_int32_scalars=2,
            persistent_int32_scalars=int_scalars,
            exact_identity_uint32_scalars=exact_identity_scalars,
            exact_identity_nbytes=LEARNING_PARTNER_WORLD_EXACT_IDENTITY_NBYTES,
            lifetime_identity_bits=64,
            telemetry_saturation=_INT32_MAX,
            rng_uint32_scalars=rng_scalars,
            persistent_state_scalars=int_scalars + exact_identity_scalars + rng_scalars,
            state_nbytes=4 * (int_scalars + exact_identity_scalars + rng_scalars),
            trainable_scalars=0,
            replay_capacity=0,
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize the world without dynamic state."""

        return {"type": type(self).__name__, "config": self._config.to_config()}

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> LearningPartnerWorld:
        """Strictly reconstruct a world from :meth:`to_config` output."""

        if not isinstance(config, Mapping):
            raise TypeError("world config must be a mapping")
        payload = dict(config)
        if set(payload) != {"type", "config"}:
            raise ValueError("world config must contain exactly 'type' and 'config'")
        if payload["type"] != cls.__name__:
            raise ValueError(f"unexpected world type: {payload['type']!r}")
        inner = payload["config"]
        if not isinstance(inner, Mapping):
            raise ValueError("world 'config' must be a mapping")
        return cls(LearningPartnerWorldConfig.from_config(inner))

    @staticmethod
    def _require_state_contract(state: LearningPartnerWorldState) -> None:
        """Validate every fixed state shape and dtype before traced work."""

        if not isinstance(state, LearningPartnerWorldState):
            raise TypeError("state must be a LearningPartnerWorldState")
        _require_prng_key_contract(state.cue_key, name="cue_key")
        _require_prng_key_contract(state.channel_key, name="channel_key")
        contracts = (
            ("cue", state.cue, (), jnp.int32),
            ("step_count", state.step_count, (), jnp.int32),
            ("step_words", state.step_words, (2,), jnp.uint32),
        )
        for name, value, shape, dtype in contracts:
            if not _array_has_contract(value, shape, dtype):
                raise TypeError(
                    f"learning-partner {name} must have shape {shape} and dtype {dtype}"
                )

    @staticmethod
    def _require_binary_input_contract(value: Any, *, name: str) -> Array:
        """Require one scalar integer input before value authentication."""

        array = jnp.asarray(value)
        if array.shape != ():
            raise ValueError(f"{name} must be scalar, got shape {array.shape}")
        if not jnp.issubdtype(array.dtype, jnp.integer):
            raise TypeError(f"{name} must have an integer dtype")
        return array

    @staticmethod
    def _require_channel(channel: Any) -> LearningPartnerChannel:
        """Require one exact host channel identifier."""

        if type(channel) is not str:
            raise TypeError("channel must be a string")
        if channel not in LEARNING_PARTNER_CHANNELS:
            raise ValueError(f"unknown learning-partner channel: {channel!r}")
        return channel

    def _state_values_valid(self, state: LearningPartnerWorldState) -> Bool[Array, ""]:
        """Authenticate cue range and saturating telemetry."""

        cue_valid = (state.cue == 0) | (state.cue == 1)
        return cue_valid & _lifetime_counter_valid(state.step_words, state.step_count)

    def _schedule(self, words: Array) -> _LearningPartnerSchedule:
        """Project exact event words onto the alternating phase schedule."""

        phase_words, phase_step = _divmod_words_by_positive_int32(words, self._phase_length)
        cycle_words = _words_shift_right_one(phase_words)
        return _LearningPartnerSchedule(
            phase_index=_words_to_saturating_int32(phase_words),
            phase_words=phase_words,
            phase_step=phase_step.astype(jnp.int32),
            cycle_index=_words_to_saturating_int32(cycle_words),
            cycle_words=cycle_words,
            context=(phase_words[1] & jnp.asarray(1, dtype=jnp.uint32)).astype(jnp.int32),
        )

    def _source_words(self, source: LearningPartnerWorldState | Array | int) -> Array:
        """Normalize state, exact words, or an unsaturated compatibility scalar."""

        if isinstance(source, LearningPartnerWorldState):
            self._require_state_contract(source)
            return source.step_words
        if type(source) is int:
            if not 0 <= source <= (1 << 64) - 1:
                raise ValueError("integer schedule source must fit uint64")
            return jnp.asarray(
                ((source >> 32) & _UINT32_MAX, source & _UINT32_MAX),
                dtype=jnp.uint32,
            )
        array = jnp.asarray(source)
        if array.shape == (2,) and array.dtype == jnp.dtype(jnp.uint32):
            return array
        if array.shape != ():
            raise ValueError("schedule source must be scalar or uint32[2]")
        if array.dtype not in (jnp.dtype(jnp.int32), jnp.dtype(jnp.uint32)):
            raise TypeError("scalar schedule source must have int32 or uint32 dtype")
        low = jnp.where(array >= 0, array, jnp.asarray(0, dtype=array.dtype)).astype(jnp.uint32)
        return jnp.stack((jnp.asarray(0, dtype=jnp.uint32), low))

    def context_of(self, source: LearningPartnerWorldState | Array | int) -> Array:
        """Return recurring public context from exact state/words or a scalar."""

        return self._schedule(self._source_words(source)).context

    def phase_index_of(self, source: LearningPartnerWorldState | Array | int) -> Array:
        """Return saturating phase telemetry for an exact schedule source."""

        return self._schedule(self._source_words(source)).phase_index

    def phase_step_of(self, source: LearningPartnerWorldState | Array | int) -> Array:
        """Return the exact within-phase offset."""

        return self._schedule(self._source_words(source)).phase_step

    def cycle_index_of(self, source: LearningPartnerWorldState | Array | int) -> Array:
        """Return saturating two-phase-cycle telemetry."""

        return self._schedule(self._source_words(source)).cycle_index

    def init(self, keys: LearningPartnerWorldKeys) -> LearningPartnerWorldState:
        """Initialize the continuing state and sample its first fair cue."""

        if not isinstance(keys, LearningPartnerWorldKeys):
            raise TypeError("keys must be LearningPartnerWorldKeys")
        _require_prng_key_contract(keys.cue, name="keys.cue")
        _require_prng_key_contract(keys.channel, name="keys.channel")
        cue_draw_key, next_cue_key = jr.split(keys.cue)
        cue = jr.randint(cue_draw_key, (), 0, 2, dtype=jnp.int32)
        return LearningPartnerWorldState(
            cue_key=next_cue_key,
            channel_key=keys.channel,
            cue=cue,
            step_count=jnp.asarray(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def observe(self, state: LearningPartnerWorldState) -> LearningPartnerObservation:
        """Build the ordinary pre-message observation, failing closed."""

        self._require_state_contract(state)
        valid = self._state_values_valid(state)
        schedule = self._schedule(state.step_words)
        return LearningPartnerObservation(
            public_context=jnp.where(
                valid,
                schedule.context,
                jnp.asarray(0, dtype=jnp.int32),
            ),
            helper_cue=jnp.where(valid, state.cue, jnp.asarray(0, dtype=jnp.int32)),
        )

    def deliver(
        self,
        state: LearningPartnerWorldState,
        helper_message: Array,
        channel: LearningPartnerChannel,
    ) -> Array:
        """Apply a causal channel without advancing the world."""

        self._require_state_contract(state)
        message = self._require_binary_input_contract(helper_message, name="helper_message")
        channel = self._require_channel(channel)
        if channel == DIRECT_CHANNEL:
            return message.astype(jnp.int32)
        if channel == CONSTANT_ZERO_CHANNEL:
            return jnp.asarray(0, dtype=jnp.int32)
        if channel == CONSTANT_ONE_CHANNEL:
            return jnp.asarray(1, dtype=jnp.int32)
        draw_key, _ = jr.split(state.channel_key)
        return jr.randint(draw_key, (), 0, 2, dtype=jnp.int32)

    def step(
        self,
        state: LearningPartnerWorldState,
        helper_message: Array,
        beneficiary_action: Array,
        channel: LearningPartnerChannel = DIRECT_CHANNEL,
    ) -> tuple[LearningPartnerTransition, LearningPartnerWorldState]:
        """Compatibility wrapper returning transition and committed state."""

        result = self.step_result(state, helper_message, beneficiary_action, channel)
        return result.transition, result.state

    def step_result(
        self,
        state: LearningPartnerWorldState,
        helper_message: Array,
        beneficiary_action: Array,
        channel: LearningPartnerChannel = DIRECT_CHANNEL,
    ) -> LearningPartnerWorldStepResult:
        """Attempt one channel-resolved transition with exact rollback."""

        delivered = self.deliver(state, helper_message, channel)
        return self.step_with_delivery_result(
            state,
            helper_message,
            delivered,
            beneficiary_action,
        )

    def step_with_delivery(
        self,
        state: LearningPartnerWorldState,
        helper_message: Array,
        delivered_message: Array,
        beneficiary_action: Array,
    ) -> tuple[LearningPartnerTransition, LearningPartnerWorldState]:
        """Compatibility wrapper for an evaluator-resolved channel output."""

        result = self.step_with_delivery_result(
            state,
            helper_message,
            delivered_message,
            beneficiary_action,
        )
        return result.transition, result.state

    def step_with_delivery_result(
        self,
        state: LearningPartnerWorldState,
        helper_message: Array,
        delivered_message: Array,
        beneficiary_action: Array,
    ) -> LearningPartnerWorldStepResult:
        """Attempt one explicit-delivery transaction with bit-exact rollback.

        Values outside the binary alphabet reject dynamically.  Shape and
        integer dtype violations reject before tracing.  Every rejected
        transaction returns a finite neutral transition and preserves all
        state leaves, including both PRNG streams.
        """

        self._require_state_contract(state)
        helper = self._require_binary_input_contract(helper_message, name="helper_message")
        delivered = self._require_binary_input_contract(
            delivered_message,
            name="delivered_message",
        )
        action = self._require_binary_input_contract(
            beneficiary_action,
            name="beneficiary_action",
        )
        helper_valid = (helper == 0) | (helper == 1)
        delivered_valid = (delivered == 0) | (delivered == 1)
        action_valid = (action == 0) | (action == 1)
        input_valid = helper_valid & delivered_valid & action_valid
        safe_helper = jnp.where(helper_valid, helper, jnp.asarray(0, dtype=helper.dtype)).astype(
            jnp.int32
        )
        safe_delivered = jnp.where(
            delivered_valid,
            delivered,
            jnp.asarray(0, dtype=delivered.dtype),
        ).astype(jnp.int32)
        safe_action = jnp.where(action_valid, action, jnp.asarray(0, dtype=action.dtype)).astype(
            jnp.int32
        )

        observation = self.observe(state)
        schedule = self._schedule(state.step_words)
        proposed_words, lifetime_capacity_available = _checked_words_increment(state.step_words)
        next_schedule = self._schedule(proposed_words)
        lifetime_counter_valid = _lifetime_counter_valid(state.step_words, state.step_count)
        state_valid = self._state_values_valid(state)

        safe_cue = jnp.where(
            (state.cue == 0) | (state.cue == 1),
            state.cue,
            jnp.asarray(0, dtype=jnp.int32),
        )
        target = jnp.bitwise_xor(safe_cue, schedule.context)
        reward = (safe_action == target).astype(jnp.float32)

        cue_draw_key, next_cue_key = jr.split(state.cue_key)
        next_cue = jr.randint(cue_draw_key, (), 0, 2, dtype=jnp.int32)
        _, next_channel_key = jr.split(state.channel_key)
        candidate_state = LearningPartnerWorldState(
            cue_key=next_cue_key,
            channel_key=next_channel_key,
            cue=next_cue,
            step_count=_words_to_saturating_int32(proposed_words),
            step_words=proposed_words,
        )
        candidate_oracle = LearningPartnerOracle(
            step_count=state.step_count,
            phase_index=schedule.phase_index,
            context=schedule.context,
            target=target,
            step_words=state.step_words,
            phase_words=schedule.phase_words,
            phase_step=schedule.phase_step,
            cycle_index=schedule.cycle_index,
            cycle_words=schedule.cycle_words,
            next_step_words=proposed_words,
            next_phase_index=next_schedule.phase_index,
            next_phase_words=next_schedule.phase_words,
            next_context=next_schedule.context,
            next_cycle_index=next_schedule.cycle_index,
            next_cycle_words=next_schedule.cycle_words,
            phase_switched=~jnp.array_equal(schedule.phase_words, next_schedule.phase_words),
            cycle_switched=~jnp.array_equal(schedule.cycle_words, next_schedule.cycle_words),
        )
        candidate_transition = LearningPartnerTransition(
            observation=observation,
            helper_message=safe_helper,
            delivered_message=safe_delivered,
            beneficiary_action=safe_action,
            reward=reward,
            next_observation=self.observe(candidate_state),
            terminated=jnp.asarray(False, dtype=jnp.bool_),
            discount=jnp.asarray(1.0, dtype=jnp.float32),
            oracle=candidate_oracle,
        )
        candidate_state_finite = _floating_tree_is_finite((candidate_state, candidate_transition))
        update_applied = (
            state_valid & input_valid & lifetime_capacity_available & candidate_state_finite
        )
        committed_state = jax.lax.cond(
            update_applied,
            lambda _: candidate_state,
            lambda _: state,
            operand=None,
        )

        neutral_oracle = LearningPartnerOracle(
            step_count=state.step_count,
            phase_index=jnp.asarray(-1, dtype=jnp.int32),
            context=jnp.asarray(-1, dtype=jnp.int32),
            target=jnp.asarray(-1, dtype=jnp.int32),
            step_words=state.step_words,
            phase_words=jnp.zeros((2,), dtype=jnp.uint32),
            phase_step=jnp.asarray(-1, dtype=jnp.int32),
            cycle_index=jnp.asarray(-1, dtype=jnp.int32),
            cycle_words=jnp.zeros((2,), dtype=jnp.uint32),
            next_step_words=state.step_words,
            next_phase_index=jnp.asarray(-1, dtype=jnp.int32),
            next_phase_words=jnp.zeros((2,), dtype=jnp.uint32),
            next_context=jnp.asarray(-1, dtype=jnp.int32),
            next_cycle_index=jnp.asarray(-1, dtype=jnp.int32),
            next_cycle_words=jnp.zeros((2,), dtype=jnp.uint32),
            phase_switched=jnp.asarray(False, dtype=jnp.bool_),
            cycle_switched=jnp.asarray(False, dtype=jnp.bool_),
        )
        neutral_transition = LearningPartnerTransition(
            observation=observation,
            helper_message=jnp.asarray(-1, dtype=jnp.int32),
            delivered_message=jnp.asarray(-1, dtype=jnp.int32),
            beneficiary_action=jnp.asarray(-1, dtype=jnp.int32),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            next_observation=observation,
            terminated=~lifetime_capacity_available,
            discount=jnp.asarray(0.0, dtype=jnp.float32),
            oracle=neutral_oracle,
        )
        transition = jax.lax.cond(
            update_applied,
            lambda _: candidate_transition,
            lambda _: neutral_transition,
            operand=None,
        )
        return LearningPartnerWorldStepResult(
            transition=transition,
            state=committed_state,
            pre_step_words=state.step_words,
            post_step_words=committed_state.step_words,
            lifetime_counter_valid=lifetime_counter_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            state_valid=state_valid,
            input_valid=input_valid,
            candidate_state_finite=candidate_state_finite,
            update_applied=update_applied,
        )


__all__ = [
    "CONSTANT_ONE_CHANNEL",
    "CONSTANT_ZERO_CHANNEL",
    "DIRECT_CHANNEL",
    "LEARNING_PARTNER_CHANNELS",
    "LEARNING_PARTNER_WORLD_CONFIG_SCHEMA",
    "LEARNING_PARTNER_WORLD_CONTRACT_VERSION",
    "LEARNING_PARTNER_WORLD_EXACT_IDENTITY_NBYTES",
    "LEARNING_PARTNER_WORLD_INPUT_SCHEMA",
    "LEARNING_PARTNER_WORLD_OUTPUT_SCHEMA",
    "LEARNING_PARTNER_WORLD_STATE_SCHEMA",
    "SHUFFLED_CHANNEL",
    "LearningPartnerChannel",
    "LearningPartnerObservation",
    "LearningPartnerOracle",
    "LearningPartnerTransition",
    "LearningPartnerWorld",
    "LearningPartnerWorldConfig",
    "LearningPartnerWorldKeys",
    "LearningPartnerWorldResourceBudget",
    "LearningPartnerWorldState",
    "LearningPartnerWorldStepResult",
    "learning_partner_world_keys",
    "measure_learning_partner_world_state_nbytes",
    "migrate_legacy_learning_partner_world_config",
    "migrate_legacy_learning_partner_world_state",
]
