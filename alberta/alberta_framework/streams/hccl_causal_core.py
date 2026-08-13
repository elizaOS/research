# mypy: disable-error-code="attr-defined,call-arg,no-any-return"
"""Development-only HCCL causal-core world and immutable event receipts.

This is the world/event-receipt rung only.  It provides bounded two-agent
physics, the evaluator-hidden A/B/A/D/A/C/A/B/C/A schedule, exact 16-channel
observations, typed G/V/P signals, immutable source-bound exogenous draws, pure
same-source proposals, and atomic commit/rollback.  The default 8,998-step and
420-step mechanics profiles remain frozen; Core-L2 and Core-L3 extend the
canonical segment lengths over one uninterrupted clock and RNG state, replacing
D with A after cycle zero.  The module implements no agents and authorizes no
benchmark run, artifact, threshold, evidence, or promotion.

The v1 stochastic choices here are explicit Alberta development resolutions,
not paper claims: a fair initial hidden sign; named typed-Threefry streams with
fixed draw counts; a 0.03 hidden-sign flip after current scoring; next-sign cue
flip probabilities 0.25/0.35; a 0.15 outcome flip applied only to convention
factor P; five standard-normal nuisance channels per agent with exactly 10x
variance (sqrt(10) standard-deviation scaling) only for observation index 11
when that agent is at x < -0.8; and physical partner-velocity observation noise
with a conservative fixed standard deviation of 0.01 before normalization.

Event preparation is pure.  A rejected transaction advances no physics,
history, clock, or stream key, so preparing again yields the bit-exact receipt.
Content tags are deterministic integrity checks, not authentication.
Checkpoints are in-memory receipts; this module contains no filesystem writer.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from numbers import Real
from typing import Any, Final, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

HCCL_CAUSAL_CORE_CONFIG_SCHEMA: Final = "alberta.hccl-causal-core.config.v1"
HCCL_CAUSAL_CORE_STATE_SCHEMA: Final = "alberta.hccl-causal-core.state.v1"
HCCL_CAUSAL_CORE_SMOKE_CONFIG_SCHEMA: Final = (
    "alberta.hccl-causal-core.config.mechanics-smoke.v1"
)
HCCL_CAUSAL_CORE_SMOKE_STATE_SCHEMA: Final = (
    "alberta.hccl-causal-core.state.mechanics-smoke.v1"
)
HCCL_CAUSAL_CORE_L2_CONFIG_SCHEMA: Final = "alberta.hccl-causal-core.config.core-l2.v1"
HCCL_CAUSAL_CORE_L2_STATE_SCHEMA: Final = "alberta.hccl-causal-core.state.core-l2.v1"
HCCL_CAUSAL_CORE_L3_CONFIG_SCHEMA: Final = "alberta.hccl-causal-core.config.core-l3.v1"
HCCL_CAUSAL_CORE_L3_STATE_SCHEMA: Final = "alberta.hccl-causal-core.state.core-l3.v1"
HCCL_CAUSAL_CORE_EVENT_SCHEMA: Final = "alberta.hccl-causal-core.event-receipt.v1"
HCCL_CAUSAL_CORE_PROPOSAL_SCHEMA: Final = "alberta.hccl-causal-core.proposal.v1"
HCCL_CAUSAL_CORE_CHECKPOINT_SCHEMA: Final = "alberta.hccl-causal-core.checkpoint.v1"
HCCL_CAUSAL_CORE_RESOURCE_SCHEMA: Final = "alberta.hccl-causal-core.resource.v1"
HCCL_CAUSAL_CORE_STATUS: Final = "l0-development-world-receipt-only-not_assessed"
HCCL_CAUSAL_CORE_EVIDENCE_LEVEL: Final = "L0"

HCCL_REGIME_A: Final = 0
HCCL_REGIME_B: Final = 1
HCCL_REGIME_C: Final = 2
HCCL_REGIME_D: Final = 3
HCCL_CAUSAL_CORE_REGIME_NAMES: Final = ("A", "B", "C", "D")
HCCL_CAUSAL_CORE_CANONICAL_PROFILE: Final = "canonical-8998-v1"
HCCL_CAUSAL_CORE_SMOKE_PROFILE: Final = "mechanics-smoke-420-v1"
HCCL_CAUSAL_CORE_L2_PROFILE: Final = "core-l2-71984-v1"
HCCL_CAUSAL_CORE_L3_PROFILE: Final = "core-l3-1007776-v1"
HCCL_CAUSAL_CORE_SCHEDULE: Final = (
    ("A", 0, 769),
    ("B", 769, 1566),
    ("A", 1566, 2395),
    ("D", 2395, 3252),
    ("A", 3252, 4135),
    ("C", 4135, 5046),
    ("A", 5046, 5987),
    ("B", 5987, 6958),
    ("C", 6958, 7967),
    ("A", 7967, 8998),
)
HCCL_CAUSAL_CORE_SMOKE_SCHEDULE: Final = (
    ("A", 0, 33),
    ("B", 33, 68),
    ("A", 68, 105),
    ("D", 105, 144),
    ("A", 144, 185),
    ("C", 185, 228),
    ("A", 228, 273),
    ("B", 273, 320),
    ("C", 320, 369),
    ("A", 369, 420),
)
HCCL_CAUSAL_CORE_SMOKE_ENTRY_WINDOW_STEPS: Final = 16
HCCL_CAUSAL_CORE_SMOKE_TAIL_WINDOW_STEPS: Final = 16
HCCL_CAUSAL_CORE_EVENT_DRAW_COUNTS: Final = {
    "world_transition": 1,
    "next_cues": 2,
    "outcome_factor": 1,
    "next_nuisance": 10,
    "next_partner_velocity_observation": 2,
}
HCCL_CAUSAL_CORE_INITIAL_DRAW_COUNTS: Final = {
    "initial_hidden_sign": 1,
    "genesis_cues": 2,
    "genesis_nuisance": 10,
    "genesis_partner_velocity_observation": 2,
}
HCCL_CAUSAL_CORE_LEARNER_OBSERVATION_FIELDS: Final = (
    "normalized_own_position",
    "normalized_relative_position",
    "normalized_own_velocity",
    "normalized_noisy_partner_velocity",
    "previous_own_action_sign",
    "previous_partner_action_sign",
    "previous_task_score",
    "previous_own_net_reward",
    "history_available",
    "noisy_hidden_sign_cue_0",
    "noisy_hidden_sign_cue_1",
    "nuisance_0_tv_sensitive",
    "nuisance_1",
    "nuisance_2",
    "nuisance_3",
    "nuisance_4",
)
HCCL_CAUSAL_CORE_LIMITATIONS: Final = (
    "world-and-event-receipt-mechanism-only",
    "no-agents-or-HCCL-causal-core-integration",
    "no-benchmark-execution-or-runbook-authority",
    "no-output-writer-artifact-threshold-validator-evidence-or-promotion",
    "caller-key-material-is-not-a-reserved-consumed-or-held-out-seed",
    "partner-velocity-noise-std-0.01-is-an-Alberta-development-resolution-not-a-paper-claim",
)

_N_AGENTS = 2
_OBSERVATION_DIM = 16
_NUISANCE_DIM = 5
_TAG_WORDS = 4
_WORLD_LIMIT = 1.0
_DAMPING = 0.75
_ACCELERATION = 0.15
_TIME_DELTA = 1.0
_MAX_SPEED = 0.25
_INITIAL_POSITIONS = (-0.5, 0.5)
_WORLD_FLIP_PROBABILITY = 0.03
_CUE_FLIP_PROBABILITIES = (0.25, 0.35)
_OUTCOME_FLIP_PROBABILITY = 0.15
_PARTNER_VELOCITY_NOISE_STD = 0.01
_TV_POSITION_THRESHOLD = -0.8
_TV_VARIANCE_MULTIPLIER = 10.0
_MAXIMUM_TRANSITIONS = 8998
_SMOKE_MAXIMUM_TRANSITIONS = 420
_CORE_L2_CYCLES = 8
_CORE_L3_CYCLES = 112
_UINT32_MAX = 2**32 - 1

_INITIAL_SIGN_STREAM_TAG = 0
_WORLD_STREAM_TAG = 1
_CUE_STREAM_TAG = 2
_OUTCOME_STREAM_TAG = 3
_NUISANCE_STREAM_TAG = 4
_PARTNER_VELOCITY_STREAM_TAG = 5
_OWNER_WORDS = jnp.asarray((0x4843434C, 0x434F5245, 0x45564E54, 0x00000001), dtype=jnp.uint32)
_SMOKE_PROFILE_WORDS = jnp.asarray(
    (0x534D4F4B, 0x45343230, 0x50524631, 0x00000001), dtype=jnp.uint32
)
_CORE_L2_PROFILE_WORDS = jnp.asarray(
    (0x434F5245, 0x4C324C49, 0x46453731, 0x00000001), dtype=jnp.uint32
)
_CORE_L3_PROFILE_WORDS = jnp.asarray(
    (0x434F5245, 0x4C334C49, 0x4645314D, 0x00000001), dtype=jnp.uint32
)
_EVENT_STREAM_NAMES = tuple(HCCL_CAUSAL_CORE_EVENT_DRAW_COUNTS)
_EVENT_DRAW_VECTOR = tuple(HCCL_CAUSAL_CORE_EVENT_DRAW_COUNTS.values())


def _checked_schedule_add(left: int, right: int, *, name: str) -> int:
    if type(left) is not int or type(right) is not int or left < 0 or right < 0:
        raise TypeError(f"{name} inputs must be nonnegative non-boolean integers")
    if left > _UINT32_MAX - right:
        raise OverflowError(f"{name} exceeds the uint32 world-clock limit")
    return left + right


def _checked_schedule_multiply(left: int, right: int, *, name: str) -> int:
    if type(left) is not int or type(right) is not int or left < 0 or right < 0:
        raise TypeError(f"{name} inputs must be nonnegative non-boolean integers")
    if left != 0 and right > _UINT32_MAX // left:
        raise OverflowError(f"{name} exceeds the uint32 world-clock limit")
    return left * right


def _build_longevity_schedule(cycle_count: int) -> tuple[tuple[str, int, int], ...]:
    if type(cycle_count) is not int or cycle_count < 2:
        raise ValueError("longevity cycle_count must be a non-boolean integer of at least two")
    expected_events = _checked_schedule_multiply(
        _MAXIMUM_TRANSITIONS,
        cycle_count,
        name="longevity schedule event count",
    )
    occurrences: list[tuple[str, int, int]] = []
    next_start = 0
    for cycle_index in range(cycle_count):
        for canonical_name, canonical_start, canonical_end in HCCL_CAUSAL_CORE_SCHEDULE:
            length = canonical_end - canonical_start
            next_end = _checked_schedule_add(
                next_start,
                length,
                name="longevity schedule boundary",
            )
            name = "A" if cycle_index > 0 and canonical_name == "D" else canonical_name
            occurrences.append((name, next_start, next_end))
            next_start = next_end
    if next_start != expected_events:
        raise AssertionError("longevity schedule does not exactly cover its checked life")
    return tuple(occurrences)


HCCL_CAUSAL_CORE_L2_SCHEDULE: Final = _build_longevity_schedule(_CORE_L2_CYCLES)
HCCL_CAUSAL_CORE_L3_SCHEDULE: Final = _build_longevity_schedule(_CORE_L3_CYCLES)


def hccl_causal_core_schedule_for_profile(
    schedule_profile: str,
) -> tuple[tuple[str, int, int], ...]:
    """Return the one immutable schedule owned by a fixed versioned profile."""

    if type(schedule_profile) is not str:
        raise TypeError("schedule_profile must be an exact string")
    if schedule_profile == HCCL_CAUSAL_CORE_CANONICAL_PROFILE:
        return HCCL_CAUSAL_CORE_SCHEDULE
    if schedule_profile == HCCL_CAUSAL_CORE_SMOKE_PROFILE:
        return HCCL_CAUSAL_CORE_SMOKE_SCHEDULE
    if schedule_profile == HCCL_CAUSAL_CORE_L2_PROFILE:
        return HCCL_CAUSAL_CORE_L2_SCHEDULE
    if schedule_profile == HCCL_CAUSAL_CORE_L3_PROFILE:
        return HCCL_CAUSAL_CORE_L3_SCHEDULE
    raise ValueError("schedule_profile must select one fixed versioned schedule")


def hccl_causal_core_cycle_count_for_profile(schedule_profile: str) -> int:
    """Return the checked number of uninterrupted canonical-length cycles."""

    if type(schedule_profile) is not str:
        raise TypeError("schedule_profile must be an exact string")
    if schedule_profile in (
        HCCL_CAUSAL_CORE_CANONICAL_PROFILE,
        HCCL_CAUSAL_CORE_SMOKE_PROFILE,
    ):
        return 1
    if schedule_profile == HCCL_CAUSAL_CORE_L2_PROFILE:
        return _CORE_L2_CYCLES
    if schedule_profile == HCCL_CAUSAL_CORE_L3_PROFILE:
        return _CORE_L3_CYCLES
    raise ValueError("schedule_profile must select one fixed versioned schedule")


def hccl_causal_core_lifetime_for_profile(schedule_profile: str) -> int:
    """Return the exact checked committed-transition capacity for a profile."""

    schedule = hccl_causal_core_schedule_for_profile(schedule_profile)
    if not schedule:
        raise AssertionError("fixed HCCL schedule cannot be empty")
    lifetime = schedule[-1][2]
    if type(lifetime) is not int or not 1 <= lifetime <= _UINT32_MAX:
        raise OverflowError("fixed HCCL schedule lifetime exceeds the uint32 world clock")
    return lifetime


def _strict_float(value: object, *, name: str, expected: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be the fixed finite development resolution")
    narrowed = float(np.float32(value))
    if not math.isfinite(float(value)) or narrowed != float(np.float32(expected)):
        raise ValueError(f"{name} must equal the fixed development resolution {expected}")
    return float(expected)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"nonfinite JSON constant {value!r} is forbidden")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_digest(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _typed_threefry_key(key: object) -> bool:
    if not (
        hasattr(key, "shape")
        and tuple(cast(Any, key).shape) == ()
        and jax.dtypes.issubdtype(cast(Any, key).dtype, jax.dtypes.prng_key)
    ):
        return False
    try:
        return str(jr.key_impl(cast(Any, key))) == "threefry2x32"
    except (TypeError, ValueError):
        return False


def _require_array(
    value: object,
    *,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
    name: str,
) -> Array:
    if not hasattr(value, "shape") or tuple(cast(Any, value).shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if jnp.dtype(cast(Any, value).dtype) != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    return cast(Array, value)


def _increment_words(words: Array) -> tuple[Array, Array]:
    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    available = ~jnp.all(words == maximum)
    low = words[1] + jnp.asarray(1, dtype=jnp.uint32)
    carry = (low == 0).astype(jnp.uint32)
    candidate = jnp.stack((words[0] + carry, low)).astype(jnp.uint32)
    return jnp.where(available, candidate, words), available


def _words_at_most_lifetime(
    words: Array, maximum_transitions: int = _MAXIMUM_TRANSITIONS
) -> Array:
    return (words[0] == 0) & (
        words[1] <= jnp.asarray(maximum_transitions, jnp.uint32)
    )


def _words_below_lifetime(
    words: Array, maximum_transitions: int = _MAXIMUM_TRANSITIONS
) -> Array:
    return (words[0] == 0) & (
        words[1] < jnp.asarray(maximum_transitions, jnp.uint32)
    )


def _array_words(value: Array) -> Array:
    if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
        return jr.key_data(value).reshape(-1).astype(jnp.uint32)
    if value.dtype == jnp.dtype(jnp.float32):
        return jax.lax.bitcast_convert_type(value, jnp.uint32).reshape(-1)
    if value.dtype == jnp.dtype(jnp.int32):
        return jax.lax.bitcast_convert_type(value, jnp.uint32).reshape(-1)
    if value.dtype == jnp.dtype(jnp.uint32):
        return value.reshape(-1)
    if value.dtype == jnp.dtype(jnp.bool_):
        return value.astype(jnp.uint32).reshape(-1)
    raise TypeError(f"unsupported tag dtype {value.dtype}")


def _rotate_left(words: Array, amount: int) -> Array:
    shift = jnp.asarray(amount, dtype=jnp.uint32)
    return jnp.bitwise_or(
        jnp.left_shift(words, shift),
        jnp.right_shift(words, jnp.asarray(32 - amount, dtype=jnp.uint32)),
    )


def _content_tag(*values: Array) -> Array:
    tag = _OWNER_WORDS
    constants = jnp.asarray((0x9E3779B9, 0x85EBCA6B, 0xC2B2AE35, 0x27D4EB2F), dtype=jnp.uint32)
    for item_index, value in enumerate(values):
        words = _array_words(value)
        indices = jnp.arange(words.size, dtype=jnp.uint32) + jnp.asarray(
            item_index + 1, dtype=jnp.uint32
        )
        first = jnp.sum(words + constants[item_index % 4] * indices, dtype=jnp.uint32)
        second = jnp.sum(
            (words ^ constants[(item_index + 1) % 4]) * (indices + 1),
            dtype=jnp.uint32,
        )
        third = jnp.bitwise_xor.reduce(words + constants[(item_index + 2) % 4])
        fourth = jnp.sum(_rotate_left(words ^ indices, (item_index % 15) + 1), dtype=jnp.uint32)
        mixed = jnp.stack((first, second, third, fourth)).astype(jnp.uint32)
        tag = _rotate_left(tag ^ mixed, (item_index % 11) + 1) + constants
    return tag.astype(jnp.uint32)


def _tree_exact_equal(left: Any, right: Any) -> Array:
    if type(left) is not type(right):
        return jnp.asarray(False, dtype=jnp.bool_)
    left_leaves, left_structure = jax.tree.flatten(left)
    right_leaves, right_structure = jax.tree.flatten(right)
    if cast(object, left_structure) != cast(object, right_structure) or len(left_leaves) != len(
        right_leaves
    ):
        return jnp.asarray(False, dtype=jnp.bool_)
    comparisons: list[Array] = []
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        if not hasattr(left_leaf, "dtype") or not hasattr(right_leaf, "dtype"):
            comparisons.append(jnp.asarray(left_leaf == right_leaf, dtype=jnp.bool_))
            continue
        left_array = cast(Array, left_leaf)
        right_array = cast(Array, right_leaf)
        if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
            comparisons.append(jnp.asarray(False, dtype=jnp.bool_))
            continue
        comparisons.append(jnp.all(_array_words(left_array) == _array_words(right_array)))
    return jnp.all(jnp.stack(tuple(comparisons)))


@dataclasses.dataclass(frozen=True)
class HCCLCausalCoreConfig:
    """Strict Alberta development resolution for the world-only rung."""

    world_limit: float = _WORLD_LIMIT
    damping: float = _DAMPING
    acceleration: float = _ACCELERATION
    time_delta: float = _TIME_DELTA
    maximum_speed: float = _MAX_SPEED
    initial_positions: tuple[float, float] = _INITIAL_POSITIONS
    world_flip_probability: float = _WORLD_FLIP_PROBABILITY
    cue_flip_probabilities: tuple[float, float] = _CUE_FLIP_PROBABILITIES
    outcome_flip_probability: float = _OUTCOME_FLIP_PROBABILITY
    partner_velocity_observation_noise_std: float = _PARTNER_VELOCITY_NOISE_STD
    tv_position_threshold: float = _TV_POSITION_THRESHOLD
    tv_nuisance_variance_multiplier: float = _TV_VARIANCE_MULTIPLIER
    maximum_committed_transitions: int = _MAXIMUM_TRANSITIONS
    schedule_profile: str = HCCL_CAUSAL_CORE_CANONICAL_PROFILE

    def __post_init__(self) -> None:
        for name, expected in (
            ("world_limit", _WORLD_LIMIT),
            ("damping", _DAMPING),
            ("acceleration", _ACCELERATION),
            ("time_delta", _TIME_DELTA),
            ("maximum_speed", _MAX_SPEED),
            ("world_flip_probability", _WORLD_FLIP_PROBABILITY),
            ("outcome_flip_probability", _OUTCOME_FLIP_PROBABILITY),
            ("partner_velocity_observation_noise_std", _PARTNER_VELOCITY_NOISE_STD),
            ("tv_position_threshold", _TV_POSITION_THRESHOLD),
            ("tv_nuisance_variance_multiplier", _TV_VARIANCE_MULTIPLIER),
        ):
            object.__setattr__(
                self, name, _strict_float(getattr(self, name), name=name, expected=expected)
            )
        if (
            type(self.initial_positions) is not tuple
            or len(self.initial_positions) != 2
            or any(type(value) is not float for value in self.initial_positions)
            or self.initial_positions != _INITIAL_POSITIONS
        ):
            raise ValueError("initial_positions must equal the fixed development resolution")
        if (
            type(self.cue_flip_probabilities) is not tuple
            or len(self.cue_flip_probabilities) != 2
            or any(type(value) is not float for value in self.cue_flip_probabilities)
            or self.cue_flip_probabilities != _CUE_FLIP_PROBABILITIES
        ):
            raise ValueError("cue_flip_probabilities must equal the fixed development resolution")
        if type(self.schedule_profile) is not str or self.schedule_profile not in (
            HCCL_CAUSAL_CORE_CANONICAL_PROFILE,
            HCCL_CAUSAL_CORE_SMOKE_PROFILE,
            HCCL_CAUSAL_CORE_L2_PROFILE,
            HCCL_CAUSAL_CORE_L3_PROFILE,
        ):
            raise ValueError("schedule_profile must select one fixed versioned schedule")
        expected_maximum = hccl_causal_core_lifetime_for_profile(self.schedule_profile)
        if (
            type(self.maximum_committed_transitions) is not int
            or self.maximum_committed_transitions != expected_maximum
        ):
            raise ValueError(
                "maximum_committed_transitions must equal the fixed schedule resolution"
            )

    @classmethod
    def mechanics_smoke(cls) -> HCCLCausalCoreConfig:
        """Select the authenticated 420-event CI mechanics schedule."""

        return cls(
            maximum_committed_transitions=_SMOKE_MAXIMUM_TRANSITIONS,
            schedule_profile=HCCL_CAUSAL_CORE_SMOKE_PROFILE,
        )

    @classmethod
    def core_l2(cls) -> HCCLCausalCoreConfig:
        """Select the uninterrupted eight-cycle 71,984-event Core-L2 life."""

        return cls(
            maximum_committed_transitions=hccl_causal_core_lifetime_for_profile(
                HCCL_CAUSAL_CORE_L2_PROFILE
            ),
            schedule_profile=HCCL_CAUSAL_CORE_L2_PROFILE,
        )

    @classmethod
    def core_l3(cls) -> HCCLCausalCoreConfig:
        """Select the uninterrupted 112-cycle 1,007,776-event Core-L3 life."""

        return cls(
            maximum_committed_transitions=hccl_causal_core_lifetime_for_profile(
                HCCL_CAUSAL_CORE_L3_PROFILE
            ),
            schedule_profile=HCCL_CAUSAL_CORE_L3_PROFILE,
        )


@chex.dataclass(frozen=True)
class HCCLCausalCoreState:
    world_key: Array
    cue_key: Array
    outcome_key: Array
    nuisance_key: Array
    partner_velocity_key: Array
    positions: Float[Array, " 2"]
    velocities: Float[Array, " 2"]
    current_cues: Float[Array, " 2"]
    current_nuisance: Float[Array, "2 5"]
    current_partner_velocity_noise: Float[Array, " 2"]
    hidden_sign: Float[Array, ""]
    previous_action_signs: Float[Array, " 2"]
    previous_task_score: Float[Array, ""]
    previous_net_reward: Float[Array, " 2"]
    history_available: Bool[Array, ""]
    step_words: UInt[Array, " 2"]
    content_tag_words: UInt[Array, " 4"]


@chex.dataclass(frozen=True)
class HCCLCausalCoreEventReceipt:
    source_state_tag_words: UInt[Array, " 4"]
    source_step_words: UInt[Array, " 2"]
    world_flipped: Bool[Array, ""]
    next_cue_flipped: Bool[Array, " 2"]
    outcome_flipped: Bool[Array, ""]
    nuisance_standard_normal: Float[Array, "2 5"]
    partner_velocity_standard_normal: Float[Array, " 2"]
    next_world_key: Array
    next_cue_key: Array
    next_outcome_key: Array
    next_nuisance_key: Array
    next_partner_velocity_key: Array
    draw_counts: Int[Array, " 5"]
    content_tag_words: UInt[Array, " 4"]

    @property
    def stream_names(self) -> tuple[str, ...]:
        return _EVENT_STREAM_NAMES


@chex.dataclass(frozen=True)
class HCCLCausalCoreFactors:
    gathering: Float[Array, ""]
    velocity: Float[Array, ""]
    convention_clean: Float[Array, ""]
    convention_noisy: Float[Array, ""]


@chex.dataclass(frozen=True)
class HCCLCausalCoreTypedSignals:
    task_score: Float[Array, ""]
    net_reward: Float[Array, " 2"]
    message_charge: Float[Array, " 2"]
    safety_cost: Float[Array, " 2"]


@chex.dataclass(frozen=True)
class HCCLCausalCoreProposal:
    source_state_tag_words: UInt[Array, " 4"]
    source_step_words: UInt[Array, " 2"]
    event_content_tag_words: UInt[Array, " 4"]
    joint_action_ids: Int[Array, " 2"]
    action_signs: Float[Array, " 2"]
    observation: Float[Array, "2 16"]
    next_observation: Float[Array, "2 16"]
    candidate_state: HCCLCausalCoreState
    factors: HCCLCausalCoreFactors
    signals: HCCLCausalCoreTypedSignals
    evaluator_regime_id: Int[Array, ""]
    current_hidden_sign: Float[Array, ""]
    world_flipped: Bool[Array, ""]
    next_hidden_sign: Float[Array, ""]
    next_cue_flipped: Bool[Array, " 2"]
    outcome_flipped: Bool[Array, ""]
    valid: Bool[Array, ""]
    content_tag_words: UInt[Array, " 4"]


@chex.dataclass(frozen=True)
class HCCLCausalCoreStepResult:
    state: HCCLCausalCoreState
    proposal: HCCLCausalCoreProposal
    pre_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    source_state_valid: Bool[Array, ""]
    event_receipt_valid: Bool[Array, ""]
    proposal_valid: Bool[Array, ""]
    downstream_candidate_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    update_applied: Bool[Array, ""]
    update_rejected: Bool[Array, ""]


@chex.dataclass(frozen=True)
class HCCLCausalCoreScanResult:
    state: HCCLCausalCoreState
    observations: Float[Array, "steps 2 16"]
    next_observations: Float[Array, "steps 2 16"]
    task_scores: Float[Array, " steps"]
    regime_ids: Int[Array, " steps"]
    event_content_tag_words: UInt[Array, "steps 4"]
    post_step_words: UInt[Array, "steps 2"]
    update_applied: Bool[Array, " steps"]


@dataclasses.dataclass(frozen=True)
class HCCLCausalCoreResourceBudget:
    schema: str
    persistent_state_nbytes: int
    event_receipt_nbytes: int
    proposal_nbytes: int
    observation_float32_scalars_per_state: int
    initialization_draws: int
    event_draws_per_receipt: int
    world_draws_per_receipt: int
    cue_draws_per_receipt: int
    outcome_draws_per_receipt: int
    nuisance_draws_per_receipt: int
    partner_velocity_draws_per_receipt: int
    maximum_committed_transitions: int
    output_write_calls: int
    artifact_bytes_written: int

    def to_config(self) -> dict[str, object]:
        return cast(dict[str, object], dataclasses.asdict(self))


@dataclasses.dataclass(frozen=True)
class HCCLCausalCoreCheckpoint:
    schema: str
    mechanism_status: str
    evidence_level: str
    output_writes_authorized: bool
    artifact_authorized: bool
    evidence_authorized: bool
    config: dict[str, object]
    config_sha256: str
    state: HCCLCausalCoreState
    state_nbytes: int
    state_sha256: str
    checkpoint_sha256: str


def _state_tag(state: HCCLCausalCoreState) -> Array:
    return _content_tag(
        state.world_key,
        state.cue_key,
        state.outcome_key,
        state.nuisance_key,
        state.partner_velocity_key,
        state.positions,
        state.velocities,
        state.current_cues,
        state.current_nuisance,
        state.current_partner_velocity_noise,
        state.hidden_sign,
        state.previous_action_signs,
        state.previous_task_score,
        state.previous_net_reward,
        state.history_available,
        state.step_words,
    )


def _state_tag_for_profile(state: HCCLCausalCoreState, schedule_profile: str) -> Array:
    canonical = _state_tag(state)
    if schedule_profile == HCCL_CAUSAL_CORE_CANONICAL_PROFILE:
        return canonical
    if schedule_profile == HCCL_CAUSAL_CORE_SMOKE_PROFILE:
        return _content_tag(_SMOKE_PROFILE_WORDS, canonical)
    if schedule_profile == HCCL_CAUSAL_CORE_L2_PROFILE:
        return _content_tag(_CORE_L2_PROFILE_WORDS, canonical)
    if schedule_profile == HCCL_CAUSAL_CORE_L3_PROFILE:
        return _content_tag(_CORE_L3_PROFILE_WORDS, canonical)
    raise ValueError("unsupported HCCL causal-core schedule profile")


def _event_tag(receipt: HCCLCausalCoreEventReceipt) -> Array:
    return _content_tag(
        receipt.source_state_tag_words,
        receipt.source_step_words,
        receipt.world_flipped,
        receipt.next_cue_flipped,
        receipt.outcome_flipped,
        receipt.nuisance_standard_normal,
        receipt.partner_velocity_standard_normal,
        receipt.next_world_key,
        receipt.next_cue_key,
        receipt.next_outcome_key,
        receipt.next_nuisance_key,
        receipt.next_partner_velocity_key,
        receipt.draw_counts,
    )


def _proposal_tag(proposal: HCCLCausalCoreProposal) -> Array:
    return _content_tag(
        proposal.source_state_tag_words,
        proposal.source_step_words,
        proposal.event_content_tag_words,
        proposal.joint_action_ids,
        proposal.action_signs,
        proposal.observation,
        proposal.next_observation,
        proposal.candidate_state.content_tag_words,
        proposal.factors.gathering,
        proposal.factors.velocity,
        proposal.factors.convention_clean,
        proposal.factors.convention_noisy,
        proposal.signals.task_score,
        proposal.signals.net_reward,
        proposal.signals.message_charge,
        proposal.signals.safety_cost,
        proposal.evaluator_regime_id,
        proposal.current_hidden_sign,
        proposal.world_flipped,
        proposal.next_hidden_sign,
        proposal.next_cue_flipped,
        proposal.outcome_flipped,
        proposal.valid,
    )


class HCCLCausalCoreWorld:
    """Pure bounded world and source-bound exogenous receipt transaction."""

    def __init__(self, config: HCCLCausalCoreConfig):
        if type(config) is not HCCLCausalCoreConfig:
            raise TypeError("config must be exact HCCLCausalCoreConfig")
        self._config = config

    @property
    def config(self) -> HCCLCausalCoreConfig:
        return self._config

    @property
    def schedule(self) -> tuple[tuple[str, int, int], ...]:
        return hccl_causal_core_schedule_for_profile(self._config.schedule_profile)

    @property
    def learner_observation_fields(self) -> tuple[str, ...]:
        return HCCL_CAUSAL_CORE_LEARNER_OBSERVATION_FIELDS

    def to_config(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "type": "HCCLCausalCoreWorld",
            "schema": HCCL_CAUSAL_CORE_CONFIG_SCHEMA,
            "state_schema": HCCL_CAUSAL_CORE_STATE_SCHEMA,
            "event_schema": HCCL_CAUSAL_CORE_EVENT_SCHEMA,
            "proposal_schema": HCCL_CAUSAL_CORE_PROPOSAL_SCHEMA,
            "checkpoint_schema": HCCL_CAUSAL_CORE_CHECKPOINT_SCHEMA,
            "mechanism_status": HCCL_CAUSAL_CORE_STATUS,
            "evidence_level": HCCL_CAUSAL_CORE_EVIDENCE_LEVEL,
            "world_limit": self._config.world_limit,
            "damping": self._config.damping,
            "acceleration": self._config.acceleration,
            "time_delta": self._config.time_delta,
            "maximum_speed": self._config.maximum_speed,
            "initial_positions": list(self._config.initial_positions),
            "action_id_domain": [0, 1],
            "action_sign_mapping": {"0": -1.0, "1": 1.0},
            "history_action_encoding": "signed-minus1-plus1-genesis-neutral-zero",
            "initial_hidden_sign_distribution": "fair-bernoulli-minus1-plus1",
            "prng_implementation": "threefry2x32",
            "named_streams": [
                "initial_hidden_sign",
                *_EVENT_STREAM_NAMES,
            ],
            "initial_draw_counts": dict(HCCL_CAUSAL_CORE_INITIAL_DRAW_COUNTS),
            "event_draw_counts": dict(HCCL_CAUSAL_CORE_EVENT_DRAW_COUNTS),
            "world_flip_probability": self._config.world_flip_probability,
            "world_flip_timing": "after-current-scoring",
            "cue_flip_probabilities": list(self._config.cue_flip_probabilities),
            "cue_target": "next-hidden-sign",
            "outcome_flip_probability": self._config.outcome_flip_probability,
            "outcome_flip_scope": "convention-factor-P-only",
            "convention_factors_exposed": ["clean_P", "noisy_P"],
            "nuisance_distribution": "standard-normal-five-channels-per-agent",
            "tv_position_threshold": self._config.tv_position_threshold,
            "tv_nuisance_observation_index": 11,
            "tv_nuisance_variance_multiplier": self._config.tv_nuisance_variance_multiplier,
            "partner_velocity_observation_noise_std": (
                self._config.partner_velocity_observation_noise_std
            ),
            "partner_velocity_noise_units": "physical-velocity-before-normalization",
            "schedule": [
                {"regime": name, "start": start, "end": end}
                for name, start, end in self.schedule
            ],
            "maximum_committed_transitions": self._config.maximum_committed_transitions,
            "learner_observation_width": _OBSERVATION_DIM,
            "learner_observation_fields": list(self.learner_observation_fields),
            "typed_signal_fields": [
                "task_score",
                "net_reward",
                "message_charge",
                "safety_cost",
            ],
            "causal_core_message_charge": 0.0,
            "causal_core_safety_cost": 0.0,
            "agent_implementation_present": False,
            "causal_core_integration_complete": False,
            "benchmark_execution_authorized": False,
            "artifact_authorized": False,
            "threshold_authorized": False,
            "evidence_authorized": False,
            "promotion_authorized": False,
            "output_writes_authorized": False,
            "seed_reservation_or_consumption_authorized": False,
            "limitations": list(HCCL_CAUSAL_CORE_LIMITATIONS),
        }
        if self._config.schedule_profile == HCCL_CAUSAL_CORE_SMOKE_PROFILE:
            payload["schema"] = HCCL_CAUSAL_CORE_SMOKE_CONFIG_SCHEMA
            payload["state_schema"] = HCCL_CAUSAL_CORE_SMOKE_STATE_SCHEMA
            payload["schedule_profile"] = HCCL_CAUSAL_CORE_SMOKE_PROFILE
            payload["entry_window_steps"] = HCCL_CAUSAL_CORE_SMOKE_ENTRY_WINDOW_STEPS
            payload["tail_window_steps"] = HCCL_CAUSAL_CORE_SMOKE_TAIL_WINDOW_STEPS
        elif self._config.schedule_profile in (
            HCCL_CAUSAL_CORE_L2_PROFILE,
            HCCL_CAUSAL_CORE_L3_PROFILE,
        ):
            if self._config.schedule_profile == HCCL_CAUSAL_CORE_L2_PROFILE:
                payload["schema"] = HCCL_CAUSAL_CORE_L2_CONFIG_SCHEMA
                payload["state_schema"] = HCCL_CAUSAL_CORE_L2_STATE_SCHEMA
            else:
                payload["schema"] = HCCL_CAUSAL_CORE_L3_CONFIG_SCHEMA
                payload["state_schema"] = HCCL_CAUSAL_CORE_L3_STATE_SCHEMA
            payload["schedule_profile"] = self._config.schedule_profile
            payload["cycle_count"] = hccl_causal_core_cycle_count_for_profile(
                self._config.schedule_profile
            )
            payload["reset_callbacks_present"] = False
            payload["boundary_callbacks_present"] = False
            payload["cycle_reseeding_present"] = False
        return payload

    def to_json(self) -> str:
        """Serialize the exact world declaration as canonical strict JSON."""

        return _canonical_bytes(self.to_config()).decode("utf-8")

    @classmethod
    def from_config(cls, payload: dict[str, object]) -> HCCLCausalCoreWorld:
        if type(payload) is not dict:
            raise TypeError("config payload must be an exact dict")
        canonical = cls(HCCLCausalCoreConfig()).to_config()
        if _canonical_bytes(payload) == _canonical_bytes(canonical):
            return cls(HCCLCausalCoreConfig())
        smoke = cls(HCCLCausalCoreConfig.mechanics_smoke()).to_config()
        if _canonical_bytes(payload) == _canonical_bytes(smoke):
            return cls(HCCLCausalCoreConfig.mechanics_smoke())
        core_l2 = cls(HCCLCausalCoreConfig.core_l2()).to_config()
        if _canonical_bytes(payload) == _canonical_bytes(core_l2):
            return cls(HCCLCausalCoreConfig.core_l2())
        core_l3 = cls(HCCLCausalCoreConfig.core_l3()).to_config()
        if _canonical_bytes(payload) == _canonical_bytes(core_l3):
            return cls(HCCLCausalCoreConfig.core_l3())
        raise ValueError("HCCL causal-core config differs from a fixed development resolution")

    @classmethod
    def from_json(cls, payload: str) -> HCCLCausalCoreWorld:
        """Parse one complete declaration, rejecting duplicates and nonfinite values."""

        if type(payload) is not str:
            raise TypeError("HCCL causal-core JSON must be an exact string")
        try:
            decoded = json.loads(
                payload,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("HCCL causal-core JSON is invalid or non-strict") from error
        if type(decoded) is not dict:
            raise ValueError("HCCL causal-core JSON must encode one object")
        return cls.from_config(decoded)

    def _state_tag(self, state: HCCLCausalCoreState) -> Array:
        return _state_tag_for_profile(state, self._config.schedule_profile)

    def _require_state_contract(self, state: HCCLCausalCoreState) -> None:
        if type(state) is not HCCLCausalCoreState:
            raise TypeError("state must be exact HCCLCausalCoreState")
        for name in (
            "world_key",
            "cue_key",
            "outcome_key",
            "nuisance_key",
            "partner_velocity_key",
        ):
            if not _typed_threefry_key(getattr(state, name)):
                raise TypeError(f"state.{name} must be a scalar typed Threefry key")
        _require_array(state.positions, shape=(2,), dtype=jnp.dtype(jnp.float32), name="positions")
        _require_array(
            state.velocities, shape=(2,), dtype=jnp.dtype(jnp.float32), name="velocities"
        )
        _require_array(
            state.current_cues, shape=(2,), dtype=jnp.dtype(jnp.float32), name="current_cues"
        )
        _require_array(
            state.current_nuisance,
            shape=(2, 5),
            dtype=jnp.dtype(jnp.float32),
            name="current_nuisance",
        )
        _require_array(
            state.current_partner_velocity_noise,
            shape=(2,),
            dtype=jnp.dtype(jnp.float32),
            name="current_partner_velocity_noise",
        )
        _require_array(
            state.hidden_sign, shape=(), dtype=jnp.dtype(jnp.float32), name="hidden_sign"
        )
        _require_array(
            state.previous_action_signs,
            shape=(2,),
            dtype=jnp.dtype(jnp.float32),
            name="previous_action_signs",
        )
        _require_array(
            state.previous_task_score,
            shape=(),
            dtype=jnp.dtype(jnp.float32),
            name="previous_task_score",
        )
        _require_array(
            state.previous_net_reward,
            shape=(2,),
            dtype=jnp.dtype(jnp.float32),
            name="previous_net_reward",
        )
        _require_array(
            state.history_available,
            shape=(),
            dtype=jnp.dtype(jnp.bool_),
            name="history_available",
        )
        _require_array(state.step_words, shape=(2,), dtype=jnp.dtype(jnp.uint32), name="step_words")
        _require_array(
            state.content_tag_words,
            shape=(4,),
            dtype=jnp.dtype(jnp.uint32),
            name="content_tag_words",
        )

    def _state_values_valid_without_tag(self, state: HCCLCausalCoreState) -> Array:
        step_zero = jnp.all(state.step_words == 0)
        signs = (state.previous_action_signs == -1.0) | (state.previous_action_signs == 1.0)
        genesis_history = (
            (~state.history_available)
            & jnp.all(state.previous_action_signs == 0.0)
            & (state.previous_task_score == 0.0)
            & jnp.all(state.previous_net_reward == 0.0)
        )
        completed_history = state.history_available & jnp.all(signs)
        return (
            _words_at_most_lifetime(
                state.step_words, self._config.maximum_committed_transitions
            )
            & jnp.all(jnp.isfinite(state.positions))
            & jnp.all(jnp.abs(state.positions) <= jnp.float32(_WORLD_LIMIT))
            & jnp.all(jnp.isfinite(state.velocities))
            & jnp.all(jnp.abs(state.velocities) <= jnp.float32(_MAX_SPEED))
            & jnp.all(jnp.isfinite(state.current_cues))
            & jnp.all((state.current_cues == -1.0) | (state.current_cues == 1.0))
            & jnp.all(jnp.isfinite(state.current_nuisance))
            & jnp.all(jnp.isfinite(state.current_partner_velocity_noise))
            & jnp.isfinite(state.hidden_sign)
            & ((state.hidden_sign == -1.0) | (state.hidden_sign == 1.0))
            & jnp.isfinite(state.previous_task_score)
            & jnp.all(jnp.isfinite(state.previous_net_reward))
            & jnp.where(step_zero, genesis_history, completed_history)
        )

    def state_valid(self, state: HCCLCausalCoreState) -> Array:
        self._require_state_contract(state)
        return self._state_values_valid_without_tag(state) & jnp.all(
            state.content_tag_words == self._state_tag(state)
        )

    def reseal_state(self, state: HCCLCausalCoreState) -> HCCLCausalCoreState:
        """Recompute the non-authenticating content tag for migration/tests.

        This does not grant execution or evidence authority.  It accepts only
        bounded, finite, history-consistent values and is deliberately eager.
        """

        self._require_state_contract(state)
        if not bool(self._state_values_valid_without_tag(state)):
            raise ValueError("state values cannot be resealed")
        return cast(Any, state).replace(content_tag_words=self._state_tag(state))

    def init(self, key: Array) -> HCCLCausalCoreState:
        """Initialize from caller key material without reserving a protocol seed."""

        if not _typed_threefry_key(key):
            raise TypeError("key must be a scalar typed Threefry key")
        sign_key = jr.fold_in(key, _INITIAL_SIGN_STREAM_TAG)
        world_key = jr.fold_in(key, _WORLD_STREAM_TAG)
        cue_root = jr.fold_in(key, _CUE_STREAM_TAG)
        outcome_key = jr.fold_in(key, _OUTCOME_STREAM_TAG)
        nuisance_root = jr.fold_in(key, _NUISANCE_STREAM_TAG)
        velocity_root = jr.fold_in(key, _PARTNER_VELOCITY_STREAM_TAG)
        cue_key, genesis_cue_key = jr.split(cue_root)
        nuisance_key, genesis_nuisance_key = jr.split(nuisance_root)
        partner_velocity_key, genesis_velocity_key = jr.split(velocity_root)
        hidden_sign = jnp.where(jr.bernoulli(sign_key, p=0.5), 1.0, -1.0).astype(jnp.float32)
        cue_flipped = jr.bernoulli(
            genesis_cue_key,
            p=jnp.asarray(_CUE_FLIP_PROBABILITIES, dtype=jnp.float32),
            shape=(2,),
        )
        cues = jnp.where(cue_flipped, -hidden_sign, hidden_sign).astype(jnp.float32)
        nuisance = jr.normal(genesis_nuisance_key, (2, 5), dtype=jnp.float32)
        velocity_noise = jnp.float32(_PARTNER_VELOCITY_NOISE_STD) * jr.normal(
            genesis_velocity_key, (2,), dtype=jnp.float32
        )
        bare = HCCLCausalCoreState(
            world_key=world_key,
            cue_key=cue_key,
            outcome_key=outcome_key,
            nuisance_key=nuisance_key,
            partner_velocity_key=partner_velocity_key,
            positions=jnp.asarray(_INITIAL_POSITIONS, dtype=jnp.float32),
            velocities=jnp.zeros((2,), dtype=jnp.float32),
            current_cues=cues,
            current_nuisance=nuisance,
            current_partner_velocity_noise=velocity_noise,
            hidden_sign=hidden_sign,
            previous_action_signs=jnp.zeros((2,), dtype=jnp.float32),
            previous_task_score=jnp.asarray(0.0, dtype=jnp.float32),
            previous_net_reward=jnp.zeros((2,), dtype=jnp.float32),
            history_available=jnp.asarray(False, dtype=jnp.bool_),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
            content_tag_words=jnp.zeros((4,), dtype=jnp.uint32),
        )
        return cast(Any, bare).replace(content_tag_words=self._state_tag(bare))

    def observe(self, state: HCCLCausalCoreState) -> Array:
        """Return two exact 16-channel learner observations with no oracle fields."""

        self._require_state_contract(state)
        other = jnp.asarray((1, 0), dtype=jnp.int32)
        own_position = state.positions / jnp.float32(_WORLD_LIMIT)
        relative_position = (state.positions[other] - state.positions) / jnp.float32(
            2.0 * _WORLD_LIMIT
        )
        own_velocity = state.velocities / jnp.float32(_MAX_SPEED)
        noisy_partner_velocity = (
            state.velocities[other] + state.current_partner_velocity_noise
        ) / jnp.float32(_MAX_SPEED)
        physical = jnp.stack(
            (own_position, relative_position, own_velocity, noisy_partner_velocity), axis=1
        )
        history = jnp.stack(
            (
                state.previous_action_signs,
                state.previous_action_signs[other],
                jnp.full((2,), state.previous_task_score, dtype=jnp.float32),
                state.previous_net_reward,
                jnp.full((2,), state.history_available.astype(jnp.float32), dtype=jnp.float32),
            ),
            axis=1,
        )
        cues = jnp.broadcast_to(state.current_cues, (2, 2))
        return jnp.concatenate((physical, history, cues, state.current_nuisance), axis=1)

    def evaluator_regime_name_for_step(self, step: int) -> str:
        maximum = self._config.maximum_committed_transitions
        if type(step) is not int or not 0 <= step < maximum:
            raise ValueError(f"step must lie in the evaluator schedule [0, {maximum})")
        for name, start, end in self.schedule:
            if start <= step < end:
                return name
        raise RuntimeError("schedule coverage is incomplete")

    def _regime_id(self, step_words: Array) -> Array:
        maximum = self._config.maximum_committed_transitions
        safe_step = jnp.minimum(
            step_words[1], jnp.asarray(maximum - 1, dtype=jnp.uint32)
        )
        schedule_ends = tuple(end for _name, _start, end in self.schedule)
        schedule_ids = tuple(
            HCCL_CAUSAL_CORE_REGIME_NAMES.index(name)
            for name, _start, _end in self.schedule
        )
        index = jnp.searchsorted(
            jnp.asarray(schedule_ends, dtype=jnp.uint32), safe_step, side="right"
        )
        return jnp.asarray(schedule_ids, dtype=jnp.int32)[index]

    def _require_event_contract(self, receipt: HCCLCausalCoreEventReceipt) -> None:
        if type(receipt) is not HCCLCausalCoreEventReceipt:
            raise TypeError("receipt must be exact HCCLCausalCoreEventReceipt")
        _require_array(
            receipt.source_state_tag_words,
            shape=(4,),
            dtype=jnp.dtype(jnp.uint32),
            name="receipt.source_state_tag_words",
        )
        _require_array(
            receipt.source_step_words,
            shape=(2,),
            dtype=jnp.dtype(jnp.uint32),
            name="receipt.source_step_words",
        )
        _require_array(
            receipt.world_flipped,
            shape=(),
            dtype=jnp.dtype(jnp.bool_),
            name="receipt.world_flipped",
        )
        _require_array(
            receipt.next_cue_flipped,
            shape=(2,),
            dtype=jnp.dtype(jnp.bool_),
            name="receipt.next_cue_flipped",
        )
        _require_array(
            receipt.outcome_flipped,
            shape=(),
            dtype=jnp.dtype(jnp.bool_),
            name="receipt.outcome_flipped",
        )
        _require_array(
            receipt.nuisance_standard_normal,
            shape=(2, 5),
            dtype=jnp.dtype(jnp.float32),
            name="receipt.nuisance_standard_normal",
        )
        _require_array(
            receipt.partner_velocity_standard_normal,
            shape=(2,),
            dtype=jnp.dtype(jnp.float32),
            name="receipt.partner_velocity_standard_normal",
        )
        for name in (
            "next_world_key",
            "next_cue_key",
            "next_outcome_key",
            "next_nuisance_key",
            "next_partner_velocity_key",
        ):
            if not _typed_threefry_key(getattr(receipt, name)):
                raise TypeError(f"receipt.{name} must be a typed Threefry key")
        _require_array(
            receipt.draw_counts,
            shape=(5,),
            dtype=jnp.dtype(jnp.int32),
            name="receipt.draw_counts",
        )
        _require_array(
            receipt.content_tag_words,
            shape=(4,),
            dtype=jnp.dtype(jnp.uint32),
            name="receipt.content_tag_words",
        )

    def prepare_event(self, state: HCCLCausalCoreState) -> HCCLCausalCoreEventReceipt:
        """Materialize all action-independent draws without mutating the source."""

        self._require_state_contract(state)
        next_world_key, world_sample_key = jr.split(state.world_key)
        next_cue_key, cue_sample_key = jr.split(state.cue_key)
        next_outcome_key, outcome_sample_key = jr.split(state.outcome_key)
        next_nuisance_key, nuisance_sample_key = jr.split(state.nuisance_key)
        next_partner_velocity_key, velocity_sample_key = jr.split(state.partner_velocity_key)
        bare = HCCLCausalCoreEventReceipt(
            source_state_tag_words=state.content_tag_words,
            source_step_words=state.step_words,
            world_flipped=jr.bernoulli(world_sample_key, p=jnp.float32(_WORLD_FLIP_PROBABILITY)),
            next_cue_flipped=jr.bernoulli(
                cue_sample_key,
                p=jnp.asarray(_CUE_FLIP_PROBABILITIES, dtype=jnp.float32),
                shape=(2,),
            ),
            outcome_flipped=jr.bernoulli(
                outcome_sample_key, p=jnp.float32(_OUTCOME_FLIP_PROBABILITY)
            ),
            nuisance_standard_normal=jr.normal(nuisance_sample_key, (2, 5), dtype=jnp.float32),
            partner_velocity_standard_normal=jr.normal(
                velocity_sample_key, (2,), dtype=jnp.float32
            ),
            next_world_key=next_world_key,
            next_cue_key=next_cue_key,
            next_outcome_key=next_outcome_key,
            next_nuisance_key=next_nuisance_key,
            next_partner_velocity_key=next_partner_velocity_key,
            draw_counts=jnp.asarray(_EVENT_DRAW_VECTOR, dtype=jnp.int32),
            content_tag_words=jnp.zeros((4,), dtype=jnp.uint32),
        )
        return cast(Any, bare).replace(content_tag_words=_event_tag(bare))

    def event_receipt_valid(
        self, state: HCCLCausalCoreState, receipt: HCCLCausalCoreEventReceipt
    ) -> Array:
        self._require_state_contract(state)
        self._require_event_contract(receipt)
        expected = self.prepare_event(state)
        return (
            self.state_valid(state)
            & jnp.all(receipt.source_state_tag_words == state.content_tag_words)
            & jnp.all(receipt.source_step_words == state.step_words)
            & jnp.all(jnp.isfinite(receipt.nuisance_standard_normal))
            & jnp.all(jnp.isfinite(receipt.partner_velocity_standard_normal))
            & jnp.all(receipt.draw_counts == jnp.asarray(_EVENT_DRAW_VECTOR, jnp.int32))
            & jnp.all(receipt.content_tag_words == _event_tag(receipt))
            & _tree_exact_equal(receipt, expected)
        )

    def task_score_for_regime(
        self, regime_id: int | Array, factors: HCCLCausalCoreFactors
    ) -> Array:
        if type(factors) is not HCCLCausalCoreFactors:
            raise TypeError("factors must be exact HCCLCausalCoreFactors")
        regime = jnp.asarray(regime_id, dtype=jnp.int32)
        if regime.shape != ():
            raise ValueError("regime_id must be scalar")
        return jnp.where(
            regime == HCCL_REGIME_A,
            factors.gathering,
            jnp.where(
                regime == HCCL_REGIME_B,
                factors.velocity,
                jnp.where(
                    regime == HCCL_REGIME_C,
                    jnp.float32(0.5) * (factors.gathering + factors.velocity),
                    factors.convention_noisy,
                ),
            ),
        ).astype(jnp.float32)

    def _require_proposal_contract(self, proposal: HCCLCausalCoreProposal) -> None:
        if type(proposal) is not HCCLCausalCoreProposal:
            raise TypeError("proposal must be exact HCCLCausalCoreProposal")
        self._require_state_contract(proposal.candidate_state)
        for name, value, shape, dtype in (
            ("source_state_tag_words", proposal.source_state_tag_words, (4,), jnp.uint32),
            ("source_step_words", proposal.source_step_words, (2,), jnp.uint32),
            ("event_content_tag_words", proposal.event_content_tag_words, (4,), jnp.uint32),
            ("joint_action_ids", proposal.joint_action_ids, (2,), jnp.int32),
            ("action_signs", proposal.action_signs, (2,), jnp.float32),
            ("observation", proposal.observation, (2, 16), jnp.float32),
            ("next_observation", proposal.next_observation, (2, 16), jnp.float32),
            ("evaluator_regime_id", proposal.evaluator_regime_id, (), jnp.int32),
            ("current_hidden_sign", proposal.current_hidden_sign, (), jnp.float32),
            ("world_flipped", proposal.world_flipped, (), jnp.bool_),
            ("next_hidden_sign", proposal.next_hidden_sign, (), jnp.float32),
            ("next_cue_flipped", proposal.next_cue_flipped, (2,), jnp.bool_),
            ("outcome_flipped", proposal.outcome_flipped, (), jnp.bool_),
            ("valid", proposal.valid, (), jnp.bool_),
            ("content_tag_words", proposal.content_tag_words, (4,), jnp.uint32),
        ):
            _require_array(value, shape=shape, dtype=jnp.dtype(dtype), name=f"proposal.{name}")

    def propose(
        self,
        state: HCCLCausalCoreState,
        receipt: HCCLCausalCoreEventReceipt,
        joint_action_ids: Array,
    ) -> HCCLCausalCoreProposal:
        """Pure same-source transition proposal; no key or state is committed."""

        self._require_state_contract(state)
        self._require_event_contract(receipt)
        actions = jnp.asarray(joint_action_ids)
        if actions.shape != (2,):
            raise ValueError("joint_action_ids must have shape (2,)")
        if not jnp.issubdtype(actions.dtype, jnp.integer):
            raise TypeError("joint_action_ids must have integer dtype")
        action_ids = actions.astype(jnp.int32)
        actions_valid = jnp.all((action_ids == 0) | (action_ids == 1))
        safe_ids = jnp.where((action_ids == 0) | (action_ids == 1), action_ids, 0)
        action_signs = (2.0 * safe_ids.astype(jnp.float32) - 1.0).astype(jnp.float32)
        state_valid = self.state_valid(state)
        event_valid = self.event_receipt_valid(state, receipt)
        next_words, uint64_capacity = _increment_words(state.step_words)
        capacity = uint64_capacity & _words_below_lifetime(
            state.step_words, self._config.maximum_committed_transitions
        )

        accelerated = (
            jnp.float32(_DAMPING) * state.velocities + jnp.float32(_ACCELERATION) * action_signs
        )
        candidate_velocities = jnp.clip(
            accelerated, -jnp.float32(_MAX_SPEED), jnp.float32(_MAX_SPEED)
        )
        unclipped_positions = state.positions + jnp.float32(_TIME_DELTA) * candidate_velocities
        positions = jnp.clip(
            unclipped_positions, -jnp.float32(_WORLD_LIMIT), jnp.float32(_WORLD_LIMIT)
        )
        velocities = jnp.where(unclipped_positions != positions, 0.0, candidate_velocities).astype(
            jnp.float32
        )

        distance = jnp.abs(positions[0] - positions[1]) / jnp.float32(2.0 * _WORLD_LIMIT)
        target_position = jnp.float32(0.6 * _WORLD_LIMIT) * state.hidden_sign
        target_distance = jnp.abs(positions - target_position)
        local = jnp.float32(0.2) * (
            1.0 - target_distance / jnp.float32(1.6 * _WORLD_LIMIT)
        ) + jnp.float32(0.8) * (target_distance <= jnp.float32(0.1 * _WORLD_LIMIT)).astype(
            jnp.float32
        )
        gathering = (
            jnp.float32(0.5) * (1.0 - distance) + jnp.float32(0.25) * jnp.sum(local)
        ).astype(jnp.float32)
        desired_velocity = jnp.float32(0.8 * _MAX_SPEED) * state.hidden_sign
        velocity = jnp.clip(
            1.0 - jnp.sum(jnp.abs(velocities - desired_velocity)) / jnp.float32(3.6 * _MAX_SPEED),
            0.0,
            1.0,
        ).astype(jnp.float32)
        clean_p = (jnp.prod(action_signs) == state.hidden_sign).astype(jnp.float32)
        noisy_p = jnp.where(receipt.outcome_flipped, 1.0 - clean_p, clean_p).astype(jnp.float32)
        factors = HCCLCausalCoreFactors(
            gathering=gathering,
            velocity=velocity,
            convention_clean=clean_p,
            convention_noisy=noisy_p,
        )
        regime_id = self._regime_id(state.step_words)
        task_score = self.task_score_for_regime(regime_id, factors)
        zeros = jnp.zeros((2,), dtype=jnp.float32)
        signals = HCCLCausalCoreTypedSignals(
            task_score=task_score,
            net_reward=jnp.full((2,), task_score, dtype=jnp.float32),
            message_charge=zeros,
            safety_cost=zeros,
        )

        next_hidden_sign = jnp.where(
            receipt.world_flipped, -state.hidden_sign, state.hidden_sign
        ).astype(jnp.float32)
        next_cues = jnp.where(receipt.next_cue_flipped, -next_hidden_sign, next_hidden_sign).astype(
            jnp.float32
        )
        tv_scale = jnp.where(
            positions < jnp.float32(_TV_POSITION_THRESHOLD),
            jnp.float32(math.sqrt(_TV_VARIANCE_MULTIPLIER)),
            jnp.float32(1.0),
        )
        nuisance = receipt.nuisance_standard_normal.at[:, 0].multiply(tv_scale)
        partner_velocity_noise = (
            jnp.float32(_PARTNER_VELOCITY_NOISE_STD) * receipt.partner_velocity_standard_normal
        )
        candidate_bare = HCCLCausalCoreState(
            world_key=receipt.next_world_key,
            cue_key=receipt.next_cue_key,
            outcome_key=receipt.next_outcome_key,
            nuisance_key=receipt.next_nuisance_key,
            partner_velocity_key=receipt.next_partner_velocity_key,
            positions=positions,
            velocities=velocities,
            current_cues=next_cues,
            current_nuisance=nuisance,
            current_partner_velocity_noise=partner_velocity_noise,
            hidden_sign=next_hidden_sign,
            previous_action_signs=action_signs,
            previous_task_score=task_score,
            previous_net_reward=signals.net_reward,
            history_available=jnp.asarray(True, dtype=jnp.bool_),
            step_words=next_words,
            content_tag_words=jnp.zeros((4,), dtype=jnp.uint32),
        )
        candidate = cast(Any, candidate_bare).replace(
            content_tag_words=self._state_tag(candidate_bare)
        )
        candidate_valid = self.state_valid(candidate)
        valid = state_valid & event_valid & actions_valid & capacity & candidate_valid
        bare = HCCLCausalCoreProposal(
            source_state_tag_words=state.content_tag_words,
            source_step_words=state.step_words,
            event_content_tag_words=receipt.content_tag_words,
            joint_action_ids=action_ids,
            action_signs=action_signs,
            observation=self.observe(state),
            next_observation=self.observe(candidate),
            candidate_state=candidate,
            factors=factors,
            signals=signals,
            evaluator_regime_id=regime_id,
            current_hidden_sign=state.hidden_sign,
            world_flipped=receipt.world_flipped,
            next_hidden_sign=next_hidden_sign,
            next_cue_flipped=receipt.next_cue_flipped,
            outcome_flipped=receipt.outcome_flipped,
            valid=valid,
            content_tag_words=jnp.zeros((4,), dtype=jnp.uint32),
        )
        return cast(Any, bare).replace(content_tag_words=_proposal_tag(bare))

    def proposal_valid(
        self,
        state: HCCLCausalCoreState,
        receipt: HCCLCausalCoreEventReceipt,
        proposal: HCCLCausalCoreProposal,
    ) -> Array:
        self._require_proposal_contract(proposal)
        expected = self.propose(state, receipt, proposal.joint_action_ids)
        return (
            proposal.valid
            & jnp.all(proposal.source_state_tag_words == state.content_tag_words)
            & jnp.all(proposal.source_step_words == state.step_words)
            & jnp.all(proposal.event_content_tag_words == receipt.content_tag_words)
            & jnp.all(proposal.content_tag_words == _proposal_tag(proposal))
            & _tree_exact_equal(proposal, expected)
        )

    def commit(
        self,
        state: HCCLCausalCoreState,
        receipt: HCCLCausalCoreEventReceipt,
        proposal: HCCLCausalCoreProposal,
        *,
        downstream_candidate_valid: Array,
    ) -> HCCLCausalCoreStepResult:
        """Atomically commit the exact proposal or return the bit-exact source."""

        self._require_state_contract(state)
        self._require_event_contract(receipt)
        self._require_proposal_contract(proposal)
        downstream = _require_array(
            downstream_candidate_valid,
            shape=(),
            dtype=jnp.dtype(jnp.bool_),
            name="downstream_candidate_valid",
        )
        source_valid = self.state_valid(state)
        receipt_valid = self.event_receipt_valid(state, receipt)
        proposed_valid = self.proposal_valid(state, receipt, proposal)
        capacity = _words_below_lifetime(
            state.step_words, self._config.maximum_committed_transitions
        )
        applied = source_valid & receipt_valid & proposed_valid & downstream & capacity
        final_state = jax.lax.cond(
            applied, lambda _: proposal.candidate_state, lambda _: state, operand=None
        )
        return HCCLCausalCoreStepResult(
            state=final_state,
            proposal=proposal,
            pre_step_words=state.step_words,
            post_step_words=final_state.step_words,
            source_state_valid=source_valid,
            event_receipt_valid=receipt_valid,
            proposal_valid=proposed_valid,
            downstream_candidate_valid=downstream,
            lifetime_capacity_available=capacity,
            update_applied=applied,
            update_rejected=~applied,
        )

    def step(
        self,
        state: HCCLCausalCoreState,
        receipt: HCCLCausalCoreEventReceipt,
        joint_action_ids: Array,
        *,
        downstream_candidate_valid: Array,
    ) -> HCCLCausalCoreStepResult:
        proposal = self.propose(state, receipt, joint_action_ids)
        return self.commit(
            state,
            receipt,
            proposal,
            downstream_candidate_valid=downstream_candidate_valid,
        )

    def resource_budget(
        self, state: HCCLCausalCoreState | None = None
    ) -> HCCLCausalCoreResourceBudget:
        reference = self.init(jr.key(0)) if state is None else state
        self._require_state_contract(reference)
        receipt = self.prepare_event(reference)
        proposal = self.propose(reference, receipt, jnp.zeros((2,), dtype=jnp.int32))
        return HCCLCausalCoreResourceBudget(
            schema=HCCL_CAUSAL_CORE_RESOURCE_SCHEMA,
            persistent_state_nbytes=measure_hccl_causal_core_state_nbytes(reference),
            event_receipt_nbytes=_tree_nbytes(receipt),
            proposal_nbytes=_tree_nbytes(proposal),
            observation_float32_scalars_per_state=2 * _OBSERVATION_DIM,
            initialization_draws=sum(HCCL_CAUSAL_CORE_INITIAL_DRAW_COUNTS.values()),
            event_draws_per_receipt=sum(HCCL_CAUSAL_CORE_EVENT_DRAW_COUNTS.values()),
            world_draws_per_receipt=1,
            cue_draws_per_receipt=2,
            outcome_draws_per_receipt=1,
            nuisance_draws_per_receipt=10,
            partner_velocity_draws_per_receipt=2,
            maximum_committed_transitions=self._config.maximum_committed_transitions,
            output_write_calls=0,
            artifact_bytes_written=0,
        )


def _tree_nbytes(tree: Any) -> int:
    total = 0
    for leaf in jax.tree.leaves(tree):
        if not hasattr(leaf, "dtype"):
            continue
        array = cast(Array, leaf)
        if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
            data = jr.key_data(array)
            total += int(data.size) * int(data.dtype.itemsize)
        else:
            total += int(array.size) * int(array.dtype.itemsize)
    return total


def measure_hccl_causal_core_state_nbytes(state: HCCLCausalCoreState) -> int:
    """Measure every logical persistent array byte, including named key words."""

    if type(state) is not HCCLCausalCoreState:
        raise TypeError("state must be exact HCCLCausalCoreState")
    return _tree_nbytes(state)


def run_hccl_causal_core_scan(
    world: HCCLCausalCoreWorld,
    state: HCCLCausalCoreState,
    joint_action_ids: Array,
    downstream_candidate_valid: Array,
) -> HCCLCausalCoreScanResult:
    """Scan actions while preparing one immutable receipt from each carry state."""

    if type(world) is not HCCLCausalCoreWorld:
        raise TypeError("world must be exact HCCLCausalCoreWorld")
    world._require_state_contract(state)
    if joint_action_ids.ndim != 2 or joint_action_ids.shape[1] != 2:
        raise ValueError("joint_action_ids must have shape (steps, 2)")
    if not jnp.issubdtype(joint_action_ids.dtype, jnp.integer):
        raise TypeError("joint_action_ids must have integer dtype")
    _require_array(
        downstream_candidate_valid,
        shape=(joint_action_ids.shape[0],),
        dtype=jnp.dtype(jnp.bool_),
        name="downstream_candidate_valid",
    )

    def body(
        carry: HCCLCausalCoreState, row: tuple[Array, Array]
    ) -> tuple[HCCLCausalCoreState, tuple[Array, ...]]:
        actions, gate = row
        receipt = world.prepare_event(carry)
        result = world.step(
            carry,
            receipt,
            actions,
            downstream_candidate_valid=gate,
        )
        proposal = result.proposal
        return result.state, (
            proposal.observation,
            proposal.next_observation,
            proposal.signals.task_score,
            proposal.evaluator_regime_id,
            receipt.content_tag_words,
            result.post_step_words,
            result.update_applied,
        )

    final_state, outputs = jax.lax.scan(body, state, (joint_action_ids, downstream_candidate_valid))
    observations, next_observations, scores, regimes, tags, words, applied = outputs
    return HCCLCausalCoreScanResult(
        state=final_state,
        observations=observations,
        next_observations=next_observations,
        task_scores=scores,
        regime_ids=regimes,
        event_content_tag_words=tags,
        post_step_words=words,
        update_applied=applied,
    )


def _state_host_payload(state: HCCLCausalCoreState) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for leaf in jax.tree.leaves(state):
        if not hasattr(leaf, "dtype"):
            continue
        array = cast(Array, leaf)
        if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
            host = np.asarray(jr.key_data(array), dtype=np.uint32)
            dtype = "typed-threefry-key-uint32"
        else:
            host = np.asarray(array)
            dtype = str(host.dtype)
        payload.append(
            {
                "shape": list(host.shape),
                "dtype": dtype,
                "bytes_hex": np.ascontiguousarray(host).tobytes().hex(),
            }
        )
    return payload


def _checkpoint_digest(checkpoint: HCCLCausalCoreCheckpoint) -> str:
    return _digest(
        {
            "schema": checkpoint.schema,
            "mechanism_status": checkpoint.mechanism_status,
            "evidence_level": checkpoint.evidence_level,
            "output_writes_authorized": checkpoint.output_writes_authorized,
            "artifact_authorized": checkpoint.artifact_authorized,
            "evidence_authorized": checkpoint.evidence_authorized,
            "config": checkpoint.config,
            "config_sha256": checkpoint.config_sha256,
            "state_nbytes": checkpoint.state_nbytes,
            "state_sha256": checkpoint.state_sha256,
        }
    )


def save_hccl_causal_core_checkpoint(
    world: HCCLCausalCoreWorld,
    state: HCCLCausalCoreState,
) -> HCCLCausalCoreCheckpoint:
    """Return an in-memory checkpoint receipt; perform zero output writes."""

    if type(world) is not HCCLCausalCoreWorld:
        raise TypeError("world must be exact HCCLCausalCoreWorld")
    world._require_state_contract(state)
    if not bool(world.state_valid(state)):
        raise ValueError("cannot checkpoint invalid HCCL causal-core state")
    config = world.to_config()
    copied_state = cast(HCCLCausalCoreState, jax.tree.map(jnp.array, state))
    bare = HCCLCausalCoreCheckpoint(
        schema=HCCL_CAUSAL_CORE_CHECKPOINT_SCHEMA,
        mechanism_status=HCCL_CAUSAL_CORE_STATUS,
        evidence_level=HCCL_CAUSAL_CORE_EVIDENCE_LEVEL,
        output_writes_authorized=False,
        artifact_authorized=False,
        evidence_authorized=False,
        config=config,
        config_sha256=_digest(config),
        state=copied_state,
        state_nbytes=measure_hccl_causal_core_state_nbytes(copied_state),
        state_sha256=_digest(_state_host_payload(copied_state)),
        checkpoint_sha256="",
    )
    return dataclasses.replace(bare, checkpoint_sha256=_checkpoint_digest(bare))


def load_hccl_causal_core_checkpoint(
    checkpoint: HCCLCausalCoreCheckpoint,
) -> tuple[HCCLCausalCoreWorld, HCCLCausalCoreState]:
    """Restore only a canonical in-memory world-state receipt."""

    if type(checkpoint) is not HCCLCausalCoreCheckpoint:
        raise TypeError("checkpoint must be exact HCCLCausalCoreCheckpoint")
    fixed = {
        "schema": HCCL_CAUSAL_CORE_CHECKPOINT_SCHEMA,
        "mechanism_status": HCCL_CAUSAL_CORE_STATUS,
        "evidence_level": HCCL_CAUSAL_CORE_EVIDENCE_LEVEL,
        "output_writes_authorized": False,
        "artifact_authorized": False,
        "evidence_authorized": False,
    }
    for name, expected in fixed.items():
        actual = getattr(checkpoint, name)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"checkpoint {name} differs")
    if type(checkpoint.config) is not dict:
        raise TypeError("checkpoint config must be an exact dict")
    _require_digest(checkpoint.config_sha256, name="checkpoint config sha256")
    _require_digest(checkpoint.state_sha256, name="checkpoint state sha256")
    _require_digest(checkpoint.checkpoint_sha256, name="checkpoint sha256")
    world = HCCLCausalCoreWorld.from_config(checkpoint.config)
    if checkpoint.config_sha256 != _digest(checkpoint.config):
        raise ValueError("checkpoint config digest differs")
    world._require_state_contract(checkpoint.state)
    if type(checkpoint.state_nbytes) is not int:
        raise TypeError("checkpoint state_nbytes must be an exact int")
    if checkpoint.state_nbytes != measure_hccl_causal_core_state_nbytes(checkpoint.state):
        raise ValueError("checkpoint state bytes differ")
    if checkpoint.state_sha256 != _digest(_state_host_payload(checkpoint.state)):
        raise ValueError("checkpoint state digest differs")
    if checkpoint.checkpoint_sha256 != _checkpoint_digest(checkpoint):
        raise ValueError("checkpoint digest differs")
    if not bool(world.state_valid(checkpoint.state)):
        raise ValueError("checkpoint state is invalid")
    restored = cast(HCCLCausalCoreState, jax.tree.map(jnp.array, checkpoint.state))
    return world, restored


__all__ = [
    "HCCL_CAUSAL_CORE_CANONICAL_PROFILE",
    "HCCL_CAUSAL_CORE_CHECKPOINT_SCHEMA",
    "HCCL_CAUSAL_CORE_CONFIG_SCHEMA",
    "HCCL_CAUSAL_CORE_EVENT_DRAW_COUNTS",
    "HCCL_CAUSAL_CORE_EVENT_SCHEMA",
    "HCCL_CAUSAL_CORE_EVIDENCE_LEVEL",
    "HCCL_CAUSAL_CORE_INITIAL_DRAW_COUNTS",
    "HCCL_CAUSAL_CORE_LEARNER_OBSERVATION_FIELDS",
    "HCCL_CAUSAL_CORE_L2_CONFIG_SCHEMA",
    "HCCL_CAUSAL_CORE_L2_PROFILE",
    "HCCL_CAUSAL_CORE_L2_SCHEDULE",
    "HCCL_CAUSAL_CORE_L2_STATE_SCHEMA",
    "HCCL_CAUSAL_CORE_L3_CONFIG_SCHEMA",
    "HCCL_CAUSAL_CORE_L3_PROFILE",
    "HCCL_CAUSAL_CORE_L3_SCHEDULE",
    "HCCL_CAUSAL_CORE_L3_STATE_SCHEMA",
    "HCCL_CAUSAL_CORE_LIMITATIONS",
    "HCCL_CAUSAL_CORE_PROPOSAL_SCHEMA",
    "HCCL_CAUSAL_CORE_REGIME_NAMES",
    "HCCL_CAUSAL_CORE_RESOURCE_SCHEMA",
    "HCCL_CAUSAL_CORE_SCHEDULE",
    "HCCL_CAUSAL_CORE_SMOKE_CONFIG_SCHEMA",
    "HCCL_CAUSAL_CORE_SMOKE_ENTRY_WINDOW_STEPS",
    "HCCL_CAUSAL_CORE_SMOKE_PROFILE",
    "HCCL_CAUSAL_CORE_SMOKE_SCHEDULE",
    "HCCL_CAUSAL_CORE_SMOKE_STATE_SCHEMA",
    "HCCL_CAUSAL_CORE_SMOKE_TAIL_WINDOW_STEPS",
    "HCCL_CAUSAL_CORE_STATE_SCHEMA",
    "HCCL_CAUSAL_CORE_STATUS",
    "HCCL_REGIME_A",
    "HCCL_REGIME_B",
    "HCCL_REGIME_C",
    "HCCL_REGIME_D",
    "HCCLCausalCoreCheckpoint",
    "HCCLCausalCoreConfig",
    "HCCLCausalCoreEventReceipt",
    "HCCLCausalCoreFactors",
    "HCCLCausalCoreProposal",
    "HCCLCausalCoreResourceBudget",
    "HCCLCausalCoreScanResult",
    "HCCLCausalCoreState",
    "HCCLCausalCoreStepResult",
    "HCCLCausalCoreTypedSignals",
    "HCCLCausalCoreWorld",
    "load_hccl_causal_core_checkpoint",
    "hccl_causal_core_cycle_count_for_profile",
    "hccl_causal_core_lifetime_for_profile",
    "hccl_causal_core_schedule_for_profile",
    "measure_hccl_causal_core_state_nbytes",
    "run_hccl_causal_core_scan",
    "save_hccl_causal_core_checkpoint",
]
