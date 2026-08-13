"""Hidden-partner life with genuine recurrent world-state feedback.

This stream preserves the eight-channel, simultaneous-action A/B/C/D partner
problem while adding an evaluator-hidden Markov world sign ``z``.  Two noisy
ordinary cues report the current sign, and the contextual joint outcome is

``y[t + 1] = z[t] * focal_sign[t] * partner_sign[t] * outcome_noise[t]``.

Consequently reward remains exactly ``(1 + y) / 2``, as required by the
integrated hidden-partner transition contract, but is no longer determined by
the joint action alone.  After scoring the current action, the world sign may
flip and the next cues are sampled.  The partner, world, cue, ordinary-signal,
and schedule RNG streams are independent and action-independent.

The previous contextual outcome and previous partner action, together with
the learner's remembered previous focal action, provide a noisy measurement
of the preceding world sign.  Current noisy cues update the propagated Markov
posterior.  No single transition reveals ``z`` exactly, so optimal inference
uses a genuine multi-step filter without exposing a task id, boundary, or
oracle channel.

Schedule time is carried by two exact uint32 words.  The legacy int32 count is
saturating telemetry only; it never selects a cycle, segment, or partner
mapping.  Malformed transactions roll back every state and RNG leaf.  Exact
lifetime exhaustion returns the sole finite terminal no-op.
"""

# mypy: disable-error-code="call-arg"

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping
from numbers import Real
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Bool, Float, Int, PRNGKeyArray, UInt

from alberta_framework.streams.hidden_partner_mapping import (
    DEFAULT_BASE_SEGMENT_LENGTHS,
    DEFAULT_JITTER_RADIUS,
    DEFAULT_PARTNER_FLIP_PROBABILITY,
    DEFAULT_REGIME_SCHEDULE,
    NEGATIVE_ACTION,
    POSITIVE_ACTION,
    REGIME_A,
    REGIME_B,
    REGIME_C,
    REGIME_D,
)

HIDDEN_PARTNER_WORLD_FEEDBACK_CONTRACT_VERSION = "hidden-partner-world-feedback-v1"
HIDDEN_PARTNER_WORLD_FEEDBACK_CONFIG_SCHEMA = "alberta.hidden-partner-world-feedback.config.v2"
HIDDEN_PARTNER_WORLD_FEEDBACK_STATE_SCHEMA = "alberta.hidden-partner-world-feedback.state.v2"
HIDDEN_PARTNER_WORLD_FEEDBACK_EXACT_IDENTITY_NBYTES = 8
# All default flip rates lie strictly inside (0, 0.5): every noisy channel is
# individually informative about the hidden sign ``z``, yet none reveals it.
# The world flip rate 0.03 gives ``z`` an expected persistence of ~33 steps,
# slow relative to the per-step cue noise, so integrating evidence across
# steps improves on reading the current cues alone.  The two cues carry
# different reliabilities (0.25 vs 0.35 flip rate), so a correct filter must
# weight them unequally rather than averaging them.
DEFAULT_WORLD_FLIP_PROBABILITY = 0.03
DEFAULT_CUE_FLIP_PROBABILITIES = (0.25, 0.35)
DEFAULT_OUTCOME_FLIP_PROBABILITY = 0.15

REGIME_NAMES = ("A", "B", "C", "D")
N_ACTIONS = 2

X_INDEX = 0
PREVIOUS_OUTCOME_INDEX = 1
PREVIOUS_PARTNER_ACTION_INDEX = 2
HAS_PARTNER_HISTORY_INDEX = 3
U_INDEX = 4
V_INDEX = 5
CUE_1_INDEX = 6
CUE_2_INDEX = 7
OBSERVATION_FIELDS = (
    "x",
    "previous_contextual_outcome",
    "previous_partner_action",
    "has_partner_history",
    "u",
    "v",
    "world_cue_1",
    "world_cue_2",
)
OBSERVATION_DIM = len(OBSERVATION_FIELDS)

_N_SEGMENTS = len(DEFAULT_REGIME_SCHEDULE)
_N_CURRENT_SIGNALS = 3  # x, u, v
_N_CUES = 2
_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1

_SCHEDULE_RNG_TAG = 0
_SIGNAL_RNG_TAG = 1
_PARTNER_RNG_TAG = 2
_WORLD_RNG_TAG = 3
_CUE_RNG_TAG = 4
_OUTCOME_RNG_TAG = 5


def _probability(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{name} must be finite and lie in [0, 1]")
    return float(value)


@dataclasses.dataclass(frozen=True)
class HiddenPartnerWorldFeedbackConfig:
    """Strict static construction for the world-feedback v1 stream."""

    contract_version: str = HIDDEN_PARTNER_WORLD_FEEDBACK_CONTRACT_VERSION
    regime_schedule: tuple[int, ...] = DEFAULT_REGIME_SCHEDULE
    base_segment_lengths: tuple[int, ...] = DEFAULT_BASE_SEGMENT_LENGTHS
    jitter_radius: int = DEFAULT_JITTER_RADIUS
    partner_flip_probability: float = DEFAULT_PARTNER_FLIP_PROBABILITY
    world_flip_probability: float = DEFAULT_WORLD_FLIP_PROBABILITY
    cue_flip_probabilities: tuple[float, float] = DEFAULT_CUE_FLIP_PROBABILITIES
    outcome_flip_probability: float = DEFAULT_OUTCOME_FLIP_PROBABILITY

    def __post_init__(self) -> None:
        if self.contract_version != HIDDEN_PARTNER_WORLD_FEEDBACK_CONTRACT_VERSION:
            raise ValueError(
                f"contract_version must be {HIDDEN_PARTNER_WORLD_FEEDBACK_CONTRACT_VERSION!r}"
            )
        if not isinstance(self.regime_schedule, tuple):
            raise ValueError("regime_schedule must be a tuple")
        if any(type(regime) is not int for regime in self.regime_schedule):
            raise ValueError("regime_schedule must contain non-boolean integers")
        if self.regime_schedule != DEFAULT_REGIME_SCHEDULE:
            raise ValueError("regime_schedule must be the exact A->B->A->D->A->C->A->B->C schedule")
        if not isinstance(self.base_segment_lengths, tuple):
            raise ValueError("base_segment_lengths must be a tuple")
        if len(self.base_segment_lengths) != _N_SEGMENTS:
            raise ValueError(f"base_segment_lengths must contain exactly {_N_SEGMENTS} values")
        if any(type(length) is not int for length in self.base_segment_lengths):
            raise ValueError("base_segment_lengths must contain non-boolean integers")
        if type(self.jitter_radius) is not int or self.jitter_radius < 0:
            raise ValueError("jitter_radius must be a non-negative integer")
        if any(length <= self.jitter_radius for length in self.base_segment_lengths):
            raise ValueError(
                "every base segment length must exceed jitter_radius so all lengths are positive"
            )
        maximum_cycle_length = sum(self.base_segment_lengths) + (_N_SEGMENTS * self.jitter_radius)
        if maximum_cycle_length > _INT32_MAX:
            raise ValueError("maximum jittered cycle length must fit in int32")
        partner_probability = _probability(
            self.partner_flip_probability,
            name="partner_flip_probability",
        )
        world_probability = _probability(
            self.world_flip_probability,
            name="world_flip_probability",
        )
        if not isinstance(self.cue_flip_probabilities, tuple):
            raise ValueError("cue_flip_probabilities must be a tuple")
        if len(self.cue_flip_probabilities) != _N_CUES:
            raise ValueError(f"cue_flip_probabilities must contain exactly {_N_CUES} values")
        cue_probabilities = tuple(
            _probability(probability, name=f"cue_flip_probabilities[{index}]")
            for index, probability in enumerate(self.cue_flip_probabilities)
        )
        outcome_probability = _probability(
            self.outcome_flip_probability,
            name="outcome_flip_probability",
        )
        object.__setattr__(self, "partner_flip_probability", partner_probability)
        object.__setattr__(self, "world_flip_probability", world_probability)
        object.__setattr__(self, "cue_flip_probabilities", cue_probabilities)
        object.__setattr__(self, "outcome_flip_probability", outcome_probability)

    def to_config(self) -> dict[str, Any]:
        return {
            "schema": HIDDEN_PARTNER_WORLD_FEEDBACK_CONFIG_SCHEMA,
            "state_schema": HIDDEN_PARTNER_WORLD_FEEDBACK_STATE_SCHEMA,
            "type": type(self).__name__,
            "contract_version": self.contract_version,
            "regime_schedule": list(self.regime_schedule),
            "base_segment_lengths": list(self.base_segment_lengths),
            "jitter_radius": self.jitter_radius,
            "partner_flip_probability": float(self.partner_flip_probability),
            "world_flip_probability": float(self.world_flip_probability),
            "cue_flip_probabilities": [
                float(probability) for probability in self.cue_flip_probabilities
            ],
            "outcome_flip_probability": float(self.outcome_flip_probability),
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_config(),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
    ) -> HiddenPartnerWorldFeedbackConfig:
        payload = dict(config)
        expected = {
            "schema",
            "state_schema",
            "type",
            "contract_version",
            "regime_schedule",
            "base_segment_lengths",
            "jitter_radius",
            "partner_flip_probability",
            "world_flip_probability",
            "cue_flip_probabilities",
            "outcome_flip_probability",
        }
        if set(payload) != expected:
            if "schema" not in payload:
                raise ValueError(
                    "legacy hidden-partner world-feedback config requires explicit migration"
                )
            raise ValueError("config fields do not match the serialized schema")
        schema = payload.pop("schema")
        if schema != HIDDEN_PARTNER_WORLD_FEEDBACK_CONFIG_SCHEMA:
            raise ValueError(f"unexpected config schema: {schema!r}")
        state_schema = payload.pop("state_schema")
        if state_schema != HIDDEN_PARTNER_WORLD_FEEDBACK_STATE_SCHEMA:
            raise ValueError(f"unexpected state schema: {state_schema!r}")
        if payload.pop("type") != cls.__name__:
            raise ValueError("unexpected config type")
        schedule = payload.pop("regime_schedule")
        lengths = payload.pop("base_segment_lengths")
        cue_probabilities = payload.pop("cue_flip_probabilities")
        if (
            not isinstance(schedule, (list, tuple))
            or not isinstance(lengths, (list, tuple))
            or not isinstance(cue_probabilities, (list, tuple))
        ):
            raise ValueError("schedule and segment lengths must serialize as sequences")
        return cls(
            regime_schedule=tuple(schedule),
            base_segment_lengths=tuple(lengths),
            cue_flip_probabilities=tuple(cue_probabilities),
            **payload,
        )


def migrate_legacy_hidden_partner_world_feedback_config(
    legacy_config: Mapping[str, Any],
) -> HiddenPartnerWorldFeedbackConfig:
    """Explicitly stamp one exact schema-less pre-v2 configuration."""

    if not isinstance(legacy_config, Mapping):
        raise TypeError("legacy hidden-partner world-feedback config must be a mapping")
    payload = dict(legacy_config)
    expected = {
        "type",
        "contract_version",
        "regime_schedule",
        "base_segment_lengths",
        "jitter_radius",
        "partner_flip_probability",
        "world_flip_probability",
        "cue_flip_probabilities",
        "outcome_flip_probability",
    }
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        extra = sorted(set(payload) - expected)
        raise ValueError(
            "legacy hidden-partner world-feedback config fields are not exact; "
            f"missing={missing}, extra={extra}"
        )
    payload["schema"] = HIDDEN_PARTNER_WORLD_FEEDBACK_CONFIG_SCHEMA
    payload["state_schema"] = HIDDEN_PARTNER_WORLD_FEEDBACK_STATE_SCHEMA
    return HiddenPartnerWorldFeedbackConfig.from_config(payload)


@dataclasses.dataclass(frozen=True)
class HiddenPartnerWorldFeedbackResourceBudget:
    """Exact logical persistent-state accounting.

    ``state_nbytes`` counts only leaves of
    :class:`HiddenPartnerWorldFeedbackState`: float32/int32 scalars, one
    boolean, two exact lifetime words, and five JAX PRNG keys counted as ten
    logical uint32 scalars.  It excludes Python/config objects, transition
    outputs, compiler buffers, and device alignment.
    """

    state_schema: str
    observation_float32_scalars: int
    persistent_float32_scalars: int
    persistent_int32_scalars: int
    persistent_bool_scalars: int
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
class HiddenPartnerWorldFeedbackState:
    """Fixed-shape causal state for one uninterrupted environment life."""

    signal_key: PRNGKeyArray
    partner_key: PRNGKeyArray
    world_key: PRNGKeyArray
    cue_key: PRNGKeyArray
    outcome_key: PRNGKeyArray
    segment_lengths: Int[Array, " 9"]
    segment_ends: Int[Array, " 9"]
    current_signals: Float[Array, " 3"]
    current_cues: Float[Array, " 2"]
    world_sign: Float[Array, ""]
    previous_outcome: Float[Array, ""]
    previous_partner_action: Int[Array, ""]
    has_partner_history: Bool[Array, ""]
    step_count: Int[Array, ""]
    step_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class HiddenPartnerWorldFeedbackOracle:
    """Evaluator-only hidden world/schedule diagnostics for one step.

    Includes the hidden world sign before and after the step, per-channel
    noise-flip indicators, and the full-information optimal focal action
    (the action a policy knowing ``z`` and the partner's intent would take,
    with its expected-reward margin and tie flag).
    """

    step_count: Int[Array, ""]
    step_words: UInt[Array, " 2"]
    cycle_index: Int[Array, ""]
    cycle_words: UInt[Array, " 2"]
    cycle_step: Int[Array, ""]
    cycle_length: Int[Array, ""]
    segment_index: Int[Array, ""]
    segment_step: Int[Array, ""]
    segment_length: Int[Array, ""]
    regime_id: Int[Array, ""]
    next_step_words: UInt[Array, " 2"]
    next_cycle_index: Int[Array, ""]
    next_cycle_words: UInt[Array, " 2"]
    next_segment_index: Int[Array, ""]
    next_regime_id: Int[Array, ""]
    schedule_switched: Bool[Array, ""]
    partner_intended_action: Int[Array, ""]
    partner_intended_sign: Float[Array, ""]
    partner_flipped: Bool[Array, ""]
    focal_action_sign: Float[Array, ""]
    partner_action_sign: Float[Array, ""]
    world_sign: Float[Array, ""]
    world_cue_flipped: Bool[Array, " 2"]
    world_flipped: Bool[Array, ""]
    next_world_sign: Float[Array, ""]
    next_world_cue_flipped: Bool[Array, " 2"]
    outcome_flipped: Bool[Array, ""]
    noiseless_contextual_outcome: Float[Array, ""]
    contextual_outcome: Float[Array, ""]
    counterfactual_rewards: Float[Array, " 2"]
    full_information_optimal_focal_action: Int[Array, ""]
    full_information_action_margin: Float[Array, ""]
    full_information_action_tied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class HiddenPartnerWorldFeedbackTransition:
    """One continuing simultaneous-action transition."""

    observation: Float[Array, " 8"]
    focal_action: Int[Array, ""]
    partner_action: Int[Array, ""]
    reward: Float[Array, ""]
    outcome: Float[Array, ""]
    next_observation: Float[Array, " 8"]
    terminated: Bool[Array, ""]
    discount: Float[Array, ""]
    oracle: HiddenPartnerWorldFeedbackOracle


@chex.dataclass(frozen=True)
class HiddenPartnerWorldFeedbackStepResult:
    """One fail-closed world transaction and its exact lifetime audit."""

    transition: HiddenPartnerWorldFeedbackTransition
    state: HiddenPartnerWorldFeedbackState
    pre_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    lifetime_counter_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    state_valid: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    candidate_state_finite: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class _SchedulePosition:
    """Internal JAX record for a position in the repeating hidden schedule."""

    cycle_index: Int[Array, ""]
    cycle_words: UInt[Array, " 2"]
    cycle_step: Int[Array, ""]
    cycle_length: Int[Array, ""]
    segment_index: Int[Array, ""]
    segment_step: Int[Array, ""]
    segment_length: Int[Array, ""]
    regime_id: Int[Array, ""]


def _array_has_contract(value: Any, shape: tuple[int, ...], dtype: Any) -> bool:
    """Return a nonthrowing exact JAX-array shape/dtype predicate."""

    return isinstance(value, Array) and value.shape == shape and value.dtype == dtype


def _checked_words_increment(
    words: Array,
) -> tuple[UInt[Array, " 2"], Bool[Array, ""]]:
    """Propose one exact lifetime increment without uint64-word wrap."""

    array = jnp.asarray(words)
    if array.shape != (2,):
        raise ValueError("hidden-partner world-feedback step_words must have shape (2,)")
    if array.dtype != jnp.dtype(jnp.uint32):
        raise TypeError("hidden-partner world-feedback step_words must have dtype uint32")
    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    capacity_available = ~jnp.all(array == maximum)
    low = array[1] + jnp.asarray(1, dtype=jnp.uint32)
    carry = (low == jnp.asarray(0, dtype=jnp.uint32)).astype(jnp.uint32)
    proposed = jnp.stack((array[0] + carry, low)).astype(jnp.uint32)
    return jnp.where(capacity_available, proposed, array), capacity_available


def _words_to_saturating_int32(words: Array) -> Int[Array, ""]:
    """Project an exact word identity to non-negative int32 telemetry."""

    array = jnp.asarray(words)
    if array.shape != (2,):
        raise ValueError("hidden-partner world-feedback step_words must have shape (2,)")
    if array.dtype != jnp.dtype(jnp.uint32):
        raise TypeError("hidden-partner world-feedback step_words must have dtype uint32")
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
        raise ValueError("hidden-partner world-feedback step_count must be scalar")
    if count.dtype != jnp.dtype(jnp.int32):
        raise TypeError("hidden-partner world-feedback step_count must have dtype int32")
    return (count >= 0) & (count == _words_to_saturating_int32(words))


def _divmod_words_by_positive_int32(
    words: Array,
    divisor: Array,
) -> tuple[UInt[Array, " 2"], UInt[Array, ""]]:
    """Exactly divide uint64 words by a dynamic positive int32 divisor."""

    array = jnp.asarray(words)
    if array.shape != (2,):
        raise ValueError("hidden-partner world-feedback schedule words must have shape (2,)")
    if array.dtype != jnp.dtype(jnp.uint32):
        raise TypeError("hidden-partner world-feedback schedule words must have dtype uint32")
    divisor_i = jnp.asarray(divisor)
    if divisor_i.shape != ():
        raise ValueError("hidden-partner world-feedback cycle length must be scalar")
    if divisor_i.dtype != jnp.dtype(jnp.int32):
        raise TypeError("hidden-partner world-feedback cycle length must have dtype int32")
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


def _floating_tree_is_finite(value: Any) -> Bool[Array, ""]:
    """Require every floating or complex JAX leaf to be finite."""

    valid = jnp.asarray(True, dtype=jnp.bool_)
    for leaf in jax.tree.leaves(value):
        if isinstance(leaf, Array) and jnp.issubdtype(leaf.dtype, jnp.inexact):
            valid = valid & jnp.all(jnp.isfinite(leaf))
    return valid


def measure_hidden_partner_world_feedback_state_nbytes(
    state: HiddenPartnerWorldFeedbackState,
) -> int:
    """Measure every persistent JAX leaf in one world-feedback state."""

    validator = HiddenPartnerWorldFeedbackWorld()
    validator._require_state_contract(state)
    return sum(int(leaf.nbytes) for leaf in jax.tree.leaves(state) if isinstance(leaf, Array))


def _host_field_mapping(legacy_state: Any) -> dict[str, Any]:
    """Return a shallow host mapping for one legacy state record."""

    if isinstance(legacy_state, Mapping):
        return dict(legacy_state)
    if dataclasses.is_dataclass(legacy_state) and not isinstance(legacy_state, type):
        return {
            field.name: getattr(legacy_state, field.name)
            for field in dataclasses.fields(legacy_state)
        }
    raise TypeError("legacy hidden-partner world-feedback state must be a mapping or dataclass")


def migrate_legacy_hidden_partner_world_feedback_state(
    legacy_state: Any,
) -> HiddenPartnerWorldFeedbackState:
    """Migrate one exact pre-v2 state whose unsaturated int32 time is unique."""

    fields = _host_field_mapping(legacy_state)
    current_names = {
        field.name for field in dataclasses.fields(cast(Any, HiddenPartnerWorldFeedbackState))
    }
    legacy_names = current_names - {"step_words"}
    if set(fields) != legacy_names:
        missing = sorted(legacy_names - set(fields))
        extra = sorted(set(fields) - legacy_names)
        raise ValueError(
            "legacy hidden-partner world-feedback state fields are not exact; "
            f"missing={missing}, extra={extra}"
        )
    step_array = jnp.asarray(fields["step_count"])
    if step_array.shape != () or step_array.dtype != jnp.dtype(jnp.int32):
        raise TypeError("legacy hidden-partner world-feedback step_count must be scalar int32")
    step = int(step_array)
    if step < 0:
        raise ValueError("negative legacy hidden-partner world-feedback step_count indicates wrap")
    if step >= _INT32_MAX:
        raise ValueError("saturated legacy hidden-partner world-feedback step_count is ambiguous")
    fields["step_words"] = jnp.asarray((0, step), dtype=jnp.uint32)
    migrated = HiddenPartnerWorldFeedbackState(**fields)
    validator = HiddenPartnerWorldFeedbackWorld()
    validator._require_state_contract(migrated)
    if not bool(jax.device_get(validator._state_values_valid(migrated))):
        raise ValueError("legacy hidden-partner world-feedback state values are inconsistent")
    return migrated


class HiddenPartnerWorldFeedbackWorld:
    """Pure JAX world-feedback stream with a scripted stochastic partner."""

    def __init__(
        self,
        config: HiddenPartnerWorldFeedbackConfig | None = None,
    ) -> None:
        self._config = HiddenPartnerWorldFeedbackConfig() if config is None else config
        if not isinstance(self._config, HiddenPartnerWorldFeedbackConfig):
            raise TypeError("config must be a HiddenPartnerWorldFeedbackConfig")
        self._base_segment_lengths = jnp.asarray(
            self._config.base_segment_lengths,
            dtype=jnp.int32,
        )
        self._regime_schedule = jnp.asarray(
            self._config.regime_schedule,
            dtype=jnp.int32,
        )

    @property
    def config(self) -> HiddenPartnerWorldFeedbackConfig:
        return self._config

    @property
    def observation_dim(self) -> int:
        return OBSERVATION_DIM

    @property
    def feature_dim(self) -> int:
        return OBSERVATION_DIM

    @property
    def n_actions(self) -> int:
        return N_ACTIONS

    @property
    def n_segments(self) -> int:
        return _N_SEGMENTS

    @property
    def resource_budget(self) -> HiddenPartnerWorldFeedbackResourceBudget:
        float_scalars = _N_CURRENT_SIGNALS + _N_CUES + 2
        int_scalars = 2 * _N_SEGMENTS + 2
        bool_scalars = 1
        exact_identity_scalars = 2
        rng_scalars = 10
        state_scalars = (
            float_scalars + int_scalars + bool_scalars + exact_identity_scalars + rng_scalars
        )
        return HiddenPartnerWorldFeedbackResourceBudget(
            state_schema=HIDDEN_PARTNER_WORLD_FEEDBACK_STATE_SCHEMA,
            observation_float32_scalars=OBSERVATION_DIM,
            persistent_float32_scalars=float_scalars,
            persistent_int32_scalars=int_scalars,
            persistent_bool_scalars=bool_scalars,
            exact_identity_uint32_scalars=exact_identity_scalars,
            exact_identity_nbytes=HIDDEN_PARTNER_WORLD_FEEDBACK_EXACT_IDENTITY_NBYTES,
            lifetime_identity_bits=64,
            telemetry_saturation=_INT32_MAX,
            rng_uint32_scalars=rng_scalars,
            persistent_state_scalars=state_scalars,
            state_nbytes=(
                4 * (float_scalars + int_scalars + exact_identity_scalars + rng_scalars)
                + bool_scalars
            ),
            trainable_scalars=0,
            replay_capacity=0,
        )

    def to_config(self) -> dict[str, Any]:
        return {"type": type(self).__name__, "config": self._config.to_config()}

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
    ) -> HiddenPartnerWorldFeedbackWorld:
        payload = dict(config)
        if set(payload) != {"type", "config"}:
            raise ValueError("world config must contain exactly 'type' and 'config'")
        if payload["type"] != cls.__name__:
            raise ValueError("unexpected world type")
        inner = payload["config"]
        if not isinstance(inner, Mapping):
            raise ValueError("world config must contain a config mapping")
        return cls(HiddenPartnerWorldFeedbackConfig.from_config(inner))

    @staticmethod
    def _require_key_contract(key: Any, *, name: str) -> None:
        """Require one scalar typed or legacy JAX PRNG key."""

        if not isinstance(key, Array):
            raise TypeError(f"{name} must be a scalar JAX PRNG key")
        try:
            words = jnp.asarray(jr.key_data(key))
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} must be a scalar JAX PRNG key") from error
        if words.shape != (2,) or words.dtype != jnp.dtype(jnp.uint32):
            raise TypeError(f"{name} must be a scalar JAX PRNG key")

    def _require_state_contract(self, state: HiddenPartnerWorldFeedbackState) -> None:
        """Validate every fixed state shape and dtype before indexed work."""

        if not isinstance(state, HiddenPartnerWorldFeedbackState):
            raise TypeError("state must be a HiddenPartnerWorldFeedbackState")
        for name in (
            "signal_key",
            "partner_key",
            "world_key",
            "cue_key",
            "outcome_key",
        ):
            self._require_key_contract(getattr(state, name), name=name)
        contracts = (
            ("segment_lengths", state.segment_lengths, (_N_SEGMENTS,), jnp.int32),
            ("segment_ends", state.segment_ends, (_N_SEGMENTS,), jnp.int32),
            ("current_signals", state.current_signals, (_N_CURRENT_SIGNALS,), jnp.float32),
            ("current_cues", state.current_cues, (_N_CUES,), jnp.float32),
            ("world_sign", state.world_sign, (), jnp.float32),
            ("previous_outcome", state.previous_outcome, (), jnp.float32),
            ("previous_partner_action", state.previous_partner_action, (), jnp.int32),
            ("has_partner_history", state.has_partner_history, (), jnp.bool_),
            ("step_count", state.step_count, (), jnp.int32),
            ("step_words", state.step_words, (2,), jnp.uint32),
        )
        for name, value, shape, dtype in contracts:
            if not _array_has_contract(value, shape, dtype):
                raise TypeError(
                    f"hidden-partner world-feedback {name} must have shape {shape} "
                    f"and dtype {dtype}"
                )

    def _state_values_valid(
        self,
        state: HiddenPartnerWorldFeedbackState,
    ) -> Bool[Array, ""]:
        """Authenticate exact time, schedule arrays, signs, and causal history."""

        lengths_valid = jnp.all(state.segment_lengths > 0)
        expected_ends = jnp.cumsum(state.segment_lengths, dtype=jnp.int32)
        ends_valid = (
            jnp.array_equal(state.segment_ends, expected_ends)
            & jnp.all(state.segment_ends > 0)
            & jnp.all(state.segment_ends[1:] > state.segment_ends[:-1])
        )
        signals_valid = jnp.all(jnp.isfinite(state.current_signals)) & jnp.all(
            (state.current_signals == -1.0) | (state.current_signals == 1.0)
        )
        cues_valid = jnp.all(jnp.isfinite(state.current_cues)) & jnp.all(
            (state.current_cues == -1.0) | (state.current_cues == 1.0)
        )
        world_valid = jnp.isfinite(state.world_sign) & (
            (state.world_sign == -1.0) | (state.world_sign == 1.0)
        )
        at_birth = jnp.all(state.step_words == jnp.asarray((0, 0), dtype=jnp.uint32))
        outcome_is_sign = (state.previous_outcome == -1.0) | (state.previous_outcome == 1.0)
        partner_is_sign = (state.previous_partner_action == -1) | (
            state.previous_partner_action == 1
        )
        history_valid = jnp.where(
            at_birth,
            (~state.has_partner_history)
            & (state.previous_outcome == 0.0)
            & (state.previous_partner_action == 0),
            state.has_partner_history & outcome_is_sign & partner_is_sign,
        )
        return (
            lengths_valid
            & ends_valid
            & signals_valid
            & cues_valid
            & world_valid
            & jnp.isfinite(state.previous_outcome)
            & history_valid
            & _lifetime_counter_valid(state.step_words, state.step_count)
        )

    def init(self, key: Array) -> HiddenPartnerWorldFeedbackState:
        """Initialize a deterministic life with independent named RNG streams."""
        self._require_key_contract(key, name="key")
        schedule_key = jr.fold_in(key, _SCHEDULE_RNG_TAG)
        signal_root = jr.fold_in(key, _SIGNAL_RNG_TAG)
        partner_key = jr.fold_in(key, _PARTNER_RNG_TAG)
        world_root = jr.fold_in(key, _WORLD_RNG_TAG)
        cue_root = jr.fold_in(key, _CUE_RNG_TAG)
        outcome_key = jr.fold_in(key, _OUTCOME_RNG_TAG)

        initial_signal_key, signal_key = jr.split(signal_root)
        initial_world_key, world_key = jr.split(world_root)
        initial_cue_key, cue_key = jr.split(cue_root)
        world_sign = self._sample_sign(initial_world_key)
        jitters = jr.randint(
            schedule_key,
            shape=(_N_SEGMENTS,),
            minval=-self._config.jitter_radius,
            maxval=self._config.jitter_radius + 1,
            dtype=jnp.int32,
        )
        segment_lengths = self._base_segment_lengths + jitters
        return HiddenPartnerWorldFeedbackState(
            signal_key=signal_key,
            partner_key=partner_key,
            world_key=world_key,
            cue_key=cue_key,
            outcome_key=outcome_key,
            segment_lengths=segment_lengths,
            segment_ends=jnp.cumsum(segment_lengths, dtype=jnp.int32),
            current_signals=self._sample_signals(initial_signal_key),
            current_cues=self._sample_cues(initial_cue_key, world_sign),
            world_sign=world_sign,
            previous_outcome=jnp.asarray(0.0, dtype=jnp.float32),
            previous_partner_action=jnp.asarray(0, dtype=jnp.int32),
            has_partner_history=jnp.asarray(False, dtype=jnp.bool_),
            step_count=jnp.asarray(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def observe(self, state: HiddenPartnerWorldFeedbackState) -> Array:
        """Return exactly the eight ordinary, task-oracle-free channels."""
        self._require_state_contract(state)
        x, u, v = state.current_signals
        cue_1, cue_2 = state.current_cues
        observation = jnp.stack(
            (
                x,
                state.previous_outcome,
                state.previous_partner_action.astype(jnp.float32),
                state.has_partner_history.astype(jnp.float32),
                u,
                v,
                cue_1,
                cue_2,
            )
        ).astype(jnp.float32)
        return jnp.where(
            self._state_values_valid(state),
            observation,
            jnp.zeros_like(observation),
        )

    def step(
        self,
        state: HiddenPartnerWorldFeedbackState,
        focal_action: Array,
    ) -> tuple[HiddenPartnerWorldFeedbackTransition, HiddenPartnerWorldFeedbackState]:
        """Apply one scalar focal action under the fail-closed transaction."""

        result = self.step_result(state, focal_action)
        return result.transition, result.state

    def step_result(
        self,
        state: HiddenPartnerWorldFeedbackState,
        focal_action: Array,
    ) -> HiddenPartnerWorldFeedbackStepResult:
        """Attempt one simultaneous transition with bit-exact rollback.

        Valid dynamic action ids are ``0`` and ``1``.  Shape and integer dtype
        are checked before tracing; an out-of-range traced value rejects the
        transaction without advancing any state leaf.
        """
        self._require_state_contract(state)
        action = jnp.asarray(focal_action)
        if action.shape != ():
            raise ValueError(f"focal_action must be scalar, got shape {action.shape}")
        if not jnp.issubdtype(action.dtype, jnp.integer):
            raise TypeError("focal_action must have an integer dtype")
        input_valid = (action == NEGATIVE_ACTION) | (action == POSITIVE_ACTION)
        action_id = action.astype(jnp.int32)

        observation = self.observe(state)
        proposed_words, lifetime_capacity_available = _checked_words_increment(state.step_words)
        lifetime_counter_valid = _lifetime_counter_valid(
            state.step_words,
            state.step_count,
        )
        state_valid = self._state_values_valid(state)
        schedule = self._schedule_position(state, state.step_words)
        intended_sign = self._partner_intended_sign(state, schedule.regime_id)

        next_partner_key, partner_flip_key = jr.split(state.partner_key)
        partner_flipped = jr.bernoulli(
            partner_flip_key,
            p=jnp.asarray(self._config.partner_flip_probability, dtype=jnp.float32),
        )
        partner_sign = jnp.where(partner_flipped, -intended_sign, intended_sign).astype(jnp.float32)
        partner_action = ((partner_sign + 1.0) / 2.0).astype(jnp.int32)

        safe_action_id = jnp.where(
            input_valid,
            action_id,
            jnp.asarray(NEGATIVE_ACTION, dtype=jnp.int32),
        )
        focal_sign = (2.0 * safe_action_id.astype(jnp.float32) - 1.0).astype(jnp.float32)
        safe_world_sign = jnp.where(
            (state.world_sign == -1.0) | (state.world_sign == 1.0),
            state.world_sign,
            jnp.asarray(1.0, dtype=jnp.float32),
        )
        noiseless_outcome = (safe_world_sign * focal_sign * partner_sign).astype(jnp.float32)
        next_outcome_key, outcome_flip_key = jr.split(state.outcome_key)
        outcome_flipped = jr.bernoulli(
            outcome_flip_key,
            p=jnp.asarray(self._config.outcome_flip_probability, dtype=jnp.float32),
        )
        outcome = jnp.where(
            outcome_flipped,
            -noiseless_outcome,
            noiseless_outcome,
        ).astype(jnp.float32)
        reward = ((1.0 + outcome) / 2.0).astype(jnp.float32)

        next_world_key, world_flip_key = jr.split(state.world_key)
        world_flipped = jr.bernoulli(
            world_flip_key,
            p=jnp.asarray(self._config.world_flip_probability, dtype=jnp.float32),
        )
        next_world_sign = jnp.where(
            world_flipped,
            -safe_world_sign,
            safe_world_sign,
        ).astype(jnp.float32)
        next_cue_key, cue_sample_key = jr.split(state.cue_key)
        next_cues = self._sample_cues(cue_sample_key, next_world_sign)
        next_signal_key, signal_sample_key = jr.split(state.signal_key)
        candidate_state = HiddenPartnerWorldFeedbackState(
            signal_key=next_signal_key,
            partner_key=next_partner_key,
            world_key=next_world_key,
            cue_key=next_cue_key,
            outcome_key=next_outcome_key,
            segment_lengths=state.segment_lengths,
            segment_ends=state.segment_ends,
            current_signals=self._sample_signals(signal_sample_key),
            current_cues=next_cues,
            world_sign=next_world_sign,
            previous_outcome=outcome,
            previous_partner_action=partner_sign.astype(jnp.int32),
            has_partner_history=jnp.asarray(True, dtype=jnp.bool_),
            step_count=_words_to_saturating_int32(proposed_words),
            step_words=proposed_words,
        )
        next_schedule = self._schedule_position(candidate_state, proposed_words)

        focal_signs = jnp.asarray((-1.0, 1.0), dtype=jnp.float32)
        outcome_noise_sign = jnp.where(outcome_flipped, -1.0, 1.0).astype(jnp.float32)
        counterfactual_rewards = (
            1.0 + safe_world_sign * focal_signs * partner_sign * outcome_noise_sign
        ) / 2.0
        intended_action = ((intended_sign + 1.0) / 2.0).astype(jnp.int32)
        full_information_coefficient = (
            (1.0 - 2.0 * self._config.partner_flip_probability)
            * (1.0 - 2.0 * self._config.outcome_flip_probability)
            * safe_world_sign
            * intended_sign
        ).astype(jnp.float32)
        optimal_sign = jnp.where(full_information_coefficient >= 0.0, 1.0, -1.0)
        optimal_action = ((optimal_sign + 1.0) / 2.0).astype(jnp.int32)
        candidate_oracle = HiddenPartnerWorldFeedbackOracle(
            step_count=state.step_count,
            step_words=state.step_words,
            cycle_index=schedule.cycle_index,
            cycle_words=schedule.cycle_words,
            cycle_step=schedule.cycle_step,
            cycle_length=schedule.cycle_length,
            segment_index=schedule.segment_index,
            segment_step=schedule.segment_step,
            segment_length=schedule.segment_length,
            regime_id=schedule.regime_id,
            next_step_words=proposed_words,
            next_cycle_index=next_schedule.cycle_index,
            next_cycle_words=next_schedule.cycle_words,
            next_segment_index=next_schedule.segment_index,
            next_regime_id=next_schedule.regime_id,
            schedule_switched=(
                (~jnp.array_equal(schedule.cycle_words, next_schedule.cycle_words))
                | (schedule.segment_index != next_schedule.segment_index)
            ),
            partner_intended_action=intended_action,
            partner_intended_sign=intended_sign,
            partner_flipped=partner_flipped,
            focal_action_sign=focal_sign,
            partner_action_sign=partner_sign,
            world_sign=safe_world_sign,
            world_cue_flipped=state.current_cues != safe_world_sign,
            world_flipped=world_flipped,
            next_world_sign=next_world_sign,
            next_world_cue_flipped=next_cues != next_world_sign,
            outcome_flipped=outcome_flipped,
            noiseless_contextual_outcome=noiseless_outcome,
            contextual_outcome=outcome,
            counterfactual_rewards=counterfactual_rewards.astype(jnp.float32),
            full_information_optimal_focal_action=optimal_action,
            full_information_action_margin=jnp.abs(full_information_coefficient),
            full_information_action_tied=full_information_coefficient == 0.0,
        )
        candidate_transition = HiddenPartnerWorldFeedbackTransition(
            observation=observation,
            focal_action=action_id,
            partner_action=partner_action,
            reward=reward,
            outcome=outcome,
            next_observation=self.observe(candidate_state),
            terminated=jnp.asarray(False, dtype=jnp.bool_),
            discount=jnp.asarray(1.0, dtype=jnp.float32),
            oracle=candidate_oracle,
        )
        candidate_state_finite = _floating_tree_is_finite((candidate_state, candidate_transition))
        candidate_state_valid = self._state_values_valid(candidate_state)
        update_applied = (
            state_valid
            & input_valid
            & lifetime_capacity_available
            & candidate_state_finite
            & candidate_state_valid
        )
        committed_state = jax.lax.cond(
            update_applied,
            lambda _: candidate_state,
            lambda _: state,
            operand=None,
        )

        neutral_oracle = HiddenPartnerWorldFeedbackOracle(
            step_count=state.step_count,
            step_words=state.step_words,
            cycle_index=jnp.asarray(-1, dtype=jnp.int32),
            cycle_words=jnp.zeros((2,), dtype=jnp.uint32),
            cycle_step=jnp.asarray(-1, dtype=jnp.int32),
            cycle_length=jnp.asarray(0, dtype=jnp.int32),
            segment_index=jnp.asarray(-1, dtype=jnp.int32),
            segment_step=jnp.asarray(-1, dtype=jnp.int32),
            segment_length=jnp.asarray(0, dtype=jnp.int32),
            regime_id=jnp.asarray(-1, dtype=jnp.int32),
            next_step_words=state.step_words,
            next_cycle_index=jnp.asarray(-1, dtype=jnp.int32),
            next_cycle_words=jnp.zeros((2,), dtype=jnp.uint32),
            next_segment_index=jnp.asarray(-1, dtype=jnp.int32),
            next_regime_id=jnp.asarray(-1, dtype=jnp.int32),
            schedule_switched=jnp.asarray(False, dtype=jnp.bool_),
            partner_intended_action=jnp.asarray(-1, dtype=jnp.int32),
            partner_intended_sign=jnp.asarray(0.0, dtype=jnp.float32),
            partner_flipped=jnp.asarray(False, dtype=jnp.bool_),
            focal_action_sign=jnp.asarray(0.0, dtype=jnp.float32),
            partner_action_sign=jnp.asarray(0.0, dtype=jnp.float32),
            world_sign=jnp.asarray(0.0, dtype=jnp.float32),
            world_cue_flipped=jnp.zeros((2,), dtype=jnp.bool_),
            world_flipped=jnp.asarray(False, dtype=jnp.bool_),
            next_world_sign=jnp.asarray(0.0, dtype=jnp.float32),
            next_world_cue_flipped=jnp.zeros((2,), dtype=jnp.bool_),
            outcome_flipped=jnp.asarray(False, dtype=jnp.bool_),
            noiseless_contextual_outcome=jnp.asarray(0.0, dtype=jnp.float32),
            contextual_outcome=jnp.asarray(0.0, dtype=jnp.float32),
            counterfactual_rewards=jnp.zeros((2,), dtype=jnp.float32),
            full_information_optimal_focal_action=jnp.asarray(-1, dtype=jnp.int32),
            full_information_action_margin=jnp.asarray(0.0, dtype=jnp.float32),
            full_information_action_tied=jnp.asarray(True, dtype=jnp.bool_),
        )
        neutral_transition = HiddenPartnerWorldFeedbackTransition(
            observation=observation,
            focal_action=jnp.asarray(-1, dtype=jnp.int32),
            partner_action=jnp.asarray(-1, dtype=jnp.int32),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            outcome=jnp.asarray(0.0, dtype=jnp.float32),
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
        return HiddenPartnerWorldFeedbackStepResult(
            transition=transition,
            state=committed_state,
            pre_step_words=state.step_words,
            post_step_words=committed_state.step_words,
            lifetime_counter_valid=lifetime_counter_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            state_valid=state_valid,
            input_valid=input_valid,
            candidate_state_finite=candidate_state_finite,
            candidate_state_valid=candidate_state_valid,
            update_applied=update_applied,
        )

    @staticmethod
    def _sample_sign(key: Array) -> Array:
        return jnp.where(jr.bernoulli(key), 1.0, -1.0).astype(jnp.float32)

    @staticmethod
    def _sample_signals(key: Array) -> Array:
        positive = jr.bernoulli(key, p=0.5, shape=(_N_CURRENT_SIGNALS,))
        return jnp.where(positive, 1.0, -1.0).astype(jnp.float32)

    def _sample_cues(self, key: Array, world_sign: Array) -> Array:
        flipped = jr.bernoulli(
            key,
            p=jnp.asarray(self._config.cue_flip_probabilities, dtype=jnp.float32),
            shape=(_N_CUES,),
        )
        return jnp.where(flipped, -world_sign, world_sign).astype(jnp.float32)

    def _partner_intended_sign(
        self,
        state: HiddenPartnerWorldFeedbackState,
        regime_id: Array,
    ) -> Array:
        safe_signals = jnp.where(
            jnp.isfinite(state.current_signals),
            state.current_signals,
            jnp.ones_like(state.current_signals),
        )
        x, u, v = safe_signals
        safe_previous_partner = jnp.where(
            (state.previous_partner_action == -1) | (state.previous_partner_action == 1),
            state.previous_partner_action,
            jnp.asarray(1, dtype=jnp.int32),
        )
        targets = jnp.stack(
            (
                x,
                -x,
                x * safe_previous_partner.astype(jnp.float32),
                u * v,
            )
        )
        safe_regime = jnp.clip(regime_id, REGIME_A, REGIME_D)
        return targets[safe_regime].astype(jnp.float32)

    def _schedule_position(
        self,
        state: HiddenPartnerWorldFeedbackState,
        step_identity: Array,
    ) -> _SchedulePosition:
        supplied = jnp.asarray(step_identity)
        if supplied.shape == ():
            if supplied.dtype != jnp.dtype(jnp.int32):
                raise TypeError("legacy schedule telemetry must have dtype int32")
            # Compatibility for existing evaluator-only callers.  The scalar
            # is never schedule authority; the authenticated state words are.
            step_words = state.step_words
        else:
            step_words = supplied
        raw_cycle_length = state.segment_ends[-1]
        cycle_length = jnp.where(
            raw_cycle_length > 0,
            raw_cycle_length,
            jnp.asarray(1, dtype=jnp.int32),
        )
        cycle_words, cycle_step_u = _divmod_words_by_positive_int32(
            step_words,
            cycle_length,
        )
        cycle_index = _words_to_saturating_int32(cycle_words)
        cycle_step = cycle_step_u.astype(jnp.int32)
        segment_index = jnp.clip(
            jnp.sum(cycle_step >= state.segment_ends).astype(jnp.int32),
            0,
            _N_SEGMENTS - 1,
        )
        prior_index = jnp.maximum(segment_index - 1, 0)
        segment_start = jnp.where(
            segment_index == 0,
            jnp.asarray(0, dtype=jnp.int32),
            state.segment_ends[prior_index],
        )
        return _SchedulePosition(
            cycle_index=cycle_index,
            cycle_words=cycle_words,
            cycle_step=cycle_step,
            cycle_length=cycle_length,
            segment_index=segment_index,
            segment_step=(cycle_step - segment_start).astype(jnp.int32),
            segment_length=state.segment_lengths[segment_index],
            regime_id=self._regime_schedule[segment_index],
        )


__all__ = [
    "CUE_1_INDEX",
    "CUE_2_INDEX",
    "DEFAULT_CUE_FLIP_PROBABILITIES",
    "DEFAULT_OUTCOME_FLIP_PROBABILITY",
    "DEFAULT_WORLD_FLIP_PROBABILITY",
    "HAS_PARTNER_HISTORY_INDEX",
    "HIDDEN_PARTNER_WORLD_FEEDBACK_CONFIG_SCHEMA",
    "HIDDEN_PARTNER_WORLD_FEEDBACK_CONTRACT_VERSION",
    "HIDDEN_PARTNER_WORLD_FEEDBACK_EXACT_IDENTITY_NBYTES",
    "HIDDEN_PARTNER_WORLD_FEEDBACK_STATE_SCHEMA",
    "HiddenPartnerWorldFeedbackConfig",
    "HiddenPartnerWorldFeedbackOracle",
    "HiddenPartnerWorldFeedbackResourceBudget",
    "HiddenPartnerWorldFeedbackState",
    "HiddenPartnerWorldFeedbackStepResult",
    "HiddenPartnerWorldFeedbackTransition",
    "HiddenPartnerWorldFeedbackWorld",
    "N_ACTIONS",
    "OBSERVATION_DIM",
    "OBSERVATION_FIELDS",
    "PREVIOUS_OUTCOME_INDEX",
    "PREVIOUS_PARTNER_ACTION_INDEX",
    "REGIME_A",
    "REGIME_B",
    "REGIME_C",
    "REGIME_D",
    "REGIME_NAMES",
    "U_INDEX",
    "V_INDEX",
    "X_INDEX",
    "measure_hidden_partner_world_feedback_state_nbytes",
    "migrate_legacy_hidden_partner_world_feedback_config",
    "migrate_legacy_hidden_partner_world_feedback_state",
]
