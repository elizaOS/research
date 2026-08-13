"""Uncued recurring partner mappings for integrated continual-learning lives.

This module provides the causal environment substrate for the
``hidden-partner-mapping-v0`` development protocol.  It is deliberately
different from the narrower context-inference and discovered-feature-retention
lanes:

* the active regime is never present in the ordinary observation;
* a partner acts simultaneously with the focal agent;
* the current partner action is independent of the focal action, so
  counterfactual action comparisons are causal;
* useful instantaneous, temporal, and disposable feature products occur in
  one uninterrupted recurring life; and
* boundaries change only the hidden mapping -- they do not terminate or reset
  the stream.

The exact default hidden schedule is

``A -> B -> A -> D -> A -> C -> A -> B -> C``

with base segment lengths

``(1536, 1792, 1536, 1280, 1536, 1792, 1536, 1792, 1792)``.

Each length receives one seeded, evaluator-only integer jitter in ``[-63, 63]``
at initialization.  The resulting nine-segment schedule repeats if a caller
runs beyond the first benchmark cycle; the global counter never resets.

Causal transition convention
----------------------------
At decision time ``t`` the focal agent observes

``[x_t, y_t, b_{t-1}, has_history, u_t, v_t, n1_t, n2_t]``,

where ``y_t`` is the preceding joint outcome (zero only at birth), and
``b_{t-1}`` is the preceding partner action sign (zero only at birth).  The
current partner action is then generated simultaneously from the hidden
regime:

* A: ``b*_t = x_t``
* B: ``b*_t = -x_t``
* C: ``b*_t = x_t * b_{t-1}``
* D: ``b*_t = u_t * v_t``

The intended sign is independently flipped with probability ``0.05``.  Focal
and partner action ids use the common mapping ``0 -> -1`` and ``1 -> +1``.
After both actions are fixed, ``y_{t+1} = a_t * b_t`` and
``r_{t+1} = (1 + y_{t+1}) / 2``.  Fresh Rademacher signals are sampled only
for the next observation.  Every admitted protocol transition has discount
one.  The sole terminal transition is a finite no-op returned if the exact
64-bit lifetime identity is exhausted.

The hidden schedule and its boundary diagnostics live only in
:class:`HiddenPartnerMappingOracle`; :meth:`HiddenPartnerMappingWorld.observe`
returns exactly the eight ordinary channels above.  The scripted stochastic
partner makes v0 a controlled causal development environment, not by itself a
claim of learning-partner co-adaptation or complete Alberta Plan evidence.
"""

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

HIDDEN_PARTNER_MAPPING_CONTRACT_VERSION = "hidden-partner-mapping-v0"
HIDDEN_PARTNER_MAPPING_CONFIG_SCHEMA = "alberta.hidden-partner-mapping.config.v2"
HIDDEN_PARTNER_MAPPING_STATE_SCHEMA = "alberta.hidden-partner-mapping.state.v2"
HIDDEN_PARTNER_MAPPING_EXACT_IDENTITY_NBYTES = 8

REGIME_A = 0
REGIME_B = 1
REGIME_C = 2
REGIME_D = 3
REGIME_NAMES = ("A", "B", "C", "D")

DEFAULT_REGIME_SCHEDULE = (
    REGIME_A,
    REGIME_B,
    REGIME_A,
    REGIME_D,
    REGIME_A,
    REGIME_C,
    REGIME_A,
    REGIME_B,
    REGIME_C,
)
DEFAULT_BASE_SEGMENT_LENGTHS = (
    1536,
    1792,
    1536,
    1280,
    1536,
    1792,
    1536,
    1792,
    1792,
)
DEFAULT_JITTER_RADIUS = 63
DEFAULT_PARTNER_FLIP_PROBABILITY = 0.05

NEGATIVE_ACTION = 0
POSITIVE_ACTION = 1
N_ACTIONS = 2

X_INDEX = 0
PREVIOUS_OUTCOME_INDEX = 1
PREVIOUS_PARTNER_ACTION_INDEX = 2
HAS_PARTNER_HISTORY_INDEX = 3
U_INDEX = 4
V_INDEX = 5
NUISANCE_1_INDEX = 6
NUISANCE_2_INDEX = 7
OBSERVATION_FIELDS = (
    "x",
    "previous_joint_outcome",
    "previous_partner_action",
    "has_partner_history",
    "u",
    "v",
    "nuisance_1",
    "nuisance_2",
)
OBSERVATION_DIM = len(OBSERVATION_FIELDS)

_N_SEGMENTS = len(DEFAULT_REGIME_SCHEDULE)
_N_CURRENT_SIGNALS = 5  # x, u, v, n1, n2
_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1

# Named substreams keep schedule generation, ordinary signals, and partner
# stochasticity invariant to changes in one another's sampling code.
_SCHEDULE_RNG_TAG = 0
_SIGNAL_RNG_TAG = 1
_PARTNER_RNG_TAG = 2


@dataclasses.dataclass(frozen=True)
class HiddenPartnerMappingConfig:
    """Static, strictly versioned configuration for the v0 environment."""

    contract_version: str = HIDDEN_PARTNER_MAPPING_CONTRACT_VERSION
    regime_schedule: tuple[int, ...] = DEFAULT_REGIME_SCHEDULE
    base_segment_lengths: tuple[int, ...] = DEFAULT_BASE_SEGMENT_LENGTHS
    jitter_radius: int = DEFAULT_JITTER_RADIUS
    partner_flip_probability: float = DEFAULT_PARTNER_FLIP_PROBABILITY

    def __post_init__(self) -> None:
        """Reject schema drift and values unsafe for fixed-shape JAX state."""
        if self.contract_version != HIDDEN_PARTNER_MAPPING_CONTRACT_VERSION:
            raise ValueError(
                f"contract_version must be {HIDDEN_PARTNER_MAPPING_CONTRACT_VERSION!r}"
            )
        if not isinstance(self.regime_schedule, tuple):
            raise ValueError("regime_schedule must be a tuple")
        if any(type(regime) is not int for regime in self.regime_schedule):
            raise ValueError("regime_schedule must contain non-boolean integers")
        if self.regime_schedule != DEFAULT_REGIME_SCHEDULE:
            raise ValueError(
                "regime_schedule must be the exact v0 A->B->A->D->A->C->A->B->C schedule"
            )
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
                "every base segment length must exceed jitter_radius so all "
                "jittered lengths remain positive"
            )
        maximum_cycle_length = sum(self.base_segment_lengths) + (_N_SEGMENTS * self.jitter_radius)
        if maximum_cycle_length > _INT32_MAX:
            raise ValueError("maximum jittered cycle length must fit in int32")
        probability = self.partner_flip_probability
        if (
            isinstance(probability, bool)
            or not isinstance(probability, Real)
            or not math.isfinite(float(probability))
            or not 0.0 <= float(probability) <= 1.0
        ):
            raise ValueError("partner_flip_probability must be finite and lie in [0, 1]")

    def to_config(self) -> dict[str, Any]:
        """Return a deterministic, JSON-compatible scientific configuration."""
        return {
            "schema": HIDDEN_PARTNER_MAPPING_CONFIG_SCHEMA,
            "state_schema": HIDDEN_PARTNER_MAPPING_STATE_SCHEMA,
            "type": type(self).__name__,
            "contract_version": self.contract_version,
            "regime_schedule": list(self.regime_schedule),
            "base_segment_lengths": list(self.base_segment_lengths),
            "jitter_radius": self.jitter_radius,
            "partner_flip_probability": float(self.partner_flip_probability),
        }

    def canonical_json(self) -> str:
        """Serialize with a canonical key order and no non-finite numbers."""
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
    ) -> HiddenPartnerMappingConfig:
        """Strictly reconstruct a v0 configuration."""
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
        }
        if set(payload) != expected:
            if "schema" not in payload:
                raise ValueError("legacy hidden-partner config requires explicit migration")
            raise ValueError("config fields do not match the serialized schema")
        schema = payload.pop("schema")
        if schema != HIDDEN_PARTNER_MAPPING_CONFIG_SCHEMA:
            raise ValueError(f"unexpected config schema: {schema!r}")
        state_schema = payload.pop("state_schema")
        if state_schema != HIDDEN_PARTNER_MAPPING_STATE_SCHEMA:
            raise ValueError(f"unexpected state schema: {state_schema!r}")
        type_name = payload.pop("type")
        if type_name != cls.__name__:
            raise ValueError(f"unexpected config type: {type_name!r}")
        regime_schedule = payload.pop("regime_schedule")
        base_segment_lengths = payload.pop("base_segment_lengths")
        if not isinstance(regime_schedule, (list, tuple)):
            raise ValueError("regime_schedule must serialize as a sequence")
        if not isinstance(base_segment_lengths, (list, tuple)):
            raise ValueError("base_segment_lengths must serialize as a sequence")
        return cls(
            regime_schedule=tuple(regime_schedule),
            base_segment_lengths=tuple(base_segment_lengths),
            **payload,
        )


def migrate_legacy_hidden_partner_mapping_config(
    legacy_config: Mapping[str, Any],
) -> HiddenPartnerMappingConfig:
    """Explicitly stamp one exact schema-less pre-v2 configuration."""

    if not isinstance(legacy_config, Mapping):
        raise TypeError("legacy hidden-partner config must be a mapping")
    payload = dict(legacy_config)
    expected = {
        "type",
        "contract_version",
        "regime_schedule",
        "base_segment_lengths",
        "jitter_radius",
        "partner_flip_probability",
    }
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        extra = sorted(set(payload) - expected)
        raise ValueError(
            f"legacy hidden-partner config fields are not exact; missing={missing}, extra={extra}"
        )
    payload["schema"] = HIDDEN_PARTNER_MAPPING_CONFIG_SCHEMA
    payload["state_schema"] = HIDDEN_PARTNER_MAPPING_STATE_SCHEMA
    return HiddenPartnerMappingConfig.from_config(payload)


@dataclasses.dataclass(frozen=True)
class HiddenPartnerMappingResourceBudget:
    """Exact logical persistent-state accounting.

    ``state_nbytes`` counts only leaves of
    :class:`HiddenPartnerMappingState`: float32/int32 scalars, one boolean,
    two exact lifetime words, and two JAX PRNG keys counted as four logical
    uint32 scalars.  It excludes Python/config objects, transition outputs,
    compiler buffers, and device alignment.
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
class HiddenPartnerMappingState:
    """Fixed-shape causal state for one uninterrupted environment life."""

    signal_key: PRNGKeyArray
    partner_key: PRNGKeyArray
    segment_lengths: Int[Array, " 9"]
    segment_ends: Int[Array, " 9"]
    current_signals: Float[Array, " 5"]
    previous_outcome: Float[Array, ""]
    previous_partner_action: Int[Array, ""]
    has_partner_history: Bool[Array, ""]
    step_count: Int[Array, ""]
    step_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class HiddenPartnerMappingOracle:
    """Evaluator-only hidden schedule and causal diagnostics for one step."""

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
    joint_outcome: Float[Array, ""]
    counterfactual_rewards: Float[Array, " 2"]


@chex.dataclass(frozen=True)
class HiddenPartnerMappingTransition:
    """One continuing simultaneous-action transition."""

    observation: Float[Array, " 8"]
    focal_action: Int[Array, ""]
    partner_action: Int[Array, ""]
    reward: Float[Array, ""]
    outcome: Float[Array, ""]
    next_observation: Float[Array, " 8"]
    terminated: Bool[Array, ""]
    discount: Float[Array, ""]
    oracle: HiddenPartnerMappingOracle


@chex.dataclass(frozen=True)
class HiddenPartnerMappingStepResult:
    """One fail-closed world transaction and its exact lifetime audit."""

    transition: HiddenPartnerMappingTransition
    state: HiddenPartnerMappingState
    pre_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    lifetime_counter_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    state_valid: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    candidate_state_finite: Bool[Array, ""]
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
    """Return a nonthrowing exact array shape/dtype predicate."""

    return isinstance(value, Array) and value.shape == shape and value.dtype == dtype


def _checked_words_increment(
    words: Array,
) -> tuple[UInt[Array, " 2"], Bool[Array, ""]]:
    """Propose one exact lifetime increment without uint64-word wrap."""

    array = jnp.asarray(words)
    if array.shape != (2,):
        raise ValueError("hidden-partner step_words must have shape (2,)")
    if array.dtype != jnp.dtype(jnp.uint32):
        raise TypeError("hidden-partner step_words must have dtype uint32")
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
        raise ValueError("hidden-partner step_words must have shape (2,)")
    if array.dtype != jnp.dtype(jnp.uint32):
        raise TypeError("hidden-partner step_words must have dtype uint32")
    below_saturation = (array[0] == jnp.asarray(0, dtype=jnp.uint32)) & (
        array[1] < jnp.asarray(_INT32_MAX, dtype=jnp.uint32)
    )
    return jnp.where(
        below_saturation,
        array[1].astype(jnp.int32),
        jnp.asarray(_INT32_MAX, dtype=jnp.int32),
    )


def _lifetime_counter_valid(
    words: Array,
    telemetry: Array,
) -> Bool[Array, ""]:
    """Authenticate saturating telemetry against exact lifetime words."""

    count = jnp.asarray(telemetry)
    if count.shape != ():
        raise ValueError("hidden-partner step_count must be scalar")
    if count.dtype != jnp.dtype(jnp.int32):
        raise TypeError("hidden-partner step_count must have dtype int32")
    return (count >= 0) & (count == _words_to_saturating_int32(words))


def _divmod_words_by_positive_int32(
    words: Array,
    divisor: Array,
) -> tuple[UInt[Array, " 2"], UInt[Array, ""]]:
    """Exactly divide uint64 words by a dynamic positive int32 divisor."""

    array = jnp.asarray(words)
    if array.shape != (2,):
        raise ValueError("hidden-partner schedule words must have shape (2,)")
    if array.dtype != jnp.dtype(jnp.uint32):
        raise TypeError("hidden-partner schedule words must have dtype uint32")
    divisor_i = jnp.asarray(divisor)
    if divisor_i.shape != ():
        raise ValueError("hidden-partner cycle length must be scalar")
    if divisor_i.dtype != jnp.dtype(jnp.int32):
        raise TypeError("hidden-partner cycle length must have dtype int32")
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
    """Require every floating or complex array leaf to be finite."""

    valid = jnp.asarray(True, dtype=jnp.bool_)
    for leaf in jax.tree.leaves(value):
        if isinstance(leaf, Array) and jnp.issubdtype(leaf.dtype, jnp.inexact):
            valid = valid & jnp.all(jnp.isfinite(leaf))
    return valid


def measure_hidden_partner_mapping_state_nbytes(
    state: HiddenPartnerMappingState,
) -> int:
    """Measure every persistent JAX leaf in one environment state."""

    validator = HiddenPartnerMappingWorld()
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
    raise TypeError("legacy hidden-partner state must be a mapping or dataclass")


def migrate_legacy_hidden_partner_mapping_state(
    legacy_state: Any,
) -> HiddenPartnerMappingState:
    """Migrate one exact pre-v2 state whose unsaturated int32 time is unique."""

    fields = _host_field_mapping(legacy_state)
    current_names = {
        field.name for field in dataclasses.fields(cast(Any, HiddenPartnerMappingState))
    }
    legacy_names = current_names - {"step_words"}
    if set(fields) != legacy_names:
        missing = sorted(legacy_names - set(fields))
        extra = sorted(set(fields) - legacy_names)
        raise ValueError(
            f"legacy hidden-partner state fields are not exact; missing={missing}, extra={extra}"
        )
    step_array = jnp.asarray(fields["step_count"])
    if step_array.shape != () or step_array.dtype != jnp.dtype(jnp.int32):
        raise TypeError("legacy hidden-partner step_count must be scalar int32")
    step = int(step_array)
    if step < 0:
        raise ValueError("negative legacy hidden-partner step_count indicates wrap")
    if step >= _INT32_MAX:
        raise ValueError("saturated legacy hidden-partner step_count is ambiguous")
    fields["step_words"] = jnp.asarray((0, step), dtype=jnp.uint32)
    migrated = HiddenPartnerMappingState(**fields)
    validator = HiddenPartnerMappingWorld()
    validator._require_state_contract(migrated)
    if not bool(jax.device_get(validator._state_values_valid(migrated))):
        raise ValueError("legacy hidden-partner state values are inconsistent")
    return migrated


class HiddenPartnerMappingWorld:
    """Pure JAX implementation of the hidden-partner-mapping-v0 contract."""

    def __init__(
        self,
        config: HiddenPartnerMappingConfig | None = None,
    ) -> None:
        self._config = HiddenPartnerMappingConfig() if config is None else config
        if not isinstance(self._config, HiddenPartnerMappingConfig):
            raise TypeError("config must be a HiddenPartnerMappingConfig")
        self._base_segment_lengths = jnp.asarray(
            self._config.base_segment_lengths,
            dtype=jnp.int32,
        )
        self._regime_schedule = jnp.asarray(
            self._config.regime_schedule,
            dtype=jnp.int32,
        )

    @property
    def config(self) -> HiddenPartnerMappingConfig:
        """Immutable static configuration."""
        return self._config

    @property
    def observation_dim(self) -> int:
        """Number of ordinary (non-oracle) input channels."""
        return OBSERVATION_DIM

    @property
    def feature_dim(self) -> int:
        """Alias for the ordinary observation width."""
        return OBSERVATION_DIM

    @property
    def n_actions(self) -> int:
        """Number of focal and partner action ids."""
        return N_ACTIONS

    @property
    def n_segments(self) -> int:
        """Number of segments in the exact v0 recurring schedule."""
        return _N_SEGMENTS

    @property
    def resource_budget(self) -> HiddenPartnerMappingResourceBudget:
        """Return exact fixed environment-state accounting."""
        float_scalars = _N_CURRENT_SIGNALS + 1
        int_scalars = 2 * _N_SEGMENTS + 2
        bool_scalars = 1
        exact_identity_scalars = 2
        rng_scalars = 4
        state_scalars = (
            float_scalars + int_scalars + bool_scalars + exact_identity_scalars + rng_scalars
        )
        state_nbytes = (
            4 * (float_scalars + int_scalars + exact_identity_scalars + rng_scalars) + bool_scalars
        )
        return HiddenPartnerMappingResourceBudget(
            state_schema=HIDDEN_PARTNER_MAPPING_STATE_SCHEMA,
            observation_float32_scalars=OBSERVATION_DIM,
            persistent_float32_scalars=float_scalars,
            persistent_int32_scalars=int_scalars,
            persistent_bool_scalars=bool_scalars,
            exact_identity_uint32_scalars=exact_identity_scalars,
            exact_identity_nbytes=HIDDEN_PARTNER_MAPPING_EXACT_IDENTITY_NBYTES,
            lifetime_identity_bits=64,
            telemetry_saturation=_INT32_MAX,
            rng_uint32_scalars=rng_scalars,
            persistent_state_scalars=state_scalars,
            state_nbytes=state_nbytes,
            trainable_scalars=0,
            replay_capacity=0,
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize the world without dynamic state."""
        return {
            "type": type(self).__name__,
            "config": self._config.to_config(),
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
    ) -> HiddenPartnerMappingWorld:
        """Strictly reconstruct a world from :meth:`to_config` output."""
        payload = dict(config)
        if set(payload) != {"type", "config"}:
            raise ValueError("world config must contain exactly 'type' and 'config'")
        type_name = payload["type"]
        if type_name != cls.__name__:
            raise ValueError(f"unexpected world type: {type_name!r}")
        inner = payload["config"]
        if not isinstance(inner, Mapping):
            raise ValueError("world 'config' must be a mapping")
        return cls(HiddenPartnerMappingConfig.from_config(inner))

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

    def _require_state_contract(self, state: HiddenPartnerMappingState) -> None:
        """Validate every fixed state shape and dtype before indexed work."""

        if not isinstance(state, HiddenPartnerMappingState):
            raise TypeError("state must be a HiddenPartnerMappingState")
        self._require_key_contract(state.signal_key, name="signal_key")
        self._require_key_contract(state.partner_key, name="partner_key")
        contracts = (
            ("segment_lengths", state.segment_lengths, (_N_SEGMENTS,), jnp.int32),
            ("segment_ends", state.segment_ends, (_N_SEGMENTS,), jnp.int32),
            (
                "current_signals",
                state.current_signals,
                (_N_CURRENT_SIGNALS,),
                jnp.float32,
            ),
            ("previous_outcome", state.previous_outcome, (), jnp.float32),
            (
                "previous_partner_action",
                state.previous_partner_action,
                (),
                jnp.int32,
            ),
            (
                "has_partner_history",
                state.has_partner_history,
                (),
                jnp.bool_,
            ),
            ("step_count", state.step_count, (), jnp.int32),
            ("step_words", state.step_words, (2,), jnp.uint32),
        )
        for name, value, shape, dtype in contracts:
            if not _array_has_contract(value, shape, dtype):
                raise TypeError(f"hidden-partner {name} must have shape {shape} and dtype {dtype}")

    def _state_values_valid(
        self,
        state: HiddenPartnerMappingState,
    ) -> Bool[Array, ""]:
        """Authenticate exact time, schedule arrays, and causal history."""

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
            & jnp.isfinite(state.previous_outcome)
            & history_valid
            & _lifetime_counter_valid(state.step_words, state.step_count)
        )

    def init(self, key: Array) -> HiddenPartnerMappingState:
        """Initialize a deterministic life with independent named RNG streams."""
        self._require_key_contract(key, name="key")
        schedule_key = jr.fold_in(key, _SCHEDULE_RNG_TAG)
        signal_root_key = jr.fold_in(key, _SIGNAL_RNG_TAG)
        partner_key = jr.fold_in(key, _PARTNER_RNG_TAG)

        initial_signal_key, signal_key = jr.split(signal_root_key)
        jitters = jr.randint(
            schedule_key,
            shape=(_N_SEGMENTS,),
            minval=-self._config.jitter_radius,
            maxval=self._config.jitter_radius + 1,
            dtype=jnp.int32,
        )
        segment_lengths = self._base_segment_lengths + jitters
        return HiddenPartnerMappingState(  # type: ignore[call-arg]
            signal_key=signal_key,
            partner_key=partner_key,
            segment_lengths=segment_lengths,
            segment_ends=jnp.cumsum(segment_lengths, dtype=jnp.int32),
            current_signals=self._sample_signals(initial_signal_key),
            previous_outcome=jnp.asarray(0.0, dtype=jnp.float32),
            previous_partner_action=jnp.asarray(0, dtype=jnp.int32),
            has_partner_history=jnp.asarray(False, dtype=jnp.bool_),
            step_count=jnp.asarray(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def observe(self, state: HiddenPartnerMappingState) -> Array:
        """Return exactly the eight ordinary, task-oracle-free channels."""
        self._require_state_contract(state)
        x, u, v, nuisance_1, nuisance_2 = state.current_signals
        observation = jnp.stack(
            (
                x,
                state.previous_outcome,
                state.previous_partner_action.astype(jnp.float32),
                state.has_partner_history.astype(jnp.float32),
                u,
                v,
                nuisance_1,
                nuisance_2,
            )
        ).astype(jnp.float32)
        return jnp.where(
            self._state_values_valid(state),
            observation,
            jnp.zeros_like(observation),
        )

    def step(
        self,
        state: HiddenPartnerMappingState,
        focal_action: Array,
    ) -> tuple[HiddenPartnerMappingTransition, HiddenPartnerMappingState]:
        """Apply one scalar focal action under the fail-closed transaction."""

        result = self.step_result(state, focal_action)
        return result.transition, result.state

    def step_result(
        self,
        state: HiddenPartnerMappingState,
        focal_action: Array,
    ) -> HiddenPartnerMappingStepResult:
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
        next_partner_key, flip_key = jr.split(state.partner_key)
        partner_flipped = jr.bernoulli(
            flip_key,
            p=jnp.asarray(
                self._config.partner_flip_probability,
                dtype=jnp.float32,
            ),
        )
        partner_sign = jnp.where(
            partner_flipped,
            -intended_sign,
            intended_sign,
        ).astype(jnp.float32)
        partner_action = ((partner_sign + 1.0) / 2.0).astype(jnp.int32)

        safe_action_id = jnp.where(
            input_valid,
            action_id,
            jnp.asarray(NEGATIVE_ACTION, dtype=jnp.int32),
        )
        focal_sign = (2.0 * safe_action_id.astype(jnp.float32) - 1.0).astype(jnp.float32)
        outcome = (focal_sign * partner_sign).astype(jnp.float32)
        reward = ((1.0 + outcome) / 2.0).astype(jnp.float32)

        next_signal_key, signal_sample_key = jr.split(state.signal_key)
        candidate_state = HiddenPartnerMappingState(  # type: ignore[call-arg]
            signal_key=next_signal_key,
            partner_key=next_partner_key,
            segment_lengths=state.segment_lengths,
            segment_ends=state.segment_ends,
            current_signals=self._sample_signals(signal_sample_key),
            previous_outcome=outcome,
            previous_partner_action=partner_sign.astype(jnp.int32),
            has_partner_history=jnp.asarray(True, dtype=jnp.bool_),
            step_count=_words_to_saturating_int32(proposed_words),
            step_words=proposed_words,
        )
        next_schedule = self._schedule_position(candidate_state, proposed_words)

        counterfactual_signs = jnp.asarray((-1.0, 1.0), dtype=jnp.float32)
        counterfactual_rewards = (1.0 + counterfactual_signs * partner_sign) / 2.0
        intended_action = ((intended_sign + 1.0) / 2.0).astype(jnp.int32)
        candidate_oracle = HiddenPartnerMappingOracle(  # type: ignore[call-arg]
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
            joint_outcome=outcome,
            counterfactual_rewards=counterfactual_rewards.astype(jnp.float32),
        )
        candidate_transition = HiddenPartnerMappingTransition(  # type: ignore[call-arg]
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
        update_applied = (
            state_valid & input_valid & lifetime_capacity_available & candidate_state_finite
        )
        committed_state = jax.lax.cond(
            update_applied,
            lambda _: candidate_state,
            lambda _: state,
            operand=None,
        )

        neutral_oracle = HiddenPartnerMappingOracle(  # type: ignore[call-arg]
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
            joint_outcome=jnp.asarray(0.0, dtype=jnp.float32),
            counterfactual_rewards=jnp.zeros((2,), dtype=jnp.float32),
        )
        neutral_transition = HiddenPartnerMappingTransition(  # type: ignore[call-arg]
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
        return HiddenPartnerMappingStepResult(  # type: ignore[call-arg]
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

    @staticmethod
    def _sample_signals(key: Array) -> Array:
        """Sample ``x, u, v, n1, n2`` as independent Rademacher signs."""
        positive = jr.bernoulli(key, p=0.5, shape=(_N_CURRENT_SIGNALS,))
        return jnp.where(positive, 1.0, -1.0).astype(jnp.float32)

    def _partner_intended_sign(
        self,
        state: HiddenPartnerMappingState,
        regime_id: Array,
    ) -> Array:
        safe_signals = jnp.where(
            jnp.isfinite(state.current_signals),
            state.current_signals,
            jnp.ones_like(state.current_signals),
        )
        x, u, v, _, _ = safe_signals
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
        state: HiddenPartnerMappingState,
        step_words: Array,
    ) -> _SchedulePosition:
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
        return _SchedulePosition(  # type: ignore[call-arg]
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
    "DEFAULT_BASE_SEGMENT_LENGTHS",
    "DEFAULT_JITTER_RADIUS",
    "DEFAULT_PARTNER_FLIP_PROBABILITY",
    "DEFAULT_REGIME_SCHEDULE",
    "HAS_PARTNER_HISTORY_INDEX",
    "HIDDEN_PARTNER_MAPPING_CONFIG_SCHEMA",
    "HIDDEN_PARTNER_MAPPING_CONTRACT_VERSION",
    "HIDDEN_PARTNER_MAPPING_EXACT_IDENTITY_NBYTES",
    "HIDDEN_PARTNER_MAPPING_STATE_SCHEMA",
    "NEGATIVE_ACTION",
    "N_ACTIONS",
    "NUISANCE_1_INDEX",
    "NUISANCE_2_INDEX",
    "OBSERVATION_DIM",
    "OBSERVATION_FIELDS",
    "POSITIVE_ACTION",
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
    "HiddenPartnerMappingConfig",
    "HiddenPartnerMappingOracle",
    "HiddenPartnerMappingResourceBudget",
    "HiddenPartnerMappingState",
    "HiddenPartnerMappingStepResult",
    "HiddenPartnerMappingTransition",
    "HiddenPartnerMappingWorld",
    "measure_hidden_partner_mapping_state_nbytes",
    "migrate_legacy_hidden_partner_mapping_config",
    "migrate_legacy_hidden_partner_mapping_state",
]
