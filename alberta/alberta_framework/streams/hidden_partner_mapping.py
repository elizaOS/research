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
for the next observation.  Every transition has discount one and never
terminates.

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
from typing import Any

import chex
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Bool, Float, Int, PRNGKeyArray

HIDDEN_PARTNER_MAPPING_CONTRACT_VERSION = "hidden-partner-mapping-v0"

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
            "type",
            "contract_version",
            "regime_schedule",
            "base_segment_lengths",
            "jitter_radius",
            "partner_flip_probability",
        }
        if set(payload) != expected:
            raise ValueError("config fields do not match the serialized schema")
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


@dataclasses.dataclass(frozen=True)
class HiddenPartnerMappingResourceBudget:
    """Exact logical persistent-state accounting.

    ``state_nbytes`` counts only leaves of
    :class:`HiddenPartnerMappingState`: float32/int32 scalars, one boolean,
    and two JAX PRNG keys counted as four logical uint32 scalars.  It excludes
    Python/config objects, transition outputs, compiler buffers, and device
    alignment.
    """

    observation_float32_scalars: int
    persistent_float32_scalars: int
    persistent_int32_scalars: int
    persistent_bool_scalars: int
    rng_uint32_scalars: int
    persistent_state_scalars: int
    state_nbytes: int
    trainable_scalars: int
    replay_capacity: int

    def to_dict(self) -> dict[str, int]:
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


@chex.dataclass(frozen=True)
class HiddenPartnerMappingOracle:
    """Evaluator-only hidden schedule and causal diagnostics for one step."""

    step_count: Int[Array, ""]
    cycle_index: Int[Array, ""]
    cycle_step: Int[Array, ""]
    cycle_length: Int[Array, ""]
    segment_index: Int[Array, ""]
    segment_step: Int[Array, ""]
    segment_length: Int[Array, ""]
    regime_id: Int[Array, ""]
    next_cycle_index: Int[Array, ""]
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
class _SchedulePosition:
    """Internal JAX record for a position in the repeating hidden schedule."""

    cycle_index: Int[Array, ""]
    cycle_step: Int[Array, ""]
    cycle_length: Int[Array, ""]
    segment_index: Int[Array, ""]
    segment_step: Int[Array, ""]
    segment_length: Int[Array, ""]
    regime_id: Int[Array, ""]


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
        rng_scalars = 4
        state_scalars = float_scalars + int_scalars + bool_scalars + rng_scalars
        state_nbytes = 4 * (float_scalars + int_scalars + rng_scalars) + bool_scalars
        return HiddenPartnerMappingResourceBudget(
            observation_float32_scalars=OBSERVATION_DIM,
            persistent_float32_scalars=float_scalars,
            persistent_int32_scalars=int_scalars,
            persistent_bool_scalars=bool_scalars,
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

    def init(self, key: Array) -> HiddenPartnerMappingState:
        """Initialize a deterministic life with independent named RNG streams."""
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
        )

    def observe(self, state: HiddenPartnerMappingState) -> Array:
        """Return exactly the eight ordinary, task-oracle-free channels."""
        x, u, v, nuisance_1, nuisance_2 = state.current_signals
        return jnp.stack(
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

    def step(
        self,
        state: HiddenPartnerMappingState,
        focal_action: Array,
    ) -> tuple[HiddenPartnerMappingTransition, HiddenPartnerMappingState]:
        """Apply one scalar focal action id under exact simultaneous timing.

        Valid dynamic action ids are ``0`` and ``1``.  As with other
        scan-compatible streams, traced scalar values are not Python-validated;
        callers must keep ids in range.  Shape and integer dtype are static and
        therefore checked before tracing.
        """
        action = jnp.asarray(focal_action)
        if action.shape != ():
            raise ValueError(f"focal_action must be scalar, got shape {action.shape}")
        if not jnp.issubdtype(action.dtype, jnp.integer):
            raise TypeError("focal_action must have an integer dtype")
        action_id = action.astype(jnp.int32)

        observation = self.observe(state)
        schedule = self._schedule_position(state, state.step_count)

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

        action_valid = (action_id == NEGATIVE_ACTION) | (action_id == POSITIVE_ACTION)
        focal_sign = (2.0 * action_id.astype(jnp.float32) - 1.0).astype(jnp.float32)
        focal_sign = jnp.where(
            action_valid,
            focal_sign,
            jnp.asarray(jnp.nan, dtype=jnp.float32),
        )
        outcome = (focal_sign * partner_sign).astype(jnp.float32)
        reward = ((1.0 + outcome) / 2.0).astype(jnp.float32)

        next_signal_key, signal_sample_key = jr.split(state.signal_key)
        next_state = HiddenPartnerMappingState(  # type: ignore[call-arg]
            signal_key=next_signal_key,
            partner_key=next_partner_key,
            segment_lengths=state.segment_lengths,
            segment_ends=state.segment_ends,
            current_signals=self._sample_signals(signal_sample_key),
            previous_outcome=outcome,
            previous_partner_action=partner_sign.astype(jnp.int32),
            has_partner_history=jnp.asarray(True, dtype=jnp.bool_),
            step_count=state.step_count + jnp.asarray(1, dtype=jnp.int32),
        )
        next_schedule = self._schedule_position(next_state, next_state.step_count)

        counterfactual_signs = jnp.asarray((-1.0, 1.0), dtype=jnp.float32)
        counterfactual_rewards = (1.0 + counterfactual_signs * partner_sign) / 2.0
        intended_action = ((intended_sign + 1.0) / 2.0).astype(jnp.int32)
        oracle = HiddenPartnerMappingOracle(  # type: ignore[call-arg]
            step_count=state.step_count,
            cycle_index=schedule.cycle_index,
            cycle_step=schedule.cycle_step,
            cycle_length=schedule.cycle_length,
            segment_index=schedule.segment_index,
            segment_step=schedule.segment_step,
            segment_length=schedule.segment_length,
            regime_id=schedule.regime_id,
            next_cycle_index=next_schedule.cycle_index,
            next_segment_index=next_schedule.segment_index,
            next_regime_id=next_schedule.regime_id,
            schedule_switched=(
                (schedule.cycle_index != next_schedule.cycle_index)
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
        transition = HiddenPartnerMappingTransition(  # type: ignore[call-arg]
            observation=observation,
            focal_action=action_id,
            partner_action=partner_action,
            reward=reward,
            outcome=outcome,
            next_observation=self.observe(next_state),
            terminated=jnp.asarray(False, dtype=jnp.bool_),
            discount=jnp.asarray(1.0, dtype=jnp.float32),
            oracle=oracle,
        )
        return transition, next_state

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
        x, u, v, _, _ = state.current_signals
        targets = jnp.stack(
            (
                x,
                -x,
                x * state.previous_partner_action.astype(jnp.float32),
                u * v,
            )
        )
        return targets[regime_id].astype(jnp.float32)

    def _schedule_position(
        self,
        state: HiddenPartnerMappingState,
        step_count: Array,
    ) -> _SchedulePosition:
        cycle_length = state.segment_ends[-1]
        cycle_index = jnp.floor_divide(step_count, cycle_length).astype(jnp.int32)
        cycle_step = jnp.mod(step_count, cycle_length).astype(jnp.int32)
        segment_index = jnp.sum(cycle_step >= state.segment_ends).astype(jnp.int32)
        prior_index = jnp.maximum(segment_index - 1, 0)
        segment_start = jnp.where(
            segment_index == 0,
            jnp.asarray(0, dtype=jnp.int32),
            state.segment_ends[prior_index],
        )
        return _SchedulePosition(  # type: ignore[call-arg]
            cycle_index=cycle_index,
            cycle_step=cycle_step,
            cycle_length=cycle_length,
            segment_index=segment_index,
            segment_step=(cycle_step - segment_start).astype(jnp.int32),
            segment_length=state.segment_lengths[segment_index],
            regime_id=self._regime_schedule[segment_index],
        )
