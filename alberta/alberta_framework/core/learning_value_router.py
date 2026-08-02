# mypy: disable-error-code="call-arg"
"""Bounded, causal routing for eight typed learning-value channels.

The router is intentionally not a reward function and exposes no universal
score.  It validates each :class:`~alberta_framework.core.delight.LearningValue`
channel independently, updates a fixed-size per-channel normalization state,
and emits consumer-specific routes whose inactive fields are exact zero with
false availability.

Terminology is strict.  The historical ``LearningValue.delight`` field is the
actor-sample quantity ``advantage * action_surprisal`` defined by the
Delightful Policy Gradient (DG) paper, *Does This Gradient Spark Joy?*
(arXiv:2603.20526), where a sample ``sparks joy`` when the Kondo gate selects
it for a backward pass.  The separate
multi-objective candidate-update safety audit in
:mod:`alberta_framework.core.delight` retains older ``gradient_joy`` API names
only for compatibility.  This router computes neither Kondo selection nor an
update-quality verdict; it prepares typed routes for both consumers.

Normalization is causal.  A value at time ``t`` is standardized only from the
count, mean, and Welford ``M2`` stored before time ``t``.  The current value is
incorporated afterward.  Until a channel has the configured number of prior
valid samples its normalized value is exact zero and unavailable, while its
independently valid raw value may still be routed.

This is a bounded mechanism contract with fixed state, explicit logical
resource accounting, and no scientific-evidence or promotion claim.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from numbers import Real
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array
from jaxtyping import Bool, Float, Int

from alberta_framework.core.delight import LearningValue, LearningValueAvailability

LEARNING_VALUE_ROUTER_CONFIG_SCHEMA = "alberta.learning-value-router.config.v1"
LEARNING_VALUE_ROUTER_CHECKPOINT_SCHEMA = "alberta.learning-value-router.checkpoint.v1"
MECHANISM_STATUS = "development_mechanism_only"
SCIENTIFIC_PROMOTION_ALLOWED = False

_CONFIG_TYPE = "LearningValueRouterConfig"
_CHANNEL_COUNT = 8
_INT32_MAX = 2_147_483_647
_MAX_COUNTER_STEPS = _INT32_MAX // 3
_FLOAT32_MAX = float(np.finfo(np.float32).max)
_FLOAT32_TINY = float(np.finfo(np.float32).tiny)

_CHANNEL_NAMES = (
    "advantage",
    "action_surprisal",
    "delight",
    "epistemic_surprise",
    "aleatoric_uncertainty",
    "learning_progress",
    "change_probability",
    "safety_cost",
)

_PAPER_DG_ACTOR_MASK = (True, True, True, False, False, False, False, False)
_EXPLORATION_MASK = (False, False, False, True, True, True, False, False)
# Intentionally identical to _EXPLORATION_MASK: exploration and model-memory
# replay currently consume the same three channels, but each consumer keeps
# its own mask so the two route contracts can diverge independently.
_MODEL_MEMORY_REPLAY_MASK = (False, False, False, True, True, True, False, False)
_ADAPTATION_CHANGE_MASK = (False, False, False, False, False, True, True, False)
_SAFETY_MASK = (False, False, False, False, False, False, False, True)
_LITERAL_GRADIENT_JOY_EVIDENCE_MASK = (True,) * _CHANNEL_COUNT


def _strict_positive_float32(value: object, *, name: str) -> float:
    """Return a positive finite normal float32-compatible scalar."""

    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be a real scalar, not boolean")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    if normalized < _FLOAT32_TINY or normalized > _FLOAT32_MAX:
        raise ValueError(f"{name} must be a finite normal float32 value")
    return normalized


def _canonical_json_bytes(payload: object) -> bytes:
    """Return the one canonical JSON encoding used by checkpoint digests."""

    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _payload_digest(payload: object) -> str:
    """Digest a canonical JSON-compatible payload for accidental tamper checks."""

    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class LearningValueRouterConfig:
    """Static bounds and causal-normalization contract.

    Every magnitude bound is channel-specific.  Signed channels use an
    absolute bound; non-negative channels use a zero lower bound.  Change
    probability always has the semantic domain ``[0, 1]``.  Scale floors are
    in the corresponding channel's native units.  ``normalization_clip`` is a
    dimensionless absolute z-score cap.

    ``max_steps`` bounds exact int32 counters.  It is restricted so three
    mutually exclusive per-channel category counters can be added without
    int32 overflow during dynamic state validation.  Once capacity is reached,
    raw and frozen-normalization routing remains available and explicitly
    reports that calibration state was not updated.
    """

    normalization_min_count: int = 2
    max_steps: int = 1_000_000
    max_abs_advantage: float = 1.0e6
    max_action_surprisal: float = 1.0e6
    max_abs_paper_dg_delight: float = 1.0e12
    max_epistemic_surprise: float = 1.0e12
    max_aleatoric_uncertainty: float = 1.0e12
    max_abs_learning_progress: float = 1.0e12
    max_safety_cost: float = 1.0e12
    advantage_scale_floor: float = 1.0e-6
    action_surprisal_scale_floor: float = 1.0e-6
    paper_dg_delight_scale_floor: float = 1.0e-6
    epistemic_surprise_scale_floor: float = 1.0e-6
    aleatoric_uncertainty_scale_floor: float = 1.0e-6
    learning_progress_scale_floor: float = 1.0e-6
    change_probability_scale_floor: float = 1.0e-6
    safety_cost_scale_floor: float = 1.0e-6
    normalization_clip: float = 10.0

    def __post_init__(self) -> None:
        """Reject ambiguous counters, unsafe moments, and invalid controls."""

        if (
            type(self.normalization_min_count) is not int
            or self.normalization_min_count < 2
        ):
            raise ValueError("normalization_min_count must be a strict integer >= 2")
        if type(self.max_steps) is not int or not 1 <= self.max_steps <= _MAX_COUNTER_STEPS:
            raise ValueError(
                f"max_steps must be a strict integer in [1, {_MAX_COUNTER_STEPS}]"
            )
        if self.normalization_min_count > self.max_steps:
            raise ValueError("normalization_min_count must not exceed max_steps")

        maxima = (
            ("max_abs_advantage", self.max_abs_advantage),
            ("max_action_surprisal", self.max_action_surprisal),
            ("max_abs_paper_dg_delight", self.max_abs_paper_dg_delight),
            ("max_epistemic_surprise", self.max_epistemic_surprise),
            ("max_aleatoric_uncertainty", self.max_aleatoric_uncertainty),
            ("max_abs_learning_progress", self.max_abs_learning_progress),
            ("max_safety_cost", self.max_safety_cost),
        )
        for name, value in maxima:
            _strict_positive_float32(value, name=name)
        floors = (
            ("advantage_scale_floor", self.advantage_scale_floor),
            ("action_surprisal_scale_floor", self.action_surprisal_scale_floor),
            ("paper_dg_delight_scale_floor", self.paper_dg_delight_scale_floor),
            ("epistemic_surprise_scale_floor", self.epistemic_surprise_scale_floor),
            ("aleatoric_uncertainty_scale_floor", self.aleatoric_uncertainty_scale_floor),
            ("learning_progress_scale_floor", self.learning_progress_scale_floor),
            ("change_probability_scale_floor", self.change_probability_scale_floor),
            ("safety_cost_scale_floor", self.safety_cost_scale_floor),
        )
        for name, value in floors:
            _strict_positive_float32(value, name=name)
        _strict_positive_float32(self.normalization_clip, name="normalization_clip")

        # A Welford M2 is bounded by n times the squared channel span.  Keep a
        # factor-of-two headroom for conservative float32 accumulation.
        for name, span in zip(_CHANNEL_NAMES, self.channel_spans(), strict=True):
            maximum_m2 = 2.0 * float(self.max_steps) * float(span) * float(span)
            if not math.isfinite(maximum_m2) or maximum_m2 > _FLOAT32_MAX:
                raise ValueError(
                    f"{name} bound and max_steps cannot keep Welford M2 finite"
                )

    def channel_lower_bounds(self) -> tuple[float, ...]:
        """Return field-order semantic lower bounds."""

        return (
            -float(self.max_abs_advantage),
            0.0,
            -float(self.max_abs_paper_dg_delight),
            0.0,
            0.0,
            -float(self.max_abs_learning_progress),
            0.0,
            0.0,
        )

    def channel_upper_bounds(self) -> tuple[float, ...]:
        """Return field-order semantic upper bounds."""

        return (
            float(self.max_abs_advantage),
            float(self.max_action_surprisal),
            float(self.max_abs_paper_dg_delight),
            float(self.max_epistemic_surprise),
            float(self.max_aleatoric_uncertainty),
            float(self.max_abs_learning_progress),
            1.0,
            float(self.max_safety_cost),
        )

    def channel_spans(self) -> tuple[float, ...]:
        """Return upper-minus-lower ranges used for moment bounds."""

        return tuple(
            upper - lower
            for lower, upper in zip(
                self.channel_lower_bounds(),
                self.channel_upper_bounds(),
                strict=True,
            )
        )

    def channel_scale_floors(self) -> tuple[float, ...]:
        """Return field-order native-unit normalization floors."""

        return (
            float(self.advantage_scale_floor),
            float(self.action_surprisal_scale_floor),
            float(self.paper_dg_delight_scale_floor),
            float(self.epistemic_surprise_scale_floor),
            float(self.aleatoric_uncertainty_scale_floor),
            float(self.learning_progress_scale_floor),
            float(self.change_probability_scale_floor),
            float(self.safety_cost_scale_floor),
        )

    def to_config(self) -> dict[str, object]:
        """Return the complete canonical JSON-compatible configuration."""

        payload: dict[str, object] = {
            field.name: getattr(self, field.name) for field in dataclasses.fields(self)
        }
        for field in dataclasses.fields(self):
            if field.name not in {"normalization_min_count", "max_steps"}:
                payload[field.name] = float(cast(Real, payload[field.name]))
        return {
            "schema": LEARNING_VALUE_ROUTER_CONFIG_SCHEMA,
            "type": _CONFIG_TYPE,
            "mechanism_status": MECHANISM_STATUS,
            "scientific_promotion_allowed": SCIENTIFIC_PROMOTION_ALLOWED,
            **payload,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> LearningValueRouterConfig:
        """Strictly reconstruct only an exact :meth:`to_config` payload."""

        payload = dict(config)
        expected = {
            field.name for field in dataclasses.fields(cls)
        } | {
            "schema",
            "type",
            "mechanism_status",
            "scientific_promotion_allowed",
        }
        if set(payload) != expected:
            raise ValueError("config fields do not match the router v1 schema")
        schema = payload.pop("schema")
        if type(schema) is not str or schema != LEARNING_VALUE_ROUTER_CONFIG_SCHEMA:
            raise ValueError("unexpected learning-value router config schema")
        config_type = payload.pop("type")
        if type(config_type) is not str or config_type != _CONFIG_TYPE:
            raise ValueError("unexpected learning-value router config type")
        mechanism_status = payload.pop("mechanism_status")
        if type(mechanism_status) is not str or mechanism_status != MECHANISM_STATUS:
            raise ValueError("learning-value router must remain mechanism-only")
        if payload.pop("scientific_promotion_allowed") is not False:
            raise ValueError("learning-value router config cannot claim promotion")
        integer_fields = {"normalization_min_count", "max_steps"}
        for name, value in payload.items():
            if name in integer_fields:
                if type(value) is not int:
                    raise ValueError(f"serialized config {name} must be a JSON integer")
            elif type(value) is not float:
                raise ValueError(f"serialized config {name} must be a JSON float")
        return cls(**cast(dict[str, Any], payload))


@dataclasses.dataclass(frozen=True)
class LearningValueChannelMetadata:
    """One channel's immutable producer, causal object, units, and domain."""

    index: int
    field_name: str
    semantic_name: str
    producer: str
    causal_object: str
    units: str
    domain: str
    lower_bound: float
    upper_bound: float
    normalization: str
    normalization_scale_floor: float
    normalized_units: str
    normalized_lower_bound: float
    normalized_upper_bound: float

    def to_config(self) -> dict[str, object]:
        """Return an explicit JSON-compatible metadata record."""

        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class LearningValueRouterResourceBudget:
    """Exact logical storage and maximum per-call accounting.

    Byte counts assume logical one-byte booleans and standard four-byte
    float32/int32 scalars.  They exclude Python objects, compiler/runtime
    buffers, device alignment, executable caches, and checkpoint JSON syntax.
    Per-step touched counts are maxima, not FLOP or bandwidth measurements.
    """

    channel_count: int
    consumer_route_count: int
    max_counter_steps: int
    persistent_float32_scalars: int
    persistent_int32_scalars: int
    persistent_state_scalars: int
    persistent_state_bytes: int
    input_float32_scalars_per_step: int
    input_bool_scalars_per_step: int
    max_float32_state_scalars_touched_per_step: int
    max_int32_state_scalars_touched_per_step: int
    output_float32_scalars: int
    output_bool_scalars: int
    output_int32_scalars: int
    output_logical_bytes: int
    trainable_scalars: int
    rng_state_bytes: int
    replay_capacity: int

    def to_config(self) -> dict[str, int]:
        """Return the fixed JSON-compatible resource record."""

        return dataclasses.asdict(self)


@chex.dataclass(frozen=True)
class LearningValueRouterState:
    """Fixed-size causal calibration state and exact category counters."""

    step_count: Int[Array, ""]
    channel_valid_counts: Int[Array, " channels"]
    channel_unavailable_counts: Int[Array, " channels"]
    channel_invalid_counts: Int[Array, " channels"]
    channel_means: Float[Array, " channels"]
    channel_m2: Float[Array, " channels"]


@chex.dataclass(frozen=True)
class PaperDGActorRoute:
    """Actor route for advantage, surprisal, and paper-defined delight.

    ``values.delight`` is the DG actor-sample quantity ``advantage *
    action_surprisal`` (see the module docstring).
    """

    values: LearningValue
    normalized_values: LearningValue
    availability: LearningValueAvailability
    normalized_availability: LearningValueAvailability
    ready: Bool[Array, ""]
    normalization_ready: Bool[Array, ""]


@chex.dataclass(frozen=True)
class ExplorationLearningValueRoute:
    """Exploration route for epistemic, aleatoric, and progress signals."""

    values: LearningValue
    normalized_values: LearningValue
    availability: LearningValueAvailability
    normalized_availability: LearningValueAvailability
    ready: Bool[Array, ""]
    normalization_ready: Bool[Array, ""]


@chex.dataclass(frozen=True)
class ModelMemoryReplayLearningValueRoute:
    """Model-memory/replay route with no combined priority or generic score."""

    values: LearningValue
    normalized_values: LearningValue
    availability: LearningValueAvailability
    normalized_availability: LearningValueAvailability
    ready: Bool[Array, ""]
    normalization_ready: Bool[Array, ""]


@chex.dataclass(frozen=True)
class AdaptationChangeLearningValueRoute:
    """Adaptation route for progress and producer-declared change probability."""

    values: LearningValue
    normalized_values: LearningValue
    availability: LearningValueAvailability
    normalized_availability: LearningValueAvailability
    ready: Bool[Array, ""]
    normalization_ready: Bool[Array, ""]


@chex.dataclass(frozen=True)
class SafetyLearningValueRoute:
    """Independent raw/normalized safety-cost route.

    Missing or invalid non-safety inputs never veto a valid safety-cost value.
    """

    values: LearningValue
    normalized_values: LearningValue
    availability: LearningValueAvailability
    normalized_availability: LearningValueAvailability
    ready: Bool[Array, ""]
    normalization_ready: Bool[Array, ""]


@chex.dataclass(frozen=True)
class LiteralGradientJoyEvidenceRoute:
    """All-eight-channel evidence route for the candidate-update audit.

    ``ready`` requires all eight independently valid raw channels.  The
    legacy ``gradient_joy`` name matches the compatibility API in
    :mod:`alberta_framework.core.delight`; the route carries evidence only
    and performs no Kondo selection or candidate-gradient assessment.
    """

    values: LearningValue
    normalized_values: LearningValue
    availability: LearningValueAvailability
    normalized_availability: LearningValueAvailability
    ready: Bool[Array, ""]
    normalization_ready: Bool[Array, ""]


@chex.dataclass(frozen=True)
class LearningValueRouterDiagnostics:
    """Per-channel validation and pre-update calibration diagnostics."""

    declared_availability: LearningValueAvailability
    finite: LearningValueAvailability
    domain_valid: LearningValueAvailability
    accepted: LearningValueAvailability
    normalization_available: LearningValueAvailability
    pre_update_count: Int[Array, " channels"]
    pre_update_mean: LearningValue
    pre_update_m2: LearningValue
    pre_update_scale: LearningValue
    state_valid: Bool[Array, ""]
    counter_capacity_available: Bool[Array, ""]
    normalization_state_updated: Bool[Array, ""]
    paper_dg_delight_prerequisites_valid: Bool[Array, ""]
    paper_dg_delight_identity_valid: Bool[Array, ""]
    current_valid_channel_count: Int[Array, ""]
    current_unavailable_channel_count: Int[Array, ""]
    current_invalid_channel_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class LearningValueRouterResult:
    """Six consumer-specific routes and auditable causal diagnostics."""

    paper_dg_actor: PaperDGActorRoute
    exploration: ExplorationLearningValueRoute
    model_memory_replay: ModelMemoryReplayLearningValueRoute
    adaptation_change: AdaptationChangeLearningValueRoute
    safety: SafetyLearningValueRoute
    literal_gradient_joy_evidence: LiteralGradientJoyEvidenceRoute
    diagnostics: LearningValueRouterDiagnostics

    @property
    def candidate_update_audit_evidence(self) -> LiteralGradientJoyEvidenceRoute:
        """Canonical name for the historical audit-evidence route."""
        return self.literal_gradient_joy_evidence


@chex.dataclass(frozen=True)
class LearningValueRouterAccounting:
    """Dynamic counters paired with the immutable allocation budget."""

    step_count: Int[Array, ""]
    channel_valid_counts: Int[Array, " channels"]
    channel_unavailable_counts: Int[Array, " channels"]
    channel_invalid_counts: Int[Array, " channels"]
    state_valid: Bool[Array, ""]
    counter_capacity_available: Bool[Array, ""]


def _learning_value_vector(value: LearningValue, *, sequence_length: int | None) -> Array:
    """Validate and stack scalar or fixed-sequence floating channel leaves."""

    arrays: list[Array] = []
    expected_shape = () if sequence_length is None else (sequence_length,)
    for name in _CHANNEL_NAMES:
        try:
            array = jnp.asarray(getattr(value, name))
        except (TypeError, ValueError) as error:
            raise TypeError(f"learning_value.{name} must be a floating array") from error
        if array.shape != expected_shape:
            raise ValueError(
                f"learning_value.{name} must have shape {expected_shape}, got {array.shape}"
            )
        if not jnp.issubdtype(array.dtype, jnp.floating) or jnp.issubdtype(
            array.dtype, jnp.complexfloating
        ):
            raise TypeError(f"learning_value.{name} must have a real floating dtype")
        arrays.append(jnp.asarray(array, dtype=jnp.float32))
    axis = 0 if sequence_length is None else 1
    return jnp.stack(arrays, axis=axis)


def _availability_vector(
    availability: LearningValueAvailability,
    *,
    sequence_length: int | None,
) -> Array:
    """Validate and stack scalar or fixed-sequence boolean availability leaves."""

    arrays: list[Array] = []
    expected_shape = () if sequence_length is None else (sequence_length,)
    for name in _CHANNEL_NAMES:
        try:
            array = jnp.asarray(getattr(availability, name))
        except (TypeError, ValueError) as error:
            raise TypeError(f"availability.{name} must be a boolean array") from error
        if array.shape != expected_shape:
            raise ValueError(
                f"availability.{name} must have shape {expected_shape}, got {array.shape}"
            )
        if array.dtype != jnp.bool_:
            raise TypeError(f"availability.{name} must have boolean dtype")
        arrays.append(array)
    axis = 0 if sequence_length is None else 1
    return jnp.stack(arrays, axis=axis)


def _vector_to_learning_value(vector: Array) -> LearningValue:
    """Map an eight-scalar vector back to the typed ``LearningValue`` container."""

    return LearningValue(**dict(zip(_CHANNEL_NAMES, vector, strict=True)))


def _vector_to_availability(vector: Array) -> LearningValueAvailability:
    """Map an eight-boolean vector back to named availability fields."""

    return LearningValueAvailability(**dict(zip(_CHANNEL_NAMES, vector, strict=True)))


class LearningValueRouter:
    """Fixed-state router for independent typed learning-value consumers."""

    def __init__(self, config: LearningValueRouterConfig | None = None):
        self._config = config or LearningValueRouterConfig()

    @property
    def config(self) -> LearningValueRouterConfig:
        """Return the immutable router configuration."""

        return self._config

    def to_config(self) -> dict[str, object]:
        """Return the strict static router configuration."""

        return self._config.to_config()

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> LearningValueRouter:
        """Construct a router from an exact serialized config."""

        return cls(LearningValueRouterConfig.from_config(config))

    def channel_metadata(self) -> tuple[LearningValueChannelMetadata, ...]:
        """Return explicit producer, causal-object, unit, and domain records."""

        lower = self._config.channel_lower_bounds()
        upper = self._config.channel_upper_bounds()
        scale_floors = self._config.channel_scale_floors()
        semantic = (
            "actor_advantage",
            "selected_action_surprisal",
            "paper_specific_dg_actor_sample_delight",
            "epistemic_surprise",
            "aleatoric_uncertainty",
            "learning_progress",
            "change_probability",
            "safety_cost",
        )
        producer = (
            "behavior_actor_baseline",
            "selected_action_log_probability",
            "paper_specific_dg_actor_sample_producer",
            "world_model_ensemble_disagreement_or_information_gain_producer",
            "producer_declared_outcome_variance_estimator",
            "predict_before_update_loss_windows",
            "producer_declared_change_detector",
            "separately_trained_safety_cost_learner",
        )
        causal_object = (
            "value_or_actor_sample",
            "selected_actor_sample",
            "selected_actor_sample",
            "transition_or_model_hypothesis",
            "environment_outcome_distribution",
            "region_or_model_head",
            "stream_regime_hypothesis",
            "state_action_or_trajectory",
        )
        units = (
            "return_or_reward_units",
            "nats",
            "return_nats",
            "producer_declared_epistemic_surprise_units",
            "squared_target_units",
            "predictive_loss_units_per_window_or_version",
            "probability",
            "native_safety_cost_units",
        )
        domain = (
            "signed_bounded",
            "nonnegative_bounded",
            "signed_bounded_paper_specific_dg_actor_sample_quantity",
            "nonnegative_bounded",
            "nonnegative_bounded",
            "signed_bounded",
            "closed_unit_interval",
            "nonnegative_bounded",
        )
        return tuple(
            LearningValueChannelMetadata(
                index=index,
                field_name=name,
                semantic_name=semantic[index],
                producer=producer[index],
                causal_object=causal_object[index],
                units=units[index],
                domain=domain[index],
                lower_bound=lower[index],
                upper_bound=upper[index],
                normalization="causal_pre_update_welford_zscore",
                normalization_scale_floor=scale_floors[index],
                normalized_units="dimensionless_zscore",
                normalized_lower_bound=-float(self._config.normalization_clip),
                normalized_upper_bound=float(self._config.normalization_clip),
            )
            for index, name in enumerate(_CHANNEL_NAMES)
        )

    def resource_budget(self) -> LearningValueRouterResourceBudget:
        """Return exact fixed allocation and logical I/O counts."""

        persistent_float32 = 2 * _CHANNEL_COUNT
        persistent_int32 = 1 + 3 * _CHANNEL_COUNT
        output_float32 = 6 * 2 * _CHANNEL_COUNT + 3 * _CHANNEL_COUNT
        output_bool = 6 * (2 * _CHANNEL_COUNT + 2) + 5 * _CHANNEL_COUNT + 5
        output_int32 = _CHANNEL_COUNT + 3
        return LearningValueRouterResourceBudget(
            channel_count=_CHANNEL_COUNT,
            consumer_route_count=6,
            max_counter_steps=self._config.max_steps,
            persistent_float32_scalars=persistent_float32,
            persistent_int32_scalars=persistent_int32,
            persistent_state_scalars=persistent_float32 + persistent_int32,
            persistent_state_bytes=4 * (persistent_float32 + persistent_int32),
            input_float32_scalars_per_step=_CHANNEL_COUNT,
            input_bool_scalars_per_step=_CHANNEL_COUNT,
            max_float32_state_scalars_touched_per_step=2 * _CHANNEL_COUNT,
            max_int32_state_scalars_touched_per_step=1 + _CHANNEL_COUNT,
            output_float32_scalars=output_float32,
            output_bool_scalars=output_bool,
            output_int32_scalars=output_int32,
            output_logical_bytes=4 * (output_float32 + output_int32) + output_bool,
            trainable_scalars=0,
            rng_state_bytes=0,
            replay_capacity=0,
        )

    def init(self) -> LearningValueRouterState:
        """Return the exact zero fixed-size causal normalization state."""

        return LearningValueRouterState(
            step_count=jnp.asarray(0, dtype=jnp.int32),
            channel_valid_counts=jnp.zeros((_CHANNEL_COUNT,), dtype=jnp.int32),
            channel_unavailable_counts=jnp.zeros((_CHANNEL_COUNT,), dtype=jnp.int32),
            channel_invalid_counts=jnp.zeros((_CHANNEL_COUNT,), dtype=jnp.int32),
            channel_means=jnp.zeros((_CHANNEL_COUNT,), dtype=jnp.float32),
            channel_m2=jnp.zeros((_CHANNEL_COUNT,), dtype=jnp.float32),
        )

    @staticmethod
    def _validate_state_static_contract(state: LearningValueRouterState) -> None:
        """Raise on structural/dtype misuse while leaving values JIT-dynamic."""

        scalar = jnp.asarray(state.step_count)
        if scalar.shape != () or scalar.dtype != jnp.int32:
            raise ValueError("state.step_count must be a scalar int32")
        for name in (
            "channel_valid_counts",
            "channel_unavailable_counts",
            "channel_invalid_counts",
        ):
            value = jnp.asarray(getattr(state, name))
            if value.shape != (_CHANNEL_COUNT,) or value.dtype != jnp.int32:
                raise ValueError(f"state.{name} must be int32 with shape ({_CHANNEL_COUNT},)")
        for name in ("channel_means", "channel_m2"):
            value = jnp.asarray(getattr(state, name))
            if value.shape != (_CHANNEL_COUNT,) or value.dtype != jnp.float32:
                raise ValueError(f"state.{name} must be float32 with shape ({_CHANNEL_COUNT},)")

    def _state_is_valid(self, state: LearningValueRouterState) -> Array:
        """Return a JIT-safe verdict for all dynamic state invariants."""

        cfg = self._config
        lower = jnp.asarray(cfg.channel_lower_bounds(), dtype=jnp.float32)
        upper = jnp.asarray(cfg.channel_upper_bounds(), dtype=jnp.float32)
        spans = jnp.asarray(cfg.channel_spans(), dtype=jnp.float32)
        m2_bounds = (
            jnp.asarray(2.0 * float(cfg.max_steps), dtype=jnp.float32)
            * jnp.square(spans)
        )
        category_sum = (
            state.channel_valid_counts
            + state.channel_unavailable_counts
            + state.channel_invalid_counts
        )
        count_zero_or_one = state.channel_valid_counts <= 1
        initial_moment_valid = (~count_zero_or_one) | (state.channel_m2 == 0.0)
        zero_count_mean_valid = (state.channel_valid_counts != 0) | (
            state.channel_means == 0.0
        )
        positive_count_mean_valid = (state.channel_valid_counts == 0) | (
            (state.channel_means >= lower) & (state.channel_means <= upper)
        )
        return (
            (state.step_count >= 0)
            & (state.step_count <= cfg.max_steps)
            & jnp.all(state.channel_valid_counts >= 0)
            & jnp.all(state.channel_unavailable_counts >= 0)
            & jnp.all(state.channel_invalid_counts >= 0)
            & jnp.all(category_sum == state.step_count)
            & jnp.all(state.channel_valid_counts <= state.step_count)
            & jnp.all(state.channel_unavailable_counts <= state.step_count)
            & jnp.all(state.channel_invalid_counts <= state.step_count)
            & jnp.all(jnp.isfinite(state.channel_means))
            & jnp.all(jnp.isfinite(state.channel_m2))
            & jnp.all(state.channel_m2 >= 0.0)
            & jnp.all(state.channel_m2 <= m2_bounds)
            & jnp.all(initial_moment_valid)
            & jnp.all(zero_count_mean_valid)
            & jnp.all(positive_count_mean_valid)
        )

    def state_valid(self, state: LearningValueRouterState) -> Array:
        """Validate static structure and return the dynamic invariant verdict."""

        self._validate_state_static_contract(state)
        return self._state_is_valid(state)

    def validate_state(self, state: LearningValueRouterState) -> None:
        """Raise when a restored/checkpoint state violates any invariant."""

        self._validate_state_static_contract(state)
        if not bool(jax.device_get(self._state_is_valid(state))):
            raise ValueError("learning-value router state violates dynamic invariants")

    def accounting(self, state: LearningValueRouterState) -> LearningValueRouterAccounting:
        """Return exact category counters and current resource availability."""

        self._validate_state_static_contract(state)
        state_valid = self._state_is_valid(state)
        return LearningValueRouterAccounting(
            step_count=state.step_count,
            channel_valid_counts=state.channel_valid_counts,
            channel_unavailable_counts=state.channel_unavailable_counts,
            channel_invalid_counts=state.channel_invalid_counts,
            state_valid=state_valid,
            counter_capacity_available=state_valid
            & (state.step_count < self._config.max_steps),
        )

    @staticmethod
    def _masked_route(
        raw: Array,
        normalized: Array,
        accepted: Array,
        normalization_available: Array,
        mask_values: tuple[bool, ...],
    ) -> tuple[
        LearningValue,
        LearningValue,
        LearningValueAvailability,
        LearningValueAvailability,
        Array,
        Array,
    ]:
        """Mask one consumer route with exact-zero inactive/invalid channels."""

        mask = jnp.asarray(mask_values, dtype=jnp.bool_)
        active_availability = mask & accepted
        active_normalization = mask & normalization_available
        zero = jnp.zeros((_CHANNEL_COUNT,), dtype=jnp.float32)
        routed_raw = jnp.where(active_availability, raw, zero)
        routed_normalized = jnp.where(active_normalization, normalized, zero)
        ready = jnp.all(jnp.where(mask, active_availability, True))
        normalization_ready = jnp.all(jnp.where(mask, active_normalization, True))
        return (
            _vector_to_learning_value(routed_raw),
            _vector_to_learning_value(routed_normalized),
            _vector_to_availability(active_availability),
            _vector_to_availability(active_normalization),
            ready,
            normalization_ready,
        )

    def route(
        self,
        state: LearningValueRouterState,
        learning_value: LearningValue,
        declared_availability: LearningValueAvailability,
    ) -> tuple[LearningValueRouterState, LearningValueRouterResult]:
        """Validate, causally normalize, route, then update one stream event.

        Runtime invalidity is isolated per channel.  A declared but non-finite
        or out-of-domain value is zero/unavailable only in routes that consume
        that channel.  In particular, unrelated failures cannot suppress an
        independently valid safety cost.  A dynamically corrupt state is a
        global control-plane failure: every route fails closed and the state is
        returned unchanged.
        """

        self._validate_state_static_contract(state)
        raw_input = _learning_value_vector(learning_value, sequence_length=None)
        declared = _availability_vector(declared_availability, sequence_length=None)
        cfg = self._config
        lower = jnp.asarray(cfg.channel_lower_bounds(), dtype=jnp.float32)
        upper = jnp.asarray(cfg.channel_upper_bounds(), dtype=jnp.float32)
        scale_floors = jnp.asarray(cfg.channel_scale_floors(), dtype=jnp.float32)
        spans = jnp.asarray(cfg.channel_spans(), dtype=jnp.float32)
        m2_bounds = (
            jnp.asarray(2.0 * float(cfg.max_steps), dtype=jnp.float32)
            * jnp.square(spans)
        )

        state_valid = self._state_is_valid(state)
        counter_capacity_available = state_valid & (state.step_count < cfg.max_steps)
        finite = jnp.isfinite(raw_input)
        domain_valid = (raw_input >= lower) & (raw_input <= upper)
        paper_dg_delight_prerequisites_valid = (
            declared[0]
            & declared[1]
            & finite[0]
            & finite[1]
            & domain_valid[0]
            & domain_valid[1]
        )
        expected_paper_dg_delight = raw_input[0] * raw_input[1]
        expected_paper_dg_delight_bits = jax.lax.bitcast_convert_type(
            expected_paper_dg_delight,
            jnp.uint32,
        )
        supplied_paper_dg_delight_bits = jax.lax.bitcast_convert_type(
            raw_input[2],
            jnp.uint32,
        )
        paper_dg_delight_identity_valid = (
            paper_dg_delight_prerequisites_valid
            & finite[2]
            & jnp.isfinite(expected_paper_dg_delight)
            & (supplied_paper_dg_delight_bits == expected_paper_dg_delight_bits)
        )
        semantic_valid = domain_valid.at[2].set(
            domain_valid[2] & paper_dg_delight_identity_valid
        )
        safe_raw = jnp.nan_to_num(
            raw_input,
            nan=0.0,
            posinf=_FLOAT32_MAX,
            neginf=-_FLOAT32_MAX,
        )
        safe_raw = jnp.clip(safe_raw, lower, upper)

        safe_state_mean = jnp.where(
            state_valid,
            state.channel_means,
            jnp.zeros((_CHANNEL_COUNT,), dtype=jnp.float32),
        )
        safe_state_m2 = jnp.where(
            state_valid,
            state.channel_m2,
            jnp.zeros((_CHANNEL_COUNT,), dtype=jnp.float32),
        )
        safe_state_count = jnp.where(
            state_valid,
            state.channel_valid_counts,
            jnp.zeros((_CHANNEL_COUNT,), dtype=jnp.int32),
        )
        count_denominator = jnp.maximum(safe_state_count - 1, 1).astype(jnp.float32)
        pre_variance = jnp.where(
            safe_state_count >= 2,
            safe_state_m2 / count_denominator,
            jnp.zeros((_CHANNEL_COUNT,), dtype=jnp.float32),
        )
        pre_scale = jnp.maximum(jnp.sqrt(jnp.maximum(pre_variance, 0.0)), scale_floors)

        candidate_count = safe_state_count + jnp.asarray(1, dtype=jnp.int32)
        delta = safe_raw - safe_state_mean
        candidate_mean = safe_state_mean + delta / candidate_count.astype(jnp.float32)
        candidate_m2 = safe_state_m2 + delta * (safe_raw - candidate_mean)
        candidate_m2 = jnp.maximum(candidate_m2, 0.0)
        derived_update_valid = (
            jnp.isfinite(candidate_mean)
            & jnp.isfinite(candidate_m2)
            & (candidate_mean >= lower)
            & (candidate_mean <= upper)
            & (candidate_m2 <= m2_bounds)
        )
        accepted = (
            state_valid
            & declared
            & finite
            & semantic_valid
            & ((~counter_capacity_available) | derived_update_valid)
        )

        normalization_available = (
            accepted & (safe_state_count >= cfg.normalization_min_count)
        )
        normalized_unmasked = jnp.clip(
            (safe_raw - safe_state_mean) / pre_scale,
            -float(cfg.normalization_clip),
            float(cfg.normalization_clip),
        )
        normalized = jnp.where(
            normalization_available,
            jnp.nan_to_num(normalized_unmasked, nan=0.0, posinf=0.0, neginf=0.0),
            jnp.zeros((_CHANNEL_COUNT,), dtype=jnp.float32),
        )
        routed_raw = jnp.where(
            accepted,
            safe_raw,
            jnp.zeros((_CHANNEL_COUNT,), dtype=jnp.float32),
        )

        state_update_enabled = state_valid & counter_capacity_available
        current_unavailable = ~declared
        current_invalid = declared & ~accepted
        candidate_state = LearningValueRouterState(
            step_count=state.step_count + jnp.asarray(1, dtype=jnp.int32),
            channel_valid_counts=state.channel_valid_counts + accepted.astype(jnp.int32),
            channel_unavailable_counts=(
                state.channel_unavailable_counts + current_unavailable.astype(jnp.int32)
            ),
            channel_invalid_counts=(
                state.channel_invalid_counts + current_invalid.astype(jnp.int32)
            ),
            channel_means=jnp.where(accepted, candidate_mean, state.channel_means),
            channel_m2=jnp.where(accepted, candidate_m2, state.channel_m2),
        )
        next_state = jax.tree_util.tree_map(
            lambda candidate, original: jnp.where(
                state_update_enabled,
                candidate,
                original,
            ),
            candidate_state,
            state,
        )

        actor_payload = self._masked_route(
            routed_raw,
            normalized,
            accepted,
            normalization_available,
            _PAPER_DG_ACTOR_MASK,
        )
        exploration_payload = self._masked_route(
            routed_raw,
            normalized,
            accepted,
            normalization_available,
            _EXPLORATION_MASK,
        )
        memory_payload = self._masked_route(
            routed_raw,
            normalized,
            accepted,
            normalization_available,
            _MODEL_MEMORY_REPLAY_MASK,
        )
        adaptation_payload = self._masked_route(
            routed_raw,
            normalized,
            accepted,
            normalization_available,
            _ADAPTATION_CHANGE_MASK,
        )
        safety_payload = self._masked_route(
            routed_raw,
            normalized,
            accepted,
            normalization_available,
            _SAFETY_MASK,
        )
        joy_payload = self._masked_route(
            routed_raw,
            normalized,
            accepted,
            normalization_available,
            _LITERAL_GRADIENT_JOY_EVIDENCE_MASK,
        )

        def route_kwargs(
            payload: tuple[
                LearningValue,
                LearningValue,
                LearningValueAvailability,
                LearningValueAvailability,
                Array,
                Array,
            ],
        ) -> dict[str, object]:
            return {
                "values": payload[0],
                "normalized_values": payload[1],
                "availability": payload[2],
                "normalized_availability": payload[3],
                "ready": payload[4],
                "normalization_ready": payload[5],
            }

        result = LearningValueRouterResult(
            paper_dg_actor=PaperDGActorRoute(**route_kwargs(actor_payload)),
            exploration=ExplorationLearningValueRoute(
                **route_kwargs(exploration_payload)
            ),
            model_memory_replay=ModelMemoryReplayLearningValueRoute(
                **route_kwargs(memory_payload)
            ),
            adaptation_change=AdaptationChangeLearningValueRoute(
                **route_kwargs(adaptation_payload)
            ),
            safety=SafetyLearningValueRoute(**route_kwargs(safety_payload)),
            literal_gradient_joy_evidence=LiteralGradientJoyEvidenceRoute(
                **route_kwargs(joy_payload)
            ),
            diagnostics=LearningValueRouterDiagnostics(
                declared_availability=_vector_to_availability(declared),
                finite=_vector_to_availability(finite),
                domain_valid=_vector_to_availability(domain_valid),
                accepted=_vector_to_availability(accepted),
                normalization_available=_vector_to_availability(
                    normalization_available
                ),
                pre_update_count=safe_state_count,
                pre_update_mean=_vector_to_learning_value(
                    jnp.where(
                        state_valid,
                        safe_state_mean,
                        jnp.zeros((_CHANNEL_COUNT,), dtype=jnp.float32),
                    )
                ),
                pre_update_m2=_vector_to_learning_value(
                    jnp.where(
                        state_valid,
                        safe_state_m2,
                        jnp.zeros((_CHANNEL_COUNT,), dtype=jnp.float32),
                    )
                ),
                pre_update_scale=_vector_to_learning_value(
                    jnp.where(
                        state_valid,
                        pre_scale,
                        jnp.zeros((_CHANNEL_COUNT,), dtype=jnp.float32),
                    )
                ),
                state_valid=state_valid,
                counter_capacity_available=counter_capacity_available,
                normalization_state_updated=state_update_enabled,
                paper_dg_delight_prerequisites_valid=(
                    paper_dg_delight_prerequisites_valid
                ),
                paper_dg_delight_identity_valid=paper_dg_delight_identity_valid,
                current_valid_channel_count=jnp.sum(accepted.astype(jnp.int32)),
                current_unavailable_channel_count=jnp.sum(
                    current_unavailable.astype(jnp.int32)
                ),
                current_invalid_channel_count=jnp.sum(current_invalid.astype(jnp.int32)),
            ),
        )
        return (
            cast(
                LearningValueRouterState,
                jax.tree_util.tree_map(jax.lax.stop_gradient, next_state),
            ),
            cast(
                LearningValueRouterResult,
                jax.tree_util.tree_map(jax.lax.stop_gradient, result),
            ),
        )

    def scan(
        self,
        state: LearningValueRouterState,
        learning_values: LearningValue,
        declared_availability: LearningValueAvailability,
    ) -> tuple[LearningValueRouterState, LearningValueRouterResult]:
        """Route a fixed-length uninterrupted sequence with :func:`jax.lax.scan`."""

        self._validate_state_static_contract(state)
        first_shape = getattr(learning_values.advantage, "shape", None)
        if not isinstance(first_shape, tuple) or len(first_shape) != 1:
            raise ValueError("learning_values must contain rank-one channel sequences")
        sequence_length = first_shape[0]
        _learning_value_vector(learning_values, sequence_length=sequence_length)
        _availability_vector(
            declared_availability,
            sequence_length=sequence_length,
        )

        def step(
            carry: LearningValueRouterState,
            inputs: tuple[LearningValue, LearningValueAvailability],
        ) -> tuple[LearningValueRouterState, LearningValueRouterResult]:
            return self.route(carry, inputs[0], inputs[1])

        return jax.lax.scan(step, state, (learning_values, declared_availability))

    def checkpoint_payload(self, state: LearningValueRouterState) -> dict[str, object]:
        """Return a strict digest-bound JSON-compatible checkpoint payload."""

        self.validate_state(state)
        config_payload = self.to_config()
        state_payload: dict[str, object] = {
            "step_count": int(jax.device_get(state.step_count)),
            "channel_valid_counts": np.asarray(
                jax.device_get(state.channel_valid_counts), dtype=np.int32
            ).tolist(),
            "channel_unavailable_counts": np.asarray(
                jax.device_get(state.channel_unavailable_counts), dtype=np.int32
            ).tolist(),
            "channel_invalid_counts": np.asarray(
                jax.device_get(state.channel_invalid_counts), dtype=np.int32
            ).tolist(),
            "channel_means": np.asarray(
                jax.device_get(state.channel_means), dtype=np.float32
            ).tolist(),
            "channel_m2": np.asarray(
                jax.device_get(state.channel_m2), dtype=np.float32
            ).tolist(),
        }
        return {
            "schema": LEARNING_VALUE_ROUTER_CHECKPOINT_SCHEMA,
            "mechanism_status": MECHANISM_STATUS,
            "scientific_promotion_allowed": SCIENTIFIC_PROMOTION_ALLOWED,
            "router": config_payload,
            "config_digest": _payload_digest(config_payload),
            "channel_metadata": [
                metadata.to_config() for metadata in self.channel_metadata()
            ],
            "resource_budget": self.resource_budget().to_config(),
            "state": state_payload,
            "state_digest": _payload_digest(state_payload),
        }

    @staticmethod
    def _checkpoint_int32_vector(value: object, *, name: str) -> Array:
        """Parse an exact eight-element JSON int32 vector."""

        try:
            untyped = np.asarray(value, dtype=object)
        except ValueError as error:
            raise ValueError(f"{name} must be a rectangular vector") from error
        if untyped.shape != (_CHANNEL_COUNT,):
            raise ValueError(f"{name} must have shape ({_CHANNEL_COUNT},)")
        if any(type(item) is not int for item in untyped.flat):
            raise ValueError(f"{name} must contain JSON integers")
        if any(item < 0 or item > _INT32_MAX for item in untyped.flat):
            raise ValueError(f"{name} contains an out-of-range int32 counter")
        return jnp.asarray(value, dtype=jnp.int32)

    @staticmethod
    def _checkpoint_float32_vector(value: object, *, name: str) -> Array:
        """Parse an exact eight-element finite JSON float32 vector."""

        try:
            untyped = np.asarray(value, dtype=object)
        except ValueError as error:
            raise ValueError(f"{name} must be a rectangular vector") from error
        if untyped.shape != (_CHANNEL_COUNT,):
            raise ValueError(f"{name} must have shape ({_CHANNEL_COUNT},)")
        if any(type(item) is not float for item in untyped.flat):
            raise ValueError(f"{name} must contain JSON float values")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            parsed: npt.NDArray[np.float32] = np.asarray(value, dtype=np.float32)
        if not np.all(np.isfinite(parsed)):
            raise ValueError(f"{name} must contain finite float32 values")
        return jnp.asarray(parsed, dtype=jnp.float32)

    @classmethod
    def from_checkpoint_payload(
        cls,
        checkpoint: Mapping[str, object],
    ) -> tuple[LearningValueRouter, LearningValueRouterState]:
        """Strictly restore an exact v1 config/state checkpoint pair."""

        expected = {
            "schema",
            "mechanism_status",
            "scientific_promotion_allowed",
            "router",
            "config_digest",
            "channel_metadata",
            "resource_budget",
            "state",
            "state_digest",
        }
        if set(checkpoint) != expected:
            raise ValueError("checkpoint fields do not match the router v1 schema")
        schema = checkpoint.get("schema")
        if type(schema) is not str or schema != LEARNING_VALUE_ROUTER_CHECKPOINT_SCHEMA:
            raise ValueError("unexpected learning-value router checkpoint schema")
        mechanism_status = checkpoint.get("mechanism_status")
        if type(mechanism_status) is not str or mechanism_status != MECHANISM_STATUS:
            raise ValueError("learning-value router checkpoint must remain mechanism-only")
        if checkpoint.get("scientific_promotion_allowed") is not False:
            raise ValueError("learning-value router checkpoint cannot claim promotion")
        router_payload = checkpoint.get("router")
        state_payload = checkpoint.get("state")
        if not isinstance(router_payload, Mapping) or not isinstance(state_payload, Mapping):
            raise ValueError("checkpoint router and state must be mappings")
        config_digest = checkpoint.get("config_digest")
        if (
            type(config_digest) is not str
            or config_digest != _payload_digest(router_payload)
        ):
            raise ValueError("learning-value router checkpoint config digest mismatch")
        state_digest = checkpoint.get("state_digest")
        if type(state_digest) is not str or state_digest != _payload_digest(state_payload):
            raise ValueError("learning-value router checkpoint state digest mismatch")
        router = cls.from_config(router_payload)
        expected_metadata = [
            metadata.to_config() for metadata in router.channel_metadata()
        ]
        try:
            metadata_matches = _canonical_json_bytes(
                checkpoint.get("channel_metadata")
            ) == _canonical_json_bytes(expected_metadata)
        except (TypeError, ValueError):
            metadata_matches = False
        if not metadata_matches:
            raise ValueError("checkpoint channel metadata differs from the router contract")
        try:
            resource_matches = _canonical_json_bytes(
                checkpoint.get("resource_budget")
            ) == _canonical_json_bytes(router.resource_budget().to_config())
        except (TypeError, ValueError):
            resource_matches = False
        if not resource_matches:
            raise ValueError("checkpoint resource budget differs from the router contract")
        expected_state = {
            "step_count",
            "channel_valid_counts",
            "channel_unavailable_counts",
            "channel_invalid_counts",
            "channel_means",
            "channel_m2",
        }
        if set(state_payload) != expected_state:
            raise ValueError("checkpoint state fields do not match the router v1 state")
        step_count = state_payload.get("step_count")
        if (
            type(step_count) is not int
            or not 0 <= step_count <= _INT32_MAX
        ):
            raise ValueError("checkpoint state.step_count must be an int32 counter")
        state = LearningValueRouterState(
            step_count=jnp.asarray(step_count, dtype=jnp.int32),
            channel_valid_counts=cls._checkpoint_int32_vector(
                state_payload.get("channel_valid_counts"),
                name="checkpoint state.channel_valid_counts",
            ),
            channel_unavailable_counts=cls._checkpoint_int32_vector(
                state_payload.get("channel_unavailable_counts"),
                name="checkpoint state.channel_unavailable_counts",
            ),
            channel_invalid_counts=cls._checkpoint_int32_vector(
                state_payload.get("channel_invalid_counts"),
                name="checkpoint state.channel_invalid_counts",
            ),
            channel_means=cls._checkpoint_float32_vector(
                state_payload.get("channel_means"),
                name="checkpoint state.channel_means",
            ),
            channel_m2=cls._checkpoint_float32_vector(
                state_payload.get("channel_m2"),
                name="checkpoint state.channel_m2",
            ),
        )
        router.validate_state(state)
        return router, state


__all__ = [
    "AdaptationChangeLearningValueRoute",
    "ExplorationLearningValueRoute",
    "LEARNING_VALUE_ROUTER_CHECKPOINT_SCHEMA",
    "LEARNING_VALUE_ROUTER_CONFIG_SCHEMA",
    "LearningValueChannelMetadata",
    "LearningValueRouter",
    "LearningValueRouterAccounting",
    "LearningValueRouterConfig",
    "LearningValueRouterDiagnostics",
    "LearningValueRouterResourceBudget",
    "LearningValueRouterResult",
    "LearningValueRouterState",
    "LiteralGradientJoyEvidenceRoute",
    "MECHANISM_STATUS",
    "ModelMemoryReplayLearningValueRoute",
    "PaperDGActorRoute",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "SafetyLearningValueRoute",
]
