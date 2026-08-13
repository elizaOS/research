# mypy: disable-error-code="call-arg,name-defined"
r"""Causal state-construction contracts and small reference implementations.

The existing Alberta components expose several useful forms of history:
``HistoryFeatureExtractor`` and ``WorkingMemoryFeaturizer`` provide fixed
traces, while ``PrototypeAgent`` optionally uses a fixed-weight echo-state
GRU.  This module gives those design points a narrow common contract and adds
both a small diagonal recurrent reference and a conventional dense trainable
GRU.

The trainable reference is deliberately modest.  It is a diagonal bank of
write/hold units,

.. math::

    g_t &= \sigma(W_g u_t + b_g) \\
    c_t &= \tanh(W_c u_t + b_c) \\
    h_t &= (1-g_t) h_{t-1} + g_t c_t ,

where ``u_t`` contains the current observation and the preceding transition's
action, reward, and discount.  It carries an RTRL-style online eligibility
matrix (real-time recurrent learning; Williams & Zipser 1989).  With
parameters held fixed, that matrix is the exact unrolled ``dh_t / dtheta``;
after an online parameter update, carrying it forward is the
usual changing-parameter eligibility approximation.  A downstream prediction
or control head can therefore pass the gradient of its loss with respect to
the emitted state to :meth:`OnlineGatedStateBuilder.learn` without replay or a
backward sweep through stored experience.

`LearnableGRUStateBuilder` uses the same transaction and sensitivity contract
with dense learned update, reset, and candidate recurrence.  Its fixed-
parameter RTRL Jacobian is exact but larger than the diagonal reference.

These are learnable recurrent baselines, not general state-discovery results:

* only the smaller online-gated builder has no cross-unit recurrence;
* the caller must supply useful online auxiliary/control gradients;
* carried sensitivities are a fixed, potentially substantial memory cost; and
* there is no generate-and-test or feature-recycling mechanism here.

All builders are pure PyTree transformations, have fixed output/state budgets,
serialize their configuration, and round-trip through Alberta's generic Orbax
checkpoint utilities.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast, runtime_checkable

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core.checkpoints import (
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)
from alberta_framework.core.recurrent_trace_actor_critic import (
    RTUParameters,
    RTUSensitivities,
    RTUState,
    rtu_forward,
    rtu_step,
    rtu_taylor_step,
    zero_rtu_sensitivities,
    zero_rtu_state,
)
from alberta_framework.core.working_memory import (
    WORKING_MEMORY_LIFETIME_COUNTER_DELTA_NBYTES,
    WORKING_MEMORY_LIFETIME_COUNTER_NBYTES,
    WORKING_MEMORY_STATE_SCHEMA,
    WorkingMemoryConfig,
    WorkingMemoryFeaturizer,
    WorkingMemoryState,
    migrate_legacy_working_memory_state,
)

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1
_FLOAT32_MAX = 3.4028234663852886e38
_FLOAT32_TINY = 1.1754943508222875e-38

ONLINE_GATED_STATE_BUILDER_STATE_SCHEMA = "alberta.online-gated-state-builder-state.v3"
LEARNABLE_GRU_STATE_BUILDER_STATE_SCHEMA = "alberta.learnable-gru-state-builder-state.v1"
RECURRENT_TRACE_UNIT_STATE_BUILDER_STATE_SCHEMA = (
    "alberta.recurrent-trace-unit-state-builder-state.v1"
)
IDENTITY_STATE_BUILDER_STATE_SCHEMA = "alberta.identity-state-builder-state.v2"
FIXED_TRACE_STATE_BUILDER_STATE_SCHEMA = WORKING_MEMORY_STATE_SCHEMA
FIXED_STATE_BUILDER_STEP_COUNTER_NBYTES = WORKING_MEMORY_LIFETIME_COUNTER_NBYTES
FIXED_STATE_BUILDER_STEP_COUNTER_DELTA_NBYTES = (
    WORKING_MEMORY_LIFETIME_COUNTER_DELTA_NBYTES
)
ONLINE_GATED_STATE_BUILDER_STEP_COUNTER_NBYTES = 12
ONLINE_GATED_STATE_BUILDER_STEP_COUNTER_DELTA_NBYTES = 8
ONLINE_GATED_STATE_BUILDER_UPDATE_COUNTER_NBYTES = 12
ONLINE_GATED_STATE_BUILDER_UPDATE_COUNTER_DELTA_NBYTES = 8


def _saturating_int32_increment(value: Array) -> Array:
    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    counter = jnp.asarray(value, dtype=jnp.int32)
    return jnp.minimum(jnp.maximum(counter, 0), maximum - 1) + 1


def _checked_update_words_increment(
    words: Array,
) -> tuple[UInt[Array, " 2"], Bool[Array, ""]]:
    """Propose the next exact accepted-update identity without wrapping."""

    if getattr(words, "shape", None) != (2,):
        raise ValueError("state-builder update words must have shape (2,)")
    if getattr(words, "dtype", None) != jnp.dtype(jnp.uint32):
        raise TypeError("state-builder update words must have dtype uint32")
    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    capacity_available = ~jnp.all(words == maximum)
    one = jnp.asarray(1, dtype=jnp.uint32)
    low = words[1] + one
    carry = (low == jnp.asarray(0, dtype=jnp.uint32)).astype(jnp.uint32)
    proposed = jnp.stack((words[0] + carry, low))
    return (
        jnp.where(capacity_available, proposed, words).astype(jnp.uint32),
        capacity_available,
    )


def _update_lifetime_counter_valid(
    words: Array,
    telemetry: Array,
) -> Bool[Array, ""]:
    """Validate exact accepted-update identity against saturating telemetry."""

    if getattr(words, "shape", None) != (2,):
        raise ValueError("state-builder update words must have shape (2,)")
    if getattr(words, "dtype", None) != jnp.dtype(jnp.uint32):
        raise TypeError("state-builder update words must have dtype uint32")
    if getattr(telemetry, "shape", None) != ():
        raise ValueError("state-builder update_count must be scalar")
    if getattr(telemetry, "dtype", None) != jnp.dtype(jnp.int32):
        raise TypeError("state-builder update_count must have dtype int32")
    maximum_i32 = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    below_saturation = (words[0] == jnp.asarray(0, dtype=jnp.uint32)) & (
        words[1] < jnp.asarray(_INT32_MAX, dtype=jnp.uint32)
    )
    expected = words[1].astype(jnp.int32)
    return (telemetry >= 0) & jnp.where(
        below_saturation,
        telemetry == expected,
        telemetry == maximum_i32,
    )


def _checked_step_words_increment(
    words: Array,
) -> tuple[UInt[Array, " 2"], Bool[Array, ""]]:
    """Propose the next exact observation-transition identity without wrapping."""

    if getattr(words, "shape", None) != (2,):
        raise ValueError("state-builder step words must have shape (2,)")
    if getattr(words, "dtype", None) != jnp.dtype(jnp.uint32):
        raise TypeError("state-builder step words must have dtype uint32")
    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    capacity_available = ~jnp.all(words == maximum)
    one = jnp.asarray(1, dtype=jnp.uint32)
    low = words[1] + one
    carry = (low == jnp.asarray(0, dtype=jnp.uint32)).astype(jnp.uint32)
    proposed = jnp.stack((words[0] + carry, low))
    return (
        jnp.where(capacity_available, proposed, words).astype(jnp.uint32),
        capacity_available,
    )


def _step_lifetime_counter_valid(
    words: Array,
    telemetry: Array,
) -> Bool[Array, ""]:
    """Validate exact observation-transition identity against telemetry."""

    if getattr(words, "shape", None) != (2,):
        raise ValueError("state-builder step words must have shape (2,)")
    if getattr(words, "dtype", None) != jnp.dtype(jnp.uint32):
        raise TypeError("state-builder step words must have dtype uint32")
    if getattr(telemetry, "shape", None) != ():
        raise ValueError("state-builder step_count must be scalar")
    if getattr(telemetry, "dtype", None) != jnp.dtype(jnp.int32):
        raise TypeError("state-builder step_count must have dtype int32")
    maximum_i32 = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    below_saturation = (words[0] == jnp.asarray(0, dtype=jnp.uint32)) & (
        words[1] < jnp.asarray(_INT32_MAX, dtype=jnp.uint32)
    )
    expected = words[1].astype(jnp.int32)
    return (telemetry >= 0) & jnp.where(
        below_saturation,
        telemetry == expected,
        telemetry == maximum_i32,
    )


def _uint64_words_not_after(left: Array, right: Array) -> Bool[Array, ""]:
    """Compare big-endian ``uint32[2]`` words without narrowing to int64."""

    _static_array_contract(left, name="left clock words", shape=(2,), dtype=jnp.uint32)
    _static_array_contract(right, name="right clock words", shape=(2,), dtype=jnp.uint32)
    return (left[0] < right[0]) | ((left[0] == right[0]) & (left[1] <= right[1]))

StateT = TypeVar("StateT")

STATE_BUILDER_CHECKPOINT_SCHEMA = "alberta.state_builder.v4"


@dataclass(frozen=True)
class StateBuilderBudget:
    """Exact history-independent resource counts for a state builder.

    ``state_scalars`` counts every scalar carried in the builder state,
    including parameters and integer counters.  ``state_bytes`` assumes the
    implementations in this module: float32, int32, and uint32 arrays.
    It excludes transient compiler buffers and a downstream learning head.
    """

    output_scalars: int
    trainable_scalars: int
    state_scalars: int
    state_bytes: int

    def to_config(self) -> dict[str, int]:
        """Return a JSON-compatible budget description."""
        return asdict(self)


@chex.dataclass(frozen=True)
class StateBuilderLearningDiagnostics:
    """Diagnostics from one representation-learning update."""

    gradient_norm: Float[Array, ""]
    clipped_gradient_norm: Float[Array, ""]
    parameter_update_norm: Float[Array, ""]
    proposal_valid: Bool[Array, ""]
    source_matches: Bool[Array, ""]
    capacity_available: Bool[Array, ""]
    candidate_parameters_valid: Bool[Array, ""]
    applied: Bool[Array, ""]
    fixed_noop: Bool[Array, ""]
    valid: Bool[Array, ""]
    rejected: Bool[Array, ""]
    pre_update_words: UInt[Array, " 2"] = field(
        default_factory=lambda: jnp.zeros((2,), dtype=jnp.uint32)
    )
    post_update_words: UInt[Array, " 2"] = field(
        default_factory=lambda: jnp.zeros((2,), dtype=jnp.uint32)
    )
    lifetime_counter_valid: Bool[Array, ""] = field(
        default_factory=lambda: jnp.asarray(False, dtype=jnp.bool_)
    )
    lifetime_capacity_available: Bool[Array, ""] = field(
        default_factory=lambda: jnp.asarray(False, dtype=jnp.bool_)
    )
    update_applied: Bool[Array, ""] = field(
        default_factory=lambda: jnp.asarray(False, dtype=jnp.bool_)
    )


@chex.dataclass(frozen=True)
class StateBuilderLearningProposal:
    """Pure, source-bound proposal for one causal representation update.

    ``source_parameters`` and ``source_update_words`` bind the proposal to the
    exact parameter version that produced it. ``source_update_count`` is
    saturating compatibility telemetry only. ``candidate_parameter_update`` is
    already clipped, step-size-scaled, and ready for a later atomic commit.
    A downstream safety layer may replace that vector only through
    :func:`replace_state_builder_learning_proposal_update`, which preserves the
    source binding and recomputes all candidate diagnostics.
    """

    builder_fingerprint: UInt[Array, " 8"]
    source_parameters: Float[Array, " parameter_count"]
    source_update_count: Int[Array, ""]
    source_update_words: UInt[Array, " 2"]
    raw_parameter_gradient: Float[Array, " parameter_count"]
    clipped_parameter_gradient: Float[Array, " parameter_count"]
    candidate_parameter_update: Float[Array, " parameter_count"]
    gradient_norm: Float[Array, ""]
    clipped_gradient_norm: Float[Array, ""]
    parameter_update_norm: Float[Array, ""]
    source_state_valid: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    raw_parameter_gradient_valid: Bool[Array, ""]
    clipped_parameter_gradient_valid: Bool[Array, ""]
    candidate_parameter_update_valid: Bool[Array, ""]
    candidate_parameters_valid: Bool[Array, ""]
    capacity_available: Bool[Array, ""]
    candidate_update_transformed: Bool[Array, ""]
    candidate_update_approved: Bool[Array, ""]
    fixed_noop: Bool[Array, ""]
    valid: Bool[Array, ""]
    rejected: Bool[Array, ""]


@runtime_checkable
class StateBuilder(Protocol[StateT]):
    """Minimal causal state-construction contract.

    ``start`` consumes the first observation and optional preceding-transition
    values.  Thereafter, ``update`` consumes ``(current_observation,
    previous_action, previous_reward, previous_discount)`` and advances state
    exactly once.  The action and outcomes must be from the transition that
    produced the current observation; passing a current action before selecting
    it would leak future information.  Both methods return the representation
    associated with the supplied current observation.

    ``encode`` is pure: it pairs a supplied raw observation with the recurrent
    memory already present in ``state`` and never advances history.  For
    history-dependent builders it is valid for the most recently consumed
    observation (or as an explicitly counterfactual raw-observation query);
    callers must not mistake it for a second recurrent transition.

    ``learn`` accepts ``d(loss) / d(representation)`` from any downstream head.
    Builders without trainable representation parameters implement it as a
    no-op.  Separating ``update`` and ``learn`` prevents target information
    from entering the emitted state before its prediction is scored.
    """

    def init(self, key: Array) -> StateT:
        """Return a fresh builder state."""
        ...

    def start(
        self,
        state: StateT,
        raw_observation: Array,
        last_action: Array | int = -1,
        last_reward: Array | float = 0.0,
        last_discount: Array | float = 1.0,
    ) -> tuple[StateT, Array]:
        """Consume the first observation and emit its representation."""
        ...

    def reset_episode(self, state: StateT) -> StateT:
        """Clear episode-local memory without resetting lifetime state.

        Learned parameters, lifetime learning counters, and the monotonic
        observation-event counter are preserved.  The next :meth:`start` call
        consumes the first observation of the new episode and increments that
        counter exactly once.
        """
        ...

    def encode(self, state: StateT, raw_observation: Array) -> Array:
        """Emit a representation without advancing state."""
        ...

    def update(
        self,
        state: StateT,
        raw_observation: Array,
        previous_action: Array | int,
        previous_reward: Array | float,
        previous_discount: Array | float,
    ) -> tuple[StateT, Array]:
        """Consume one continuing transition and emit the new representation."""
        ...

    def learn(
        self,
        state: StateT,
        representation_gradient: Array,
    ) -> tuple[StateT, StateBuilderLearningDiagnostics]:
        """Apply an online representation gradient."""
        ...

    def propose_learning_update(
        self,
        source_state: StateT,
        dL_drepresentation: Array,  # noqa: N803 - mathematical dL notation
    ) -> StateBuilderLearningProposal:
        """Form a pure parameter update bound to ``source_state``."""
        ...

    def commit_learning_update(
        self,
        destination_state: StateT,
        proposal: StateBuilderLearningProposal,
    ) -> tuple[StateT, StateBuilderLearningDiagnostics]:
        """Atomically apply a still-current proposal to ``destination_state``."""
        ...

    def feature_dim(self) -> int:
        """Return the fixed representation dimension."""
        ...

    def observation_dim(self) -> int:
        """Return the raw observation dimension consumed by the builder."""
        ...

    def resource_budget(self) -> StateBuilderBudget:
        """Return exact persistent-state and trainable-parameter counts."""
        ...

    def state_valid(self, state: StateT) -> Bool[Array, ""]:
        """Validate the complete dynamic state contract."""
        ...

    def to_config(self) -> dict[str, Any]:
        """Return a JSON-compatible builder configuration."""
        ...


def _zero_learning_diagnostics(
    *,
    valid: Array | bool = True,
    proposal_valid: Array | bool | None = None,
    source_matches: Array | bool = True,
    capacity_available: Array | bool = True,
    candidate_parameters_valid: Array | bool = True,
    fixed_noop: Array | bool = True,
) -> StateBuilderLearningDiagnostics:
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    valid_array = jnp.asarray(valid, dtype=jnp.bool_)
    proposal_valid_array = (
        valid_array
        if proposal_valid is None
        else jnp.asarray(proposal_valid, dtype=jnp.bool_)
    )
    return StateBuilderLearningDiagnostics(
        gradient_norm=zero,
        clipped_gradient_norm=zero,
        parameter_update_norm=zero,
        proposal_valid=proposal_valid_array,
        source_matches=jnp.asarray(source_matches, dtype=jnp.bool_),
        capacity_available=jnp.asarray(capacity_available, dtype=jnp.bool_),
        candidate_parameters_valid=jnp.asarray(
            candidate_parameters_valid,
            dtype=jnp.bool_,
        ),
        applied=jnp.asarray(False),
        fixed_noop=jnp.asarray(fixed_noop, dtype=jnp.bool_),
        valid=valid_array,
        rejected=~valid_array,
        pre_update_words=jnp.zeros((2,), dtype=jnp.uint32),
        post_update_words=jnp.zeros((2,), dtype=jnp.uint32),
        lifetime_counter_valid=jnp.asarray(True, dtype=jnp.bool_),
        lifetime_capacity_available=jnp.asarray(
            capacity_available,
            dtype=jnp.bool_,
        ),
        update_applied=jnp.asarray(False, dtype=jnp.bool_),
    )


def _static_array_contract(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> None:
    """Validate shape and dtype using metadata available during JAX tracing."""
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must be an array with static shape and dtype metadata")
    actual_shape = tuple(value.shape)
    if actual_shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {actual_shape}")
    expected_dtype = jnp.dtype(dtype)
    actual_dtype = jnp.dtype(value.dtype)
    if actual_dtype != expected_dtype:
        raise TypeError(f"{name} must have dtype {expected_dtype}; got {actual_dtype}")


def _canonical_start_action(value: Array | int) -> Array | int:
    """Materialize an exact Python action scalar before a nested JIT call."""
    if type(value) is int:
        if value < -(2**31) or value > _INT32_MAX:
            raise ValueError("last_action must be in the signed int32 range")
        return jnp.asarray(value, dtype=jnp.int32)
    return value


def _canonical_start_outcome(value: Array | float, *, name: str) -> Array | float:
    """Materialize a finite, float32-range Python outcome before nested JIT."""
    if type(value) is float:
        if not math.isfinite(value) or abs(value) > _FLOAT32_MAX:
            raise ValueError(f"{name} must be finite and in the float32 range")
        return jnp.asarray(value, dtype=jnp.float32)
    return value


def _scale_safe_l2_norm(values: Array) -> Float[Array, ""]:
    """Return a finite, saturating float32 L2 norm without squaring overflow."""
    vector = jnp.asarray(values, dtype=jnp.float32).reshape((-1,))
    maximum = jnp.max(jnp.abs(vector), initial=jnp.asarray(0.0, dtype=jnp.float32))
    scaled = _divide_by_positive_scale(vector, maximum)
    scaled_norm = jnp.sqrt(jnp.sum(scaled * scaled))
    safe_scaled_norm = jnp.where(scaled_norm > 0.0, scaled_norm, 1.0)
    float32_max = jnp.asarray(_FLOAT32_MAX, dtype=jnp.float32)
    overflow = maximum > float32_max / safe_scaled_norm
    product = maximum * scaled_norm
    return jnp.where(maximum == 0.0, 0.0, jnp.where(overflow, float32_max, product))


def _divide_by_positive_scale(values: Array, scale: Array) -> Array:
    """Divide by a positive float32 scale without reciprocal underflow."""
    mantissa, exponent = jnp.frexp(scale)
    safe_mantissa = jnp.where(scale > 0.0, mantissa, 1.0)
    scaled = jnp.ldexp(values, -exponent) / safe_mantissa
    return jnp.where(scale > 0.0, scaled, 0.0)


def _scale_safe_clip_by_l2_norm(
    values: Array,
    clip: Array,
) -> tuple[Array, Float[Array, ""]]:
    """Clip a finite float32 vector without forming an overflowing norm."""
    vector = jnp.asarray(values, dtype=jnp.float32)
    flat = vector.reshape((-1,))
    maximum = jnp.max(jnp.abs(flat), initial=jnp.asarray(0.0, dtype=jnp.float32))
    scaled = _divide_by_positive_scale(vector, maximum)
    scaled_norm = jnp.sqrt(jnp.sum(scaled.reshape((-1,)) ** 2))
    safe_scaled_norm = jnp.where(scaled_norm > 0.0, scaled_norm, 1.0)
    within_limit = maximum <= clip / safe_scaled_norm
    clipped = scaled * (clip / safe_scaled_norm)
    result = jnp.where(within_limit, vector, clipped)
    return result, _scale_safe_l2_norm(vector)


def _builder_learning_fingerprint(config: dict[str, Any]) -> UInt[Array, " 8"]:
    """Return a compact, deterministic binding for one exact builder config."""
    canonical = json.dumps(
        config,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).digest()
    return jnp.asarray(
        [int.from_bytes(digest[offset : offset + 4], "big") for offset in range(0, 32, 4)],
        dtype=jnp.uint32,
    )


def _validate_learning_proposal_static_contract(
    proposal: StateBuilderLearningProposal,
    parameter_count: int,
) -> None:
    """Reject structural proposal corruption before eager or compiled execution."""
    if not isinstance(proposal, StateBuilderLearningProposal):
        raise TypeError("proposal must be a StateBuilderLearningProposal")
    _static_array_contract(
        proposal.builder_fingerprint,
        name="proposal.builder_fingerprint",
        shape=(8,),
        dtype=jnp.uint32,
    )
    for name in (
        "source_parameters",
        "raw_parameter_gradient",
        "clipped_parameter_gradient",
        "candidate_parameter_update",
    ):
        _static_array_contract(
            getattr(proposal, name),
            name=f"proposal.{name}",
            shape=(parameter_count,),
            dtype=jnp.float32,
        )
    _static_array_contract(
        proposal.source_update_count,
        name="proposal.source_update_count",
        shape=(),
        dtype=jnp.int32,
    )
    _static_array_contract(
        proposal.source_update_words,
        name="proposal.source_update_words",
        shape=(2,),
        dtype=jnp.uint32,
    )
    for name in (
        "gradient_norm",
        "clipped_gradient_norm",
        "parameter_update_norm",
    ):
        _static_array_contract(
            getattr(proposal, name),
            name=f"proposal.{name}",
            shape=(),
            dtype=jnp.float32,
        )
    for name in (
        "source_state_valid",
        "input_valid",
        "raw_parameter_gradient_valid",
        "clipped_parameter_gradient_valid",
        "candidate_parameter_update_valid",
        "candidate_parameters_valid",
        "capacity_available",
        "candidate_update_transformed",
        "candidate_update_approved",
        "fixed_noop",
        "valid",
        "rejected",
    ):
        _static_array_contract(
            getattr(proposal, name),
            name=f"proposal.{name}",
            shape=(),
            dtype=jnp.bool_,
        )


def _finite_or_max_norm(values: Array, valid: Array) -> Float[Array, ""]:
    """Return a finite norm, conservatively saturating invalid vectors."""
    safe_values = jnp.where(valid, values, 0.0)
    return jnp.where(
        valid,
        _scale_safe_l2_norm(safe_values),
        jnp.asarray(_FLOAT32_MAX, dtype=jnp.float32),
    )


def _float32_vectors_bitwise_equal(left: Array, right: Array) -> Bool[Array, ""]:
    """Compare exact float32 encodings, including signed zero payloads."""
    left_bits = jax.lax.bitcast_convert_type(left, jnp.uint32)
    right_bits = jax.lax.bitcast_convert_type(right, jnp.uint32)
    return jnp.all(left_bits == right_bits)


def _diagnostic_scalar_matches(actual: Array, expected: Array) -> Bool[Array, ""]:
    """Allow only backend-rounding tolerance in recomputed finite diagnostics."""
    exact = actual == expected
    rounded = (expected > 0.0) & jnp.isclose(actual, expected, rtol=1.0e-6, atol=0.0)
    return jnp.isfinite(actual) & (actual >= 0.0) & (exact | rounded)


def replace_state_builder_learning_proposal_update(
    proposal: StateBuilderLearningProposal,
    candidate_parameter_update: Array,
    approved: Array,
) -> StateBuilderLearningProposal:
    """Replace only a valid proposal's candidate update and refresh its checks.

    This is the supported boundary for a downstream update filter.  It keeps
    the source binding and raw/clipped gradients immutable, requires the exact
    float32 vector contract, requires an exact scalar-bool approval decision,
    and cannot turn an already-rejected proposal into an accepted one.  A
    false approval is an explicit veto even when the replacement vector is
    zero.
    """
    if not isinstance(proposal, StateBuilderLearningProposal):
        raise TypeError("proposal must be a StateBuilderLearningProposal")
    if not hasattr(proposal.source_parameters, "shape"):
        raise TypeError("proposal.source_parameters must expose static shape metadata")
    source_shape = tuple(proposal.source_parameters.shape)
    if len(source_shape) != 1:
        raise ValueError(
            "proposal.source_parameters must be a rank-one parameter vector; "
            f"got {source_shape}"
        )
    parameter_count = source_shape[0]
    _validate_learning_proposal_static_contract(proposal, parameter_count)
    _static_array_contract(
        candidate_parameter_update,
        name="candidate_parameter_update",
        shape=(parameter_count,),
        dtype=jnp.float32,
    )
    _static_array_contract(
        approved,
        name="approved",
        shape=(),
        dtype=jnp.bool_,
    )

    update = jnp.asarray(candidate_parameter_update, dtype=jnp.float32)
    approved_array = jnp.asarray(approved, dtype=jnp.bool_)
    update_valid = jnp.all(jnp.isfinite(update))
    candidate_parameters = proposal.source_parameters + update
    candidate_parameters_valid = jnp.all(jnp.isfinite(candidate_parameters))
    parameter_update_norm = _finite_or_max_norm(update, update_valid)
    valid = (
        proposal.valid
        & ~proposal.fixed_noop
        & approved_array
        & update_valid
        & candidate_parameters_valid
    )
    return cast(
        StateBuilderLearningProposal,
        replace(
            cast(Any, proposal),
            candidate_parameter_update=update,
            parameter_update_norm=parameter_update_norm,
            candidate_parameter_update_valid=update_valid,
            candidate_parameters_valid=candidate_parameters_valid,
            candidate_update_transformed=jnp.asarray(True),
            candidate_update_approved=approved_array,
            valid=valid,
            rejected=~valid,
        ),
    )


def _validate_observation_dim(observation_dim: int) -> None:
    if observation_dim < 1:
        raise ValueError("observation_dim must be positive")


def _action_features(action: Array | int, n_actions: int) -> Array:
    if n_actions == 0:
        return jnp.zeros((0,), dtype=jnp.float32)
    action_id = jnp.asarray(action, dtype=jnp.int32)
    return jax.nn.one_hot(action_id, n_actions, dtype=jnp.float32)


def _fixed_learning_proposal(
    builder_fingerprint: Array,
    source_state_valid: Array,
    input_valid: Array,
) -> StateBuilderLearningProposal:
    """Build an honest empty proposal for a non-trainable representation."""
    empty = jnp.zeros((0,), dtype=jnp.float32)
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    true = jnp.asarray(True)
    false = jnp.asarray(False)
    valid = source_state_valid & input_valid
    return StateBuilderLearningProposal(
        builder_fingerprint=jnp.asarray(builder_fingerprint, dtype=jnp.uint32),
        source_parameters=empty,
        source_update_count=jnp.asarray(0, dtype=jnp.int32),
        source_update_words=jnp.zeros((2,), dtype=jnp.uint32),
        raw_parameter_gradient=empty,
        clipped_parameter_gradient=empty,
        candidate_parameter_update=empty,
        gradient_norm=zero,
        clipped_gradient_norm=zero,
        parameter_update_norm=zero,
        source_state_valid=source_state_valid,
        input_valid=input_valid,
        raw_parameter_gradient_valid=true,
        clipped_parameter_gradient_valid=true,
        candidate_parameter_update_valid=true,
        candidate_parameters_valid=true,
        capacity_available=true,
        candidate_update_transformed=false,
        candidate_update_approved=true,
        fixed_noop=true,
        valid=valid,
        rejected=~valid,
    )


def _fixed_learning_proposal_integrity(proposal: StateBuilderLearningProposal) -> Array:
    """Validate all derived fields of an empty fixed-builder proposal."""
    expected_valid = proposal.source_state_valid & proposal.input_valid
    empty_vectors = (
        (proposal.source_parameters.size == 0)
        & (proposal.raw_parameter_gradient.size == 0)
        & (proposal.clipped_parameter_gradient.size == 0)
        & (proposal.candidate_parameter_update.size == 0)
    )
    return (
        empty_vectors
        & (proposal.source_update_count == 0)
        & jnp.all(proposal.source_update_words == jnp.asarray(0, dtype=jnp.uint32))
        & (proposal.gradient_norm == 0.0)
        & (proposal.clipped_gradient_norm == 0.0)
        & (proposal.parameter_update_norm == 0.0)
        & proposal.raw_parameter_gradient_valid
        & proposal.clipped_parameter_gradient_valid
        & proposal.candidate_parameter_update_valid
        & proposal.candidate_parameters_valid
        & proposal.capacity_available
        & ~proposal.candidate_update_transformed
        & proposal.candidate_update_approved
        & proposal.fixed_noop
        & (proposal.valid == expected_valid)
        & (proposal.rejected == ~expected_valid)
    )


def _fixed_learning_commit_diagnostics(
    proposal: StateBuilderLearningProposal,
    builder_fingerprint: Array,
    destination_state_valid: Array,
) -> StateBuilderLearningDiagnostics:
    """Return acceptance diagnostics for an atomic fixed-builder no-op."""
    source_matches = jnp.array_equal(
        proposal.builder_fingerprint,
        builder_fingerprint,
    )
    proposal_valid = _fixed_learning_proposal_integrity(proposal) & proposal.valid
    valid = destination_state_valid & source_matches & proposal_valid
    return _zero_learning_diagnostics(
        valid=valid,
        proposal_valid=proposal_valid,
        source_matches=source_matches,
        capacity_available=proposal.capacity_available,
        candidate_parameters_valid=proposal.candidate_parameters_valid,
        fixed_noop=True,
    )


@dataclass(frozen=True)
class IdentityStateBuilderConfig:
    """Configuration for the observation-only state baseline."""

    observation_dim: int

    def __post_init__(self) -> None:
        _validate_observation_dim(self.observation_dim)

    def to_config(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "type": "IdentityStateBuilder",
            "state_schema": IDENTITY_STATE_BUILDER_STATE_SCHEMA,
            "observation_dim": self.observation_dim,
        }

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> IdentityStateBuilderConfig:
        """Strictly reconstruct current-schema :meth:`to_config` output."""
        data = dict(payload)
        expected = {"type", "state_schema", "observation_dim"}
        if set(data) != expected:
            missing = sorted(expected - set(data))
            extra = sorted(set(data) - expected)
            raise ValueError(
                "identity state-builder config manifest is not exact; "
                f"missing={missing}, extra={extra}"
            )
        if data.pop("type") != "IdentityStateBuilder":
            raise ValueError("identity state-builder config type is unsupported")
        if data.pop("state_schema") != IDENTITY_STATE_BUILDER_STATE_SCHEMA:
            raise ValueError("identity state-builder state schema is unsupported")
        return cls(observation_dim=int(data.pop("observation_dim")))


@chex.dataclass(frozen=True)
class IdentityStateBuilderState:
    """Observation-only state with exact event identity and int32 telemetry."""

    step_count: Array
    step_words: UInt[Array, " 2"]


class IdentityStateBuilder:
    """Observation-only state builder and lower memory control."""

    def __init__(self, config: IdentityStateBuilderConfig):
        self._config = config
        self._learning_fingerprint = _builder_learning_fingerprint(config.to_config())

    @property
    def config(self) -> IdentityStateBuilderConfig:
        """Return the immutable configuration."""
        return self._config

    def init(self, key: Array) -> IdentityStateBuilderState:
        """Return a fresh state; ``key`` is accepted for protocol parity."""
        del key
        return IdentityStateBuilderState(
            step_count=jnp.asarray(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def feature_dim(self) -> int:
        """Return the raw observation dimension."""
        return self._config.observation_dim

    def observation_dim(self) -> int:
        """Return the raw observation dimension."""
        return self._config.observation_dim

    def resource_budget(self) -> StateBuilderBudget:
        """Return the exact lifetime-counter persistent budget."""
        return StateBuilderBudget(
            output_scalars=self.feature_dim(),
            trainable_scalars=0,
            state_scalars=3,
            state_bytes=FIXED_STATE_BUILDER_STEP_COUNTER_NBYTES,
        )

    def state_valid(self, state: IdentityStateBuilderState) -> Bool[Array, ""]:
        """Validate the exact identity-builder state contract."""
        self._validate_state_static_contract(state)
        return _step_lifetime_counter_valid(state.step_words, state.step_count)

    def to_config(self) -> dict[str, Any]:
        """Serialize the builder configuration."""
        return self._config.to_config()

    @functools.partial(jax.jit, static_argnums=(0,))
    def encode(
        self,
        state: IdentityStateBuilderState,
        raw_observation: Array,
    ) -> Float[Array, " observation_dim"]:
        """Return the raw observation without touching ``state``."""
        del state
        return jnp.asarray(raw_observation, dtype=jnp.float32).reshape(
            (self._config.observation_dim,)
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: IdentityStateBuilderState,
        raw_observation: Array,
        previous_action: Array | int,
        previous_reward: Array | float,
        previous_discount: Array | float,
    ) -> tuple[IdentityStateBuilderState, Float[Array, " observation_dim"]]:
        """Advance the counter and emit the raw observation."""
        del previous_action, previous_reward, previous_discount
        features = self.encode(state, raw_observation)
        proposed_words, capacity_available = _checked_step_words_increment(
            state.step_words
        )
        commit = (
            _step_lifetime_counter_valid(state.step_words, state.step_count)
            & capacity_available
            & jnp.all(jnp.isfinite(features))
        )
        next_state = IdentityStateBuilderState(
            step_count=jnp.where(
                commit,
                _saturating_int32_increment(state.step_count),
                state.step_count,
            ),
            step_words=jnp.where(commit, proposed_words, state.step_words).astype(
                jnp.uint32
            ),
        )
        return next_state, features

    def start(
        self,
        state: IdentityStateBuilderState,
        raw_observation: Array,
        last_action: Array | int = -1,
        last_reward: Array | float = 0.0,
        last_discount: Array | float = 1.0,
    ) -> tuple[IdentityStateBuilderState, Array]:
        """Consume the first observation."""
        return cast(
            tuple[IdentityStateBuilderState, Array],
            self.update(
                state,
                raw_observation,
                last_action,
                last_reward,
                last_discount,
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset_episode(
        self,
        state: IdentityStateBuilderState,
    ) -> IdentityStateBuilderState:
        """Preserve the lifetime event counter; there is no recurrent memory."""
        return state

    def _validate_state_static_contract(self, state: IdentityStateBuilderState) -> None:
        if not isinstance(state, IdentityStateBuilderState):
            raise TypeError("state must be an IdentityStateBuilderState")
        _static_array_contract(
            state.step_count,
            name="state.step_count",
            shape=(),
            dtype=jnp.int32,
        )
        _static_array_contract(
            state.step_words,
            name="state.step_words",
            shape=(2,),
            dtype=jnp.uint32,
        )

    def _validate_gradient_static_contract(self, representation_gradient: Array) -> None:
        _static_array_contract(
            representation_gradient,
            name="representation_gradient",
            shape=(self.feature_dim(),),
            dtype=jnp.float32,
        )

    def propose_learning_update(
        self,
        source_state: IdentityStateBuilderState,
        dL_drepresentation: Array,  # noqa: N803 - mathematical dL notation
    ) -> StateBuilderLearningProposal:
        """Return an honest source-checked no-op proposal."""
        self._validate_state_static_contract(source_state)
        self._validate_gradient_static_contract(dL_drepresentation)
        return cast(
            StateBuilderLearningProposal,
            self._propose_learning_update_jit(source_state, dL_drepresentation),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _propose_learning_update_jit(
        self,
        source_state: IdentityStateBuilderState,
        representation_gradient: Array,
    ) -> StateBuilderLearningProposal:
        source_state_valid = _step_lifetime_counter_valid(
            source_state.step_words,
            source_state.step_count,
        )
        input_valid = jnp.all(jnp.isfinite(representation_gradient))
        return _fixed_learning_proposal(
            self._learning_fingerprint,
            source_state_valid,
            input_valid,
        )

    def commit_learning_update(
        self,
        destination_state: IdentityStateBuilderState,
        proposal: StateBuilderLearningProposal,
    ) -> tuple[IdentityStateBuilderState, StateBuilderLearningDiagnostics]:
        """Validate and accept a fixed-representation no-op atomically."""
        self._validate_state_static_contract(destination_state)
        _validate_learning_proposal_static_contract(proposal, 0)
        return cast(
            tuple[IdentityStateBuilderState, StateBuilderLearningDiagnostics],
            self._commit_learning_update_jit(destination_state, proposal),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _commit_learning_update_jit(
        self,
        destination_state: IdentityStateBuilderState,
        proposal: StateBuilderLearningProposal,
    ) -> tuple[IdentityStateBuilderState, StateBuilderLearningDiagnostics]:
        diagnostics = _fixed_learning_commit_diagnostics(
            proposal,
            self._learning_fingerprint,
            _step_lifetime_counter_valid(
                destination_state.step_words,
                destination_state.step_count,
            ),
        )
        return destination_state, diagnostics

    def learn(
        self,
        state: IdentityStateBuilderState,
        representation_gradient: Array,
    ) -> tuple[IdentityStateBuilderState, StateBuilderLearningDiagnostics]:
        """Propose and commit the fixed representation's no-op update."""
        proposal = self.propose_learning_update(state, representation_gradient)
        return self.commit_learning_update(state, proposal)


@dataclass(frozen=True)
class FixedTraceStateBuilderConfig:
    """Configuration for a fixed observation/action/outcome trace bank."""

    observation_dim: int
    n_actions: int = 0
    observation_decay_rates: tuple[float, ...] = (0.5, 0.9, 0.99)
    action_decay_rates: tuple[float, ...] = (0.5, 0.9)
    outcome_decay_rates: tuple[float, ...] = (0.5, 0.9)
    include_raw_observation: bool = True

    def __post_init__(self) -> None:
        _validate_observation_dim(self.observation_dim)
        if self.n_actions < 0:
            raise ValueError("n_actions must be non-negative")
        for name, rates in (
            ("observation_decay_rates", self.observation_decay_rates),
            ("action_decay_rates", self.action_decay_rates),
            ("outcome_decay_rates", self.outcome_decay_rates),
        ):
            if any(not math.isfinite(rate) or rate < 0.0 or rate >= 1.0 for rate in rates):
                raise ValueError(f"{name} must contain finite values in [0, 1)")
        if (
            not self.include_raw_observation
            and not self.observation_decay_rates
            and not self.action_decay_rates
            and not self.outcome_decay_rates
        ):
            raise ValueError("configuration must emit at least one feature")

    def to_config(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "type": "FixedTraceStateBuilder",
            "state_schema": FIXED_TRACE_STATE_BUILDER_STATE_SCHEMA,
            "observation_dim": self.observation_dim,
            "n_actions": self.n_actions,
            "observation_decay_rates": list(self.observation_decay_rates),
            "action_decay_rates": list(self.action_decay_rates),
            "outcome_decay_rates": list(self.outcome_decay_rates),
            "include_raw_observation": self.include_raw_observation,
        }

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> FixedTraceStateBuilderConfig:
        """Strictly reconstruct current-schema :meth:`to_config` output."""
        data = dict(payload)
        expected = {
            "type",
            "state_schema",
            "observation_dim",
            "n_actions",
            "observation_decay_rates",
            "action_decay_rates",
            "outcome_decay_rates",
            "include_raw_observation",
        }
        if set(data) != expected:
            missing = sorted(expected - set(data))
            extra = sorted(set(data) - expected)
            raise ValueError(
                "fixed-trace state-builder config manifest is not exact; "
                f"missing={missing}, extra={extra}"
            )
        if data.pop("type") != "FixedTraceStateBuilder":
            raise ValueError("fixed-trace state-builder config type is unsupported")
        if data.pop("state_schema") != FIXED_TRACE_STATE_BUILDER_STATE_SCHEMA:
            raise ValueError("fixed-trace state-builder state schema is unsupported")
        return cls(
            observation_dim=int(data.pop("observation_dim")),
            n_actions=int(data.pop("n_actions")),
            observation_decay_rates=tuple(data.pop("observation_decay_rates")),
            action_decay_rates=tuple(data.pop("action_decay_rates")),
            outcome_decay_rates=tuple(data.pop("outcome_decay_rates")),
            include_raw_observation=bool(data.pop("include_raw_observation")),
        )


class FixedTraceStateBuilder:
    """Fixed multi-timescale trace baseline using ``WorkingMemoryFeaturizer``.

    Reward and discount are treated as a two-channel outcome vector.  Returned
    traces are post-update, so :meth:`encode` reproduces the current recurrent
    state without applying a transition twice.
    """

    def __init__(self, config: FixedTraceStateBuilderConfig):
        self._config = config
        self._learning_fingerprint = _builder_learning_fingerprint(config.to_config())
        self._memory = WorkingMemoryFeaturizer(
            WorkingMemoryConfig(
                observation_dim=config.observation_dim,
                action_dim=config.n_actions,
                reward_dim=2,
                observation_decay_rates=config.observation_decay_rates,
                action_decay_rates=config.action_decay_rates,
                reward_decay_rates=config.outcome_decay_rates,
                include_current_observation=config.include_raw_observation,
                include_current_action=False,
                include_current_reward=False,
                include_traces=True,
                include_innovations=False,
            )
        )

    @property
    def config(self) -> FixedTraceStateBuilderConfig:
        """Return the immutable configuration."""
        return self._config

    def init(self, key: Array) -> WorkingMemoryState:
        """Return an all-zero trace state; ``key`` is unused."""
        del key
        return self._memory.init()

    def feature_dim(self) -> int:
        """Return the fixed trace representation dimension."""
        return int(self._memory.feature_dim())

    def observation_dim(self) -> int:
        """Return the raw observation dimension."""
        return self._config.observation_dim

    def resource_budget(self) -> StateBuilderBudget:
        """Return exact trace-bank and counter storage."""
        cfg = self._config
        trace_scalars = (
            cfg.observation_dim * len(cfg.observation_decay_rates)
            + cfg.n_actions * len(cfg.action_decay_rates)
            + 2 * len(cfg.outcome_decay_rates)
        )
        state_scalars = trace_scalars + 6  # exact counter words + telemetry + gates
        return StateBuilderBudget(
            output_scalars=self.feature_dim(),
            trainable_scalars=0,
            state_scalars=state_scalars,
            state_bytes=4 * state_scalars,
        )

    def state_valid(self, state: WorkingMemoryState) -> Bool[Array, ""]:
        """Validate every trace, gate, and lifetime counter."""
        self._validate_state_static_contract(state)
        return self._state_is_valid(state)

    def to_config(self) -> dict[str, Any]:
        """Serialize the builder configuration."""
        return self._config.to_config()

    @functools.partial(jax.jit, static_argnums=(0,))
    def encode(
        self,
        state: WorkingMemoryState,
        raw_observation: Array,
    ) -> Float[Array, " feature_dim"]:
        """Combine a raw observation with the already-current trace state."""
        observation = jnp.asarray(raw_observation, dtype=jnp.float32).reshape(
            (self._config.observation_dim,)
        )
        return cast(
            Float[Array, " feature_dim"],
            self._memory.features(
                state,
                observation,
                self._memory.zero_action(),
                self._memory.zero_reward(),
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: WorkingMemoryState,
        raw_observation: Array,
        previous_action: Array | int,
        previous_reward: Array | float,
        previous_discount: Array | float,
    ) -> tuple[WorkingMemoryState, Float[Array, " feature_dim"]]:
        """Advance all trace banks and emit the post-update memory state."""
        observation = jnp.asarray(raw_observation, dtype=jnp.float32).reshape(
            (self._config.observation_dim,)
        )
        action_vector = _action_features(previous_action, self._config.n_actions)
        outcomes = jnp.stack(
            [
                jnp.asarray(previous_reward, dtype=jnp.float32),
                jnp.asarray(previous_discount, dtype=jnp.float32),
            ]
        )
        next_state = self._memory.update(
            state,
            observation,
            action_vector,
            outcomes,
        )
        return next_state, self.encode(next_state, observation)

    def start(
        self,
        state: WorkingMemoryState,
        raw_observation: Array,
        last_action: Array | int = -1,
        last_reward: Array | float = 0.0,
        last_discount: Array | float = 1.0,
    ) -> tuple[WorkingMemoryState, Array]:
        """Consume the first observation and seed the trace bank."""
        return cast(
            tuple[WorkingMemoryState, Array],
            self.update(
                state,
                raw_observation,
                last_action,
                last_reward,
                last_discount,
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset_episode(self, state: WorkingMemoryState) -> WorkingMemoryState:
        """Clear trace banks and gates while preserving the lifetime counter."""
        reset = self._memory.reset()
        return WorkingMemoryState(
            observation_traces=reset.observation_traces,
            action_traces=reset.action_traces,
            reward_traces=reset.reward_traces,
            step_count=state.step_count,
            step_words=state.step_words,
            last_gate=reset.last_gate,
        )

    def _validate_state_static_contract(self, state: WorkingMemoryState) -> None:
        if not isinstance(state, WorkingMemoryState):
            raise TypeError("state must be a WorkingMemoryState")
        cfg = self._config
        for name, shape in (
            (
                "observation_traces",
                (len(cfg.observation_decay_rates), cfg.observation_dim),
            ),
            ("action_traces", (len(cfg.action_decay_rates), cfg.n_actions)),
            ("reward_traces", (len(cfg.outcome_decay_rates), 2)),
            ("last_gate", (3,)),
        ):
            _static_array_contract(
                getattr(state, name),
                name=f"state.{name}",
                shape=shape,
                dtype=jnp.float32,
            )
        _static_array_contract(
            state.step_count,
            name="state.step_count",
            shape=(),
            dtype=jnp.int32,
        )
        _static_array_contract(
            state.step_words,
            name="state.step_words",
            shape=(2,),
            dtype=jnp.uint32,
        )

    def _validate_gradient_static_contract(self, representation_gradient: Array) -> None:
        _static_array_contract(
            representation_gradient,
            name="representation_gradient",
            shape=(self.feature_dim(),),
            dtype=jnp.float32,
        )

    @staticmethod
    def _state_is_valid(state: WorkingMemoryState) -> Array:
        return (
            jnp.all(jnp.isfinite(state.observation_traces))
            & jnp.all(jnp.isfinite(state.action_traces))
            & jnp.all(jnp.isfinite(state.reward_traces))
            & jnp.all(jnp.isfinite(state.last_gate))
            & jnp.all(state.last_gate >= 0.0)
            & jnp.all(state.last_gate <= 1.0)
            & _step_lifetime_counter_valid(state.step_words, state.step_count)
        )

    def propose_learning_update(
        self,
        source_state: WorkingMemoryState,
        dL_drepresentation: Array,  # noqa: N803 - mathematical dL notation
    ) -> StateBuilderLearningProposal:
        """Return an honest source-checked no-op proposal."""
        self._validate_state_static_contract(source_state)
        self._validate_gradient_static_contract(dL_drepresentation)
        return cast(
            StateBuilderLearningProposal,
            self._propose_learning_update_jit(source_state, dL_drepresentation),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _propose_learning_update_jit(
        self,
        source_state: WorkingMemoryState,
        representation_gradient: Array,
    ) -> StateBuilderLearningProposal:
        return _fixed_learning_proposal(
            self._learning_fingerprint,
            self._state_is_valid(source_state),
            jnp.all(jnp.isfinite(representation_gradient)),
        )

    def commit_learning_update(
        self,
        destination_state: WorkingMemoryState,
        proposal: StateBuilderLearningProposal,
    ) -> tuple[WorkingMemoryState, StateBuilderLearningDiagnostics]:
        """Validate and accept a fixed-trace no-op atomically."""
        self._validate_state_static_contract(destination_state)
        _validate_learning_proposal_static_contract(proposal, 0)
        return cast(
            tuple[WorkingMemoryState, StateBuilderLearningDiagnostics],
            self._commit_learning_update_jit(destination_state, proposal),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _commit_learning_update_jit(
        self,
        destination_state: WorkingMemoryState,
        proposal: StateBuilderLearningProposal,
    ) -> tuple[WorkingMemoryState, StateBuilderLearningDiagnostics]:
        diagnostics = _fixed_learning_commit_diagnostics(
            proposal,
            self._learning_fingerprint,
            self._state_is_valid(destination_state),
        )
        return destination_state, diagnostics

    def learn(
        self,
        state: WorkingMemoryState,
        representation_gradient: Array,
    ) -> tuple[WorkingMemoryState, StateBuilderLearningDiagnostics]:
        """Propose and commit the fixed trace representation's no-op update."""
        proposal = self.propose_learning_update(state, representation_gradient)
        return self.commit_learning_update(state, proposal)


@dataclass(frozen=True)
class OnlineGatedStateBuilderConfig:
    """Configuration for the online learnable write/hold state builder."""

    observation_dim: int
    n_actions: int = 0
    hidden_dim: int = 8
    step_size: float = 0.01
    gradient_clip: float = 10.0
    initial_gate_bias: float = -2.0
    initialization_scale: float = 0.2
    include_raw_observation: bool = True

    def __post_init__(self) -> None:
        _validate_observation_dim(self.observation_dim)
        if self.n_actions < 0:
            raise ValueError("n_actions must be non-negative")
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if not math.isfinite(self.step_size) or self.step_size <= 0.0:
            raise ValueError("step_size must be finite and positive")
        if not math.isfinite(self.gradient_clip) or self.gradient_clip <= 0.0:
            raise ValueError("gradient_clip must be finite and positive")
        if not math.isfinite(self.initial_gate_bias):
            raise ValueError("initial_gate_bias must be finite")
        if not math.isfinite(self.initialization_scale) or self.initialization_scale <= 0.0:
            raise ValueError("initialization_scale must be finite and positive")

    def event_dim(self) -> int:
        """Return observation + one-hot action + reward + discount width."""
        return self.observation_dim + self.n_actions + 2

    def parameter_count(self) -> int:
        """Return write/candidate weights and biases."""
        return 2 * self.hidden_dim * (self.event_dim() + 1)

    def feature_dim(self) -> int:
        """Return raw-observation plus hidden-state width."""
        return self.hidden_dim + (self.observation_dim if self.include_raw_observation else 0)

    def to_config(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        payload = asdict(self)
        payload["type"] = "OnlineGatedStateBuilder"
        payload["state_schema"] = ONLINE_GATED_STATE_BUILDER_STATE_SCHEMA
        return payload

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> OnlineGatedStateBuilderConfig:
        """Strictly reconstruct current-schema :meth:`to_config` output."""
        data = dict(payload)
        expected = {
            "type",
            "state_schema",
            "observation_dim",
            "n_actions",
            "hidden_dim",
            "step_size",
            "gradient_clip",
            "initial_gate_bias",
            "initialization_scale",
            "include_raw_observation",
        }
        if set(data) != expected:
            missing = sorted(expected - set(data))
            extra = sorted(set(data) - expected)
            raise ValueError(
                "online-gated state-builder config manifest is not exact; "
                f"missing={missing}, extra={extra}"
            )
        if data.pop("type") != "OnlineGatedStateBuilder":
            raise ValueError("online-gated state-builder config type is unsupported")
        if data.pop("state_schema") != ONLINE_GATED_STATE_BUILDER_STATE_SCHEMA:
            raise ValueError("online-gated state-builder state schema is unsupported")
        return cls(**data)


@chex.dataclass(frozen=True)
class OnlineGatedStateBuilderState:
    """Parameters, recurrent state, and RTRL-style eligibility sensitivities."""

    parameters: Float[Array, " parameter_count"]
    hidden: Float[Array, " hidden_dim"]
    parameter_sensitivity: Float[Array, "hidden_dim parameter_count"]
    step_count: Array
    step_words: UInt[Array, " 2"]
    update_count: Array
    update_words: UInt[Array, " 2"]
    last_gradient_norm: Float[Array, ""]


@chex.dataclass(frozen=True)
class OnlineGatedStateBuilderTransitionResult:
    """One observation transition with an exact, non-wrapping identity."""

    state: OnlineGatedStateBuilderState
    representation: Float[Array, " feature_dim"]
    pre_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    state_valid: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    candidate_representation_valid: Bool[Array, ""]
    step_counter_valid: Bool[Array, ""]
    step_capacity_available: Bool[Array, ""]
    transition_applied: Bool[Array, ""]


class OnlineGatedStateBuilder:
    """Learnable gated recurrent state with an online sensitivity trace.

    The recurrence is advanced by :meth:`start`/:meth:`update`.  Learning is a
    separate call because a prediction must be scored before its target-derived
    gradient is allowed to modify the representation.  ``learn`` updates only
    recurrent parameters; a downstream head owns and checkpoints its own
    parameters. Sensitivities are exact for an unroll with fixed parameters.
    Carrying them across :meth:`learn` calls is an online eligibility
    approximation; it is not the derivative of the stored hidden state with
    respect to the newly updated parameter vector.
    """

    def __init__(self, config: OnlineGatedStateBuilderConfig):
        self._config = config
        self._learning_fingerprint = _builder_learning_fingerprint(config.to_config())

    @property
    def config(self) -> OnlineGatedStateBuilderConfig:
        """Return the immutable configuration."""
        return self._config

    def feature_dim(self) -> int:
        """Return the fixed emitted representation width."""
        return self._config.feature_dim()

    def observation_dim(self) -> int:
        """Return the raw observation dimension."""
        return self._config.observation_dim

    def resource_budget(self) -> StateBuilderBudget:
        """Return exact persistent state, including recurrent sensitivities."""
        parameter_count = self._config.parameter_count()
        state_scalars = (
            parameter_count
            + self._config.hidden_dim
            + self._config.hidden_dim * parameter_count
            + 7
        )
        return StateBuilderBudget(
            output_scalars=self.feature_dim(),
            trainable_scalars=parameter_count,
            state_scalars=state_scalars,
            state_bytes=4 * state_scalars,
        )

    def state_valid(
        self,
        state: OnlineGatedStateBuilderState,
    ) -> Bool[Array, ""]:
        """Validate parameters, recurrence, sensitivities, norms, and counters."""
        self._validate_state_static_contract(state)
        return self._state_is_valid(state)

    def to_config(self) -> dict[str, Any]:
        """Serialize the builder configuration."""
        return self._config.to_config()

    def init(self, key: Array) -> OnlineGatedStateBuilderState:
        """Initialize trainable parameters, hidden state, and sensitivities."""
        gate_key, candidate_key = jr.split(key)
        cfg = self._config
        event_dim = cfg.event_dim()
        hidden_dim = cfg.hidden_dim
        scale = jnp.asarray(cfg.initialization_scale, dtype=jnp.float32)
        gate_weights = scale * jr.normal(
            gate_key,
            (hidden_dim, event_dim),
            dtype=jnp.float32,
        )
        gate_bias = jnp.full(
            (hidden_dim,),
            cfg.initial_gate_bias,
            dtype=jnp.float32,
        )
        candidate_weights = scale * jr.normal(
            candidate_key,
            (hidden_dim, event_dim),
            dtype=jnp.float32,
        )
        candidate_bias = jnp.zeros((hidden_dim,), dtype=jnp.float32)
        parameters = jnp.concatenate(
            [
                gate_weights.reshape(-1),
                gate_bias,
                candidate_weights.reshape(-1),
                candidate_bias,
            ]
        )
        return OnlineGatedStateBuilderState(
            parameters=parameters,
            hidden=jnp.zeros((hidden_dim,), dtype=jnp.float32),
            parameter_sensitivity=jnp.zeros(
                (hidden_dim, cfg.parameter_count()),
                dtype=jnp.float32,
            ),
            step_count=jnp.asarray(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
            update_count=jnp.asarray(0, dtype=jnp.int32),
            update_words=jnp.zeros((2,), dtype=jnp.uint32),
            last_gradient_norm=jnp.asarray(0.0, dtype=jnp.float32),
        )

    def _unpack_parameters(
        self,
        parameters: Array,
    ) -> tuple[Array, Array, Array, Array]:
        cfg = self._config
        matrix_size = cfg.hidden_dim * cfg.event_dim()
        offset = 0
        gate_weights = parameters[offset : offset + matrix_size].reshape(
            (cfg.hidden_dim, cfg.event_dim())
        )
        offset += matrix_size
        gate_bias = parameters[offset : offset + cfg.hidden_dim]
        offset += cfg.hidden_dim
        candidate_weights = parameters[offset : offset + matrix_size].reshape(
            (cfg.hidden_dim, cfg.event_dim())
        )
        offset += matrix_size
        candidate_bias = parameters[offset : offset + cfg.hidden_dim]
        return gate_weights, gate_bias, candidate_weights, candidate_bias

    def _transition(self, parameters: Array, hidden: Array, event: Array) -> Array:
        gate_weights, gate_bias, candidate_weights, candidate_bias = self._unpack_parameters(
            parameters
        )
        gate = jax.nn.sigmoid(gate_weights @ event + gate_bias)
        candidate = jnp.tanh(candidate_weights @ event + candidate_bias)
        return hidden + gate * (candidate - hidden)

    def _next_parameter_sensitivity(
        self,
        parameters: Array,
        hidden: Array,
        event: Array,
        previous_sensitivity: Array,
    ) -> Array:
        """Propagate the diagonal recurrence's online parameter sensitivity."""

        direct_sensitivity = jax.jacfwd(self._transition, argnums=0)(
            parameters,
            hidden,
            event,
        )
        gate_weights, gate_bias, _candidate_weights, _candidate_bias = (
            self._unpack_parameters(parameters)
        )
        gate = jax.nn.sigmoid(gate_weights @ event + gate_bias)
        return cast(
            Array,
            direct_sensitivity + (1.0 - gate)[:, None] * previous_sensitivity,
        )

    def _event(
        self,
        raw_observation: Array,
        previous_action: Array | int,
        previous_reward: Array | float,
        previous_discount: Array | float,
    ) -> Array:
        observation = jnp.asarray(raw_observation, dtype=jnp.float32).reshape(
            (self._config.observation_dim,)
        )
        return jnp.concatenate(
            [
                observation,
                _action_features(previous_action, self._config.n_actions),
                jnp.atleast_1d(jnp.asarray(previous_reward, dtype=jnp.float32)),
                jnp.atleast_1d(jnp.asarray(previous_discount, dtype=jnp.float32)),
            ]
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def encode(
        self,
        state: OnlineGatedStateBuilderState,
        raw_observation: Array,
    ) -> Float[Array, " feature_dim"]:
        """Pair raw input with the current hidden state without advancing it."""
        observation = jnp.asarray(raw_observation, dtype=jnp.float32).reshape(
            (self._config.observation_dim,)
        )
        if self._config.include_raw_observation:
            return jnp.concatenate([observation, state.hidden])
        return state.hidden

    @functools.partial(jax.jit, static_argnums=(0,))
    def update_with_status(
        self,
        state: OnlineGatedStateBuilderState,
        raw_observation: Array,
        previous_action: Array | int,
        previous_reward: Array | float,
        previous_discount: Array | float,
    ) -> OnlineGatedStateBuilderTransitionResult:
        """Atomically advance recurrence and report the exact step boundary."""
        # Preserve the documented Python-scalar compatibility surface when JIT
        # is disabled (as it is in several integration tests). Under tracing,
        # array/tracer inputs pass through unchanged and retain strict dtype
        # validation below.
        previous_action = _canonical_start_action(previous_action)
        previous_reward = _canonical_start_outcome(
            previous_reward,
            name="previous_reward",
        )
        previous_discount = _canonical_start_outcome(
            previous_discount,
            name="previous_discount",
        )
        self._validate_state_static_contract(state)
        proposed_step_words, step_capacity_available = (
            _checked_step_words_increment(state.step_words)
        )
        step_counter_valid = _step_lifetime_counter_valid(
            state.step_words,
            state.step_count,
        )
        _static_array_contract(
            raw_observation,
            name="raw_observation",
            shape=(self._config.observation_dim,),
            dtype=jnp.float32,
        )
        _static_array_contract(
            previous_action,
            name="previous_action",
            shape=(),
            dtype=jnp.int32,
        )
        _static_array_contract(
            previous_reward,
            name="previous_reward",
            shape=(),
            dtype=jnp.float32,
        )
        _static_array_contract(
            previous_discount,
            name="previous_discount",
            shape=(),
            dtype=jnp.float32,
        )
        observation = jnp.asarray(raw_observation)
        action = jnp.asarray(previous_action)
        reward = jnp.asarray(previous_reward)
        discount = jnp.asarray(previous_discount)
        action_valid = jnp.asarray(True, dtype=jnp.bool_)
        if self._config.n_actions > 0:
            action_valid = (action >= -1) & (action < self._config.n_actions)
        input_valid = (
            jnp.all(jnp.isfinite(observation))
            & action_valid
            & jnp.isfinite(reward)
            & jnp.isfinite(discount)
            & (discount >= 0.0)
            & (discount <= 1.0)
        )
        state_valid = self._state_is_valid(state)
        safe_observation = jnp.where(jnp.isfinite(observation), observation, 0.0)
        safe_action = jnp.where(action_valid, action, jnp.asarray(0, dtype=jnp.int32))
        safe_reward = jnp.where(jnp.isfinite(reward), reward, 0.0)
        safe_discount = jnp.where(jnp.isfinite(discount), discount, 0.0)
        safe_parameters = jnp.where(jnp.isfinite(state.parameters), state.parameters, 0.0)
        safe_hidden = jnp.where(jnp.isfinite(state.hidden), state.hidden, 0.0)
        safe_sensitivity = jnp.where(
            jnp.isfinite(state.parameter_sensitivity),
            state.parameter_sensitivity,
            0.0,
        )
        event = self._event(
            safe_observation,
            safe_action,
            safe_reward,
            safe_discount,
        )
        new_hidden = self._transition(safe_parameters, safe_hidden, event)
        new_sensitivity = self._next_parameter_sensitivity(
            safe_parameters,
            safe_hidden,
            event,
            safe_sensitivity,
        )
        candidate_state = OnlineGatedStateBuilderState(
            parameters=state.parameters,
            hidden=new_hidden,
            parameter_sensitivity=new_sensitivity,
            step_count=_saturating_int32_increment(state.step_count),
            step_words=proposed_step_words,
            update_count=state.update_count,
            update_words=state.update_words,
            last_gradient_norm=state.last_gradient_norm,
        )
        candidate_state_valid = self._state_is_valid(candidate_state)
        candidate_representation = self.encode(candidate_state, safe_observation)
        candidate_representation_valid = jnp.all(jnp.isfinite(candidate_representation))
        transition_applied = (
            state_valid
            & input_valid
            & step_counter_valid
            & step_capacity_available
            & candidate_state_valid
            & candidate_representation_valid
        )

        next_state = cast(
            OnlineGatedStateBuilderState,
            jax.lax.cond(
                transition_applied,
                lambda _: candidate_state,
                lambda _: state,
                operand=None,
            ),
        )
        committed_representation = self.encode(state, safe_observation)
        finite_committed_representation = jnp.where(
            jnp.isfinite(committed_representation),
            committed_representation,
            0.0,
        )
        representation = jax.lax.cond(
            transition_applied,
            lambda _: candidate_representation,
            lambda _: finite_committed_representation,
            operand=None,
        )
        return OnlineGatedStateBuilderTransitionResult(
            state=next_state,
            representation=representation,
            pre_step_words=state.step_words,
            post_step_words=next_state.step_words,
            state_valid=state_valid,
            input_valid=input_valid,
            candidate_state_valid=candidate_state_valid,
            candidate_representation_valid=candidate_representation_valid,
            step_counter_valid=step_counter_valid,
            step_capacity_available=step_capacity_available,
            transition_applied=transition_applied,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: OnlineGatedStateBuilderState,
        raw_observation: Array,
        previous_action: Array | int,
        previous_reward: Array | float,
        previous_discount: Array | float,
    ) -> tuple[OnlineGatedStateBuilderState, Float[Array, " feature_dim"]]:
        """Advance recurrence while preserving the tuple-return compatibility API."""
        result = self.update_with_status(
            state,
            raw_observation,
            previous_action,
            previous_reward,
            previous_discount,
        )
        return result.state, result.representation

    def start(
        self,
        state: OnlineGatedStateBuilderState,
        raw_observation: Array,
        last_action: Array | int = -1,
        last_reward: Array | float = 0.0,
        last_discount: Array | float = 1.0,
    ) -> tuple[OnlineGatedStateBuilderState, Array]:
        """Consume the initial observation."""
        action = _canonical_start_action(last_action)
        reward = _canonical_start_outcome(last_reward, name="last_reward")
        discount = _canonical_start_outcome(last_discount, name="last_discount")
        return cast(
            tuple[OnlineGatedStateBuilderState, Array],
            self.update(
                state,
                raw_observation,
                action,
                reward,
                discount,
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset_episode(
        self,
        state: OnlineGatedStateBuilderState,
    ) -> OnlineGatedStateBuilderState:
        """Apply an unauthenticated episode reset without advancing either clock.

        This administrative boundary clears episode-local recurrence and
        sensitivities. It is deliberately not an observation transition or an
        accepted learning update, so callers must record episode resets
        separately when reconstructing a complete event ledger.
        """
        return OnlineGatedStateBuilderState(
            parameters=state.parameters,
            hidden=jnp.zeros_like(state.hidden),
            parameter_sensitivity=jnp.zeros_like(state.parameter_sensitivity),
            step_count=state.step_count,
            step_words=state.step_words,
            update_count=state.update_count,
            update_words=state.update_words,
            last_gradient_norm=state.last_gradient_norm,
        )

    def _validate_state_static_contract(
        self,
        state: OnlineGatedStateBuilderState,
    ) -> None:
        """Reject structural state corruption before eager or compiled execution."""
        if not isinstance(state, OnlineGatedStateBuilderState):
            raise TypeError("state must be an OnlineGatedStateBuilderState")
        cfg = self._config
        parameter_count = cfg.parameter_count()
        _static_array_contract(
            state.parameters,
            name="state.parameters",
            shape=(parameter_count,),
            dtype=jnp.float32,
        )
        _static_array_contract(
            state.hidden,
            name="state.hidden",
            shape=(cfg.hidden_dim,),
            dtype=jnp.float32,
        )
        _static_array_contract(
            state.parameter_sensitivity,
            name="state.parameter_sensitivity",
            shape=(cfg.hidden_dim, parameter_count),
            dtype=jnp.float32,
        )
        _static_array_contract(
            state.step_count,
            name="state.step_count",
            shape=(),
            dtype=jnp.int32,
        )
        _static_array_contract(
            state.step_words,
            name="state.step_words",
            shape=(2,),
            dtype=jnp.uint32,
        )
        _static_array_contract(
            state.update_count,
            name="state.update_count",
            shape=(),
            dtype=jnp.int32,
        )
        _static_array_contract(
            state.update_words,
            name="state.update_words",
            shape=(2,),
            dtype=jnp.uint32,
        )
        _static_array_contract(
            state.last_gradient_norm,
            name="state.last_gradient_norm",
            shape=(),
            dtype=jnp.float32,
        )

    def _validate_gradient_static_contract(self, representation_gradient: Array) -> None:
        _static_array_contract(
            representation_gradient,
            name="representation_gradient",
            shape=(self.feature_dim(),),
            dtype=jnp.float32,
        )

    @staticmethod
    def _state_is_valid(state: OnlineGatedStateBuilderState) -> Array:
        return (
            jnp.all(jnp.isfinite(state.parameters))
            & jnp.all(jnp.isfinite(state.hidden))
            & jnp.all(jnp.isfinite(state.parameter_sensitivity))
            & jnp.isfinite(state.last_gradient_norm)
            & (state.last_gradient_norm >= 0.0)
            & _step_lifetime_counter_valid(
                state.step_words,
                state.step_count,
            )
            & _update_lifetime_counter_valid(
                state.update_words,
                state.update_count,
            )
        )

    def propose_learning_update(
        self,
        source_state: OnlineGatedStateBuilderState,
        dL_drepresentation: Array,  # noqa: N803 - mathematical dL notation
    ) -> StateBuilderLearningProposal:
        """Form a pure clipped update from the source state's sensitivity."""
        self._validate_state_static_contract(source_state)
        self._validate_gradient_static_contract(dL_drepresentation)
        return cast(
            StateBuilderLearningProposal,
            self._propose_learning_update_jit(source_state, dL_drepresentation),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _propose_learning_update_jit(
        self,
        source_state: OnlineGatedStateBuilderState,
        representation_gradient: Array,
    ) -> StateBuilderLearningProposal:
        """Numerically validate and form one source-bound online proposal."""
        gradient = jnp.asarray(representation_gradient, dtype=jnp.float32)
        hidden_gradient = gradient[-self._config.hidden_dim :]

        source_state_valid = self._state_is_valid(source_state)
        input_valid = jnp.all(jnp.isfinite(gradient))
        raw_parameter_gradient = source_state.parameter_sensitivity.T @ hidden_gradient
        raw_parameter_gradient_valid = jnp.all(jnp.isfinite(raw_parameter_gradient))
        safe_parameter_gradient = jnp.where(
            input_valid & raw_parameter_gradient_valid,
            raw_parameter_gradient,
            0.0,
        )

        clip = jnp.asarray(self._config.gradient_clip, dtype=jnp.float32)
        clipped_parameter_gradient, safe_gradient_norm = _scale_safe_clip_by_l2_norm(
            safe_parameter_gradient,
            clip,
        )
        gradient_norm = jnp.where(
            input_valid & raw_parameter_gradient_valid,
            safe_gradient_norm,
            jnp.asarray(_FLOAT32_MAX, dtype=jnp.float32),
        )
        clipped_parameter_gradient_valid = jnp.all(
            jnp.isfinite(clipped_parameter_gradient)
        )
        candidate_parameter_update = (
            -jnp.asarray(self._config.step_size, dtype=jnp.float32)
            * clipped_parameter_gradient
        )
        candidate_parameter_update_valid = jnp.all(
            jnp.isfinite(candidate_parameter_update)
        )
        candidate_parameters = source_state.parameters + candidate_parameter_update
        candidate_parameters_valid = jnp.all(jnp.isfinite(candidate_parameters))
        _, capacity_available = _checked_update_words_increment(
            source_state.update_words
        )
        valid = (
            source_state_valid
            & input_valid
            & raw_parameter_gradient_valid
            & clipped_parameter_gradient_valid
            & candidate_parameter_update_valid
            & candidate_parameters_valid
            & capacity_available
        )
        return StateBuilderLearningProposal(
            builder_fingerprint=self._learning_fingerprint,
            source_parameters=source_state.parameters,
            source_update_count=source_state.update_count,
            source_update_words=source_state.update_words,
            raw_parameter_gradient=raw_parameter_gradient,
            clipped_parameter_gradient=clipped_parameter_gradient,
            candidate_parameter_update=candidate_parameter_update,
            gradient_norm=gradient_norm,
            clipped_gradient_norm=_finite_or_max_norm(
                clipped_parameter_gradient,
                clipped_parameter_gradient_valid,
            ),
            parameter_update_norm=_finite_or_max_norm(
                candidate_parameter_update,
                candidate_parameter_update_valid,
            ),
            source_state_valid=source_state_valid,
            input_valid=input_valid,
            raw_parameter_gradient_valid=raw_parameter_gradient_valid,
            clipped_parameter_gradient_valid=clipped_parameter_gradient_valid,
            candidate_parameter_update_valid=candidate_parameter_update_valid,
            candidate_parameters_valid=candidate_parameters_valid,
            capacity_available=capacity_available,
            candidate_update_transformed=jnp.asarray(False),
            candidate_update_approved=jnp.asarray(True),
            fixed_noop=jnp.asarray(False),
            valid=valid,
            rejected=~valid,
        )

    def _proposal_has_integrity(self, proposal: StateBuilderLearningProposal) -> Array:
        """Recompute proposal diagnostics and update formation at commit time."""
        raw_valid = jnp.all(jnp.isfinite(proposal.raw_parameter_gradient))
        safe_raw_gradient = jnp.where(
            proposal.input_valid & raw_valid,
            proposal.raw_parameter_gradient,
            0.0,
        )
        expected_clipped, safe_gradient_norm = _scale_safe_clip_by_l2_norm(
            safe_raw_gradient,
            jnp.asarray(self._config.gradient_clip, dtype=jnp.float32),
        )
        expected_gradient_norm = jnp.where(
            proposal.input_valid & raw_valid,
            safe_gradient_norm,
            jnp.asarray(_FLOAT32_MAX, dtype=jnp.float32),
        )
        clipped_valid = jnp.all(jnp.isfinite(proposal.clipped_parameter_gradient))
        expected_formed_update = (
            -jnp.asarray(self._config.step_size, dtype=jnp.float32)
            * proposal.clipped_parameter_gradient
        )
        formed_update_matches = proposal.candidate_update_transformed | (
            _float32_vectors_bitwise_equal(
                proposal.candidate_parameter_update,
                expected_formed_update,
            )
        )
        update_valid = jnp.all(jnp.isfinite(proposal.candidate_parameter_update))
        candidate_parameters = (
            proposal.source_parameters + proposal.candidate_parameter_update
        )
        candidate_parameters_valid = jnp.all(jnp.isfinite(candidate_parameters))
        _, capacity_available = _checked_update_words_increment(
            proposal.source_update_words
        )
        lifetime_counter_valid = _update_lifetime_counter_valid(
            proposal.source_update_words,
            proposal.source_update_count,
        )
        expected_valid = (
            proposal.source_state_valid
            & lifetime_counter_valid
            & proposal.input_valid
            & raw_valid
            & clipped_valid
            & update_valid
            & candidate_parameters_valid
            & capacity_available
            & proposal.candidate_update_approved
        )
        return (
            ~proposal.fixed_noop
            & (proposal.candidate_update_transformed | proposal.candidate_update_approved)
            & (~proposal.source_state_valid | jnp.all(jnp.isfinite(proposal.source_parameters)))
            & (proposal.raw_parameter_gradient_valid == raw_valid)
            & _float32_vectors_bitwise_equal(
                proposal.clipped_parameter_gradient,
                expected_clipped,
            )
            & (proposal.clipped_parameter_gradient_valid == clipped_valid)
            & formed_update_matches
            & (proposal.candidate_parameter_update_valid == update_valid)
            & (proposal.candidate_parameters_valid == candidate_parameters_valid)
            & (proposal.capacity_available == capacity_available)
            & _diagnostic_scalar_matches(
                proposal.gradient_norm,
                expected_gradient_norm,
            )
            & _diagnostic_scalar_matches(
                proposal.clipped_gradient_norm,
                _finite_or_max_norm(proposal.clipped_parameter_gradient, clipped_valid),
            )
            & _diagnostic_scalar_matches(
                proposal.parameter_update_norm,
                _finite_or_max_norm(proposal.candidate_parameter_update, update_valid),
            )
            & (proposal.valid == expected_valid)
            & (proposal.rejected == ~expected_valid)
        )

    def commit_learning_update(
        self,
        destination_state: OnlineGatedStateBuilderState,
        proposal: StateBuilderLearningProposal,
    ) -> tuple[OnlineGatedStateBuilderState, StateBuilderLearningDiagnostics]:
        """Atomically commit a valid proposal if its parameter source is current."""
        self._validate_state_static_contract(destination_state)
        _validate_learning_proposal_static_contract(
            proposal,
            self._config.parameter_count(),
        )
        return cast(
            tuple[OnlineGatedStateBuilderState, StateBuilderLearningDiagnostics],
            self._commit_learning_update_jit(destination_state, proposal),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _commit_learning_update_jit(
        self,
        destination_state: OnlineGatedStateBuilderState,
        proposal: StateBuilderLearningProposal,
    ) -> tuple[OnlineGatedStateBuilderState, StateBuilderLearningDiagnostics]:
        proposal_integrity = self._proposal_has_integrity(proposal)
        fingerprint_matches = jnp.array_equal(
            proposal.builder_fingerprint,
            self._learning_fingerprint,
        )
        parameters_match = _float32_vectors_bitwise_equal(
            destination_state.parameters,
            proposal.source_parameters,
        )
        count_matches = destination_state.update_count == proposal.source_update_count
        words_match = jnp.array_equal(
            destination_state.update_words,
            proposal.source_update_words,
        )
        source_matches = (
            fingerprint_matches & parameters_match & count_matches & words_match
        )
        destination_state_valid = self._state_is_valid(destination_state)
        proposed_update_words, destination_capacity_available = (
            _checked_update_words_increment(destination_state.update_words)
        )
        lifetime_counter_valid = _update_lifetime_counter_valid(
            destination_state.update_words,
            destination_state.update_count,
        )
        capacity_available = destination_capacity_available & proposal.capacity_available
        candidate_parameters = (
            destination_state.parameters + proposal.candidate_parameter_update
        )
        candidate_parameters_valid = jnp.all(jnp.isfinite(candidate_parameters))
        proposal_valid = proposal_integrity & proposal.valid
        applied = (
            destination_state_valid
            & source_matches
            & proposal_valid
            & capacity_available
            & candidate_parameters_valid
        )
        candidate_state = OnlineGatedStateBuilderState(
            parameters=candidate_parameters,
            hidden=destination_state.hidden,
            parameter_sensitivity=destination_state.parameter_sensitivity,
            step_count=destination_state.step_count,
            step_words=destination_state.step_words,
            update_count=_saturating_int32_increment(destination_state.update_count),
            update_words=proposed_update_words,
            last_gradient_norm=proposal.gradient_norm,
        )
        next_state = cast(
            OnlineGatedStateBuilderState,
            jax.lax.cond(applied, lambda: candidate_state, lambda: destination_state),
        )
        norms_valid = (
            jnp.isfinite(proposal.gradient_norm)
            & (proposal.gradient_norm >= 0.0)
            & jnp.isfinite(proposal.clipped_gradient_norm)
            & (proposal.clipped_gradient_norm >= 0.0)
            & jnp.isfinite(proposal.parameter_update_norm)
            & (proposal.parameter_update_norm >= 0.0)
        )
        safe_gradient_norm = jnp.where(
            norms_valid,
            proposal.gradient_norm,
            jnp.asarray(_FLOAT32_MAX, dtype=jnp.float32),
        )
        safe_clipped_norm = jnp.where(
            norms_valid,
            proposal.clipped_gradient_norm,
            jnp.asarray(_FLOAT32_MAX, dtype=jnp.float32),
        )
        safe_update_norm = jnp.where(
            norms_valid,
            proposal.parameter_update_norm,
            jnp.asarray(_FLOAT32_MAX, dtype=jnp.float32),
        )
        diagnostics = StateBuilderLearningDiagnostics(
            gradient_norm=safe_gradient_norm,
            clipped_gradient_norm=safe_clipped_norm,
            parameter_update_norm=safe_update_norm,
            proposal_valid=proposal_valid,
            source_matches=source_matches,
            capacity_available=capacity_available,
            candidate_parameters_valid=candidate_parameters_valid,
            applied=applied,
            fixed_noop=jnp.asarray(False),
            valid=applied,
            rejected=~applied,
            pre_update_words=destination_state.update_words,
            post_update_words=next_state.update_words,
            lifetime_counter_valid=lifetime_counter_valid,
            lifetime_capacity_available=capacity_available,
            update_applied=applied,
        )
        return next_state, diagnostics

    def learn(
        self,
        state: OnlineGatedStateBuilderState,
        representation_gradient: Array,
    ) -> tuple[OnlineGatedStateBuilderState, StateBuilderLearningDiagnostics]:
        """Propose and commit one update against the same parameter version."""
        proposal = self.propose_learning_update(state, representation_gradient)
        return self.commit_learning_update(state, proposal)


@dataclass(frozen=True)
class LearnableGRUStateBuilderConfig(OnlineGatedStateBuilderConfig):
    """Configuration for a conventional fully recurrent trainable GRU.

    Unlike :class:`OnlineGatedStateBuilder`, every hidden unit can influence
    every other hidden unit through learned update, reset, and candidate
    matrices.  Exact fixed-parameter RTRL sensitivities make the emitted state
    compatible with the same causal proposal/commit learning boundary.
    """

    # Keep the parent compatibility field non-configurable and out of the
    # manifest.  The full GRU has distinct update/reset gate biases below.
    initial_gate_bias: float = field(
        default=-2.0,
        init=False,
        repr=False,
        compare=False,
    )
    initial_update_bias: float = 1.0
    initial_reset_bias: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("initial_update_bias", "initial_reset_bias"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"{name} must be an exact finite float")
        if type(self.include_raw_observation) is not bool:
            raise ValueError("include_raw_observation must be an exact bool")

    def event_dim(self) -> int:
        """Return observation + one-hot action + reward + discount width."""

        return self.observation_dim + self.n_actions + 2

    def parameter_count(self) -> int:
        """Return three input matrices, recurrent matrices, and bias vectors."""

        per_gate = self.hidden_dim * (
            self.event_dim() + self.hidden_dim + 1
        )
        return 3 * per_gate

    def feature_dim(self) -> int:
        """Return raw-observation plus hidden-state width when configured."""

        return self.hidden_dim + (
            self.observation_dim if self.include_raw_observation else 0
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize the exact builder configuration."""

        payload = asdict(self)
        payload.pop("initial_gate_bias")
        payload["type"] = "LearnableGRUStateBuilder"
        payload["state_schema"] = LEARNABLE_GRU_STATE_BUILDER_STATE_SCHEMA
        return payload

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> LearnableGRUStateBuilderConfig:
        """Strictly reconstruct current-schema :meth:`to_config` output."""

        data = dict(payload)
        expected = {
            "type",
            "state_schema",
            "observation_dim",
            "n_actions",
            "hidden_dim",
            "step_size",
            "gradient_clip",
            "initial_update_bias",
            "initial_reset_bias",
            "initialization_scale",
            "include_raw_observation",
        }
        if set(data) != expected:
            missing = sorted(expected - set(data))
            extra = sorted(set(data) - expected)
            raise ValueError(
                "learnable-GRU state-builder config manifest is not exact; "
                f"missing={missing}, extra={extra}"
            )
        if data.pop("type") != "LearnableGRUStateBuilder":
            raise ValueError("learnable-GRU state-builder config type is unsupported")
        if data.pop("state_schema") != LEARNABLE_GRU_STATE_BUILDER_STATE_SCHEMA:
            raise ValueError("learnable-GRU state-builder state schema is unsupported")
        return cls(**data)


# The full GRU carries the same exact clocks, parameter vector, hidden vector,
# and RTRL matrix as the diagonal reference.  These aliases intentionally make
# checkpoint/state consumers share one stable structural contract.
LearnableGRUStateBuilderState = OnlineGatedStateBuilderState
LearnableGRUStateBuilderTransitionResult = OnlineGatedStateBuilderTransitionResult


class LearnableGRUStateBuilder(OnlineGatedStateBuilder):
    """Conventional full GRU with exact fixed-parameter RTRL sensitivities.

    The sensitivity recurrence is ``S_t = d h_t / d theta + J_t S_{t-1}``,
    where ``J_t`` is the dense hidden-to-hidden Jacobian.  Carrying this matrix
    across online parameter changes has the same explicitly documented
    changing-parameter eligibility interpretation as the diagonal builder.
    """

    def __init__(self, config: LearnableGRUStateBuilderConfig):
        if type(config) is not LearnableGRUStateBuilderConfig:
            raise TypeError("config must be an exact LearnableGRUStateBuilderConfig")
        self._config = config
        self._learning_fingerprint = _builder_learning_fingerprint(config.to_config())

    @property
    def config(self) -> LearnableGRUStateBuilderConfig:
        """Return the immutable full-GRU configuration."""

        return cast(LearnableGRUStateBuilderConfig, self._config)

    def init(self, key: Array) -> LearnableGRUStateBuilderState:
        """Initialize all three gates and the exact online sensitivity matrix."""

        cfg = self.config
        keys = jr.split(key, 6)
        scale = jnp.asarray(cfg.initialization_scale, dtype=jnp.float32)

        def matrix(draw_key: Array, shape: tuple[int, int]) -> Array:
            fan_in = jnp.asarray(max(shape[1], 1), dtype=jnp.float32)
            return scale * jr.normal(draw_key, shape, dtype=jnp.float32) / jnp.sqrt(
                fan_in
            )

        gate_parameters: list[Array] = []
        for gate_index, bias in enumerate(
            (cfg.initial_update_bias, cfg.initial_reset_bias, 0.0)
        ):
            input_weights = matrix(
                keys[2 * gate_index],
                (cfg.hidden_dim, cfg.event_dim()),
            )
            recurrent_weights = matrix(
                keys[2 * gate_index + 1],
                (cfg.hidden_dim, cfg.hidden_dim),
            )
            gate_parameters.extend(
                (
                    input_weights.reshape((-1,)),
                    recurrent_weights.reshape((-1,)),
                    jnp.full((cfg.hidden_dim,), bias, dtype=jnp.float32),
                )
            )
        parameters = jnp.concatenate(gate_parameters).astype(jnp.float32)
        return OnlineGatedStateBuilderState(
            parameters=parameters,
            hidden=jnp.zeros((cfg.hidden_dim,), dtype=jnp.float32),
            parameter_sensitivity=jnp.zeros(
                (cfg.hidden_dim, cfg.parameter_count()),
                dtype=jnp.float32,
            ),
            step_count=jnp.asarray(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
            update_count=jnp.asarray(0, dtype=jnp.int32),
            update_words=jnp.zeros((2,), dtype=jnp.uint32),
            last_gradient_norm=jnp.asarray(0.0, dtype=jnp.float32),
        )

    def _unpack_gru_parameters(
        self,
        parameters: Array,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]:
        cfg = self.config
        input_size = cfg.hidden_dim * cfg.event_dim()
        recurrent_size = cfg.hidden_dim * cfg.hidden_dim
        offset = 0
        unpacked: list[Array] = []
        for _ in range(3):
            unpacked.append(
                parameters[offset : offset + input_size].reshape(
                    (cfg.hidden_dim, cfg.event_dim())
                )
            )
            offset += input_size
            unpacked.append(
                parameters[offset : offset + recurrent_size].reshape(
                    (cfg.hidden_dim, cfg.hidden_dim)
                )
            )
            offset += recurrent_size
            unpacked.append(parameters[offset : offset + cfg.hidden_dim])
            offset += cfg.hidden_dim
        return cast(
            tuple[
                Array,
                Array,
                Array,
                Array,
                Array,
                Array,
                Array,
                Array,
                Array,
            ],
            tuple(unpacked),
        )

    def _transition(self, parameters: Array, hidden: Array, event: Array) -> Array:
        (
            update_input,
            update_recurrent,
            update_bias,
            reset_input,
            reset_recurrent,
            reset_bias,
            candidate_input,
            candidate_recurrent,
            candidate_bias,
        ) = self._unpack_gru_parameters(parameters)
        update_gate = jax.nn.sigmoid(
            update_input @ event + update_recurrent @ hidden + update_bias
        )
        reset_gate = jax.nn.sigmoid(
            reset_input @ event + reset_recurrent @ hidden + reset_bias
        )
        candidate = jnp.tanh(
            candidate_input @ event
            + candidate_recurrent @ (reset_gate * hidden)
            + candidate_bias
        )
        return update_gate * hidden + (1.0 - update_gate) * candidate

    def _next_parameter_sensitivity(
        self,
        parameters: Array,
        hidden: Array,
        event: Array,
        previous_sensitivity: Array,
    ) -> Array:
        direct, recurrent_jacobian = jax.jacfwd(
            self._transition,
            argnums=(0, 1),
        )(parameters, hidden, event)
        return cast(Array, direct + recurrent_jacobian @ previous_sensitivity)


@dataclass(frozen=True)
class RecurrentTraceUnitStateBuilderConfig:
    """Configuration for a diagonal complex recurrent trace unit.

    The event input is causal: the current observation is paired with the
    action, reward, and discount from the transition that produced it.  The
    compressed sensitivity is exact when recurrent parameters are fixed.
    Without ``rtrl_taylor_correction``, carrying it across accepted parameter
    updates is explicitly the ordinary moving-parameter approximation.

    The optional Taylor path remains an approximation under moving
    parameters.  It is enabled only together with persistent ownership of the
    exact recurrent-parameter source and the actual accumulated delta used by
    the next transition.
    """

    observation_dim: int
    n_actions: int = 0
    hidden_dim: int = 8
    step_size: float = 0.01
    gradient_clip: float = 10.0
    r_min: float = 0.0
    r_max: float = 1.0
    max_phase: float = 6.28
    rtu_epsilon: float = 1.0e-8
    include_raw_observation: bool = True
    rtrl_taylor_correction: bool = False

    def __post_init__(self) -> None:
        for name in ("observation_dim", "hidden_dim"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be an exact positive integer")
        if type(self.n_actions) is not int or self.n_actions < 0:
            raise ValueError("n_actions must be an exact non-negative integer")
        for name in (
            "step_size",
            "gradient_clip",
            "r_min",
            "r_max",
            "max_phase",
            "rtu_epsilon",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"{name} must be an exact finite float")
            magnitude = abs(value)
            if magnitude > _FLOAT32_MAX or (
                magnitude != 0.0 and magnitude < _FLOAT32_TINY
            ):
                raise ValueError(
                    f"{name} must be zero or a finite normal float32 value"
                )
        if self.step_size <= 0.0:
            raise ValueError("step_size must be positive")
        if self.gradient_clip <= 0.0:
            raise ValueError("gradient_clip must be positive")
        if not 0.0 <= self.r_min < self.r_max <= 1.0:
            raise ValueError("r_min and r_max must satisfy 0 <= r_min < r_max <= 1")
        minimum_radius_squared = float(self.r_min**2)
        maximum_radius_squared = float(self.r_max**2)
        if self.r_min != 0.0 and minimum_radius_squared < _FLOAT32_TINY:
            raise ValueError("nonzero r_min squared must be a finite normal float32")
        if maximum_radius_squared < _FLOAT32_TINY:
            raise ValueError("r_max squared must be a finite normal float32")
        if not max(minimum_radius_squared, _FLOAT32_TINY) < maximum_radius_squared:
            raise ValueError("r_min and r_max define an empty float32 interval")
        if not 0.0 < self.max_phase <= 2.0 * math.pi:
            raise ValueError("max_phase must lie in (0, 2*pi]")
        if not 0.0 < self.rtu_epsilon < 1.0:
            raise ValueError("rtu_epsilon must lie in (0, 1)")
        for name in ("include_raw_observation", "rtrl_taylor_correction"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be an exact bool")

    def event_dim(self) -> int:
        """Return observation + one-hot action + reward + discount width."""

        return self.observation_dim + self.n_actions + 2

    def parameter_count(self) -> int:
        """Return both polar vectors and both dense event projections."""

        return 2 * self.hidden_dim + 2 * self.hidden_dim * self.event_dim()

    def sensitivity_scalar_count(self) -> int:
        """Return the exact compressed fixed-parameter RTRL footprint."""

        return 4 * self.hidden_dim + 4 * self.hidden_dim * self.event_dim()

    def feature_dim(self) -> int:
        """Return raw observation plus real and imaginary hidden components."""

        recurrent_width = 2 * self.hidden_dim
        return recurrent_width + (
            self.observation_dim if self.include_raw_observation else 0
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize the exact builder configuration."""

        payload = asdict(self)
        payload["type"] = "RecurrentTraceUnitStateBuilder"
        payload["state_schema"] = RECURRENT_TRACE_UNIT_STATE_BUILDER_STATE_SCHEMA
        return payload

    @classmethod
    def from_config(
        cls,
        payload: dict[str, Any],
    ) -> RecurrentTraceUnitStateBuilderConfig:
        """Strictly reconstruct current-schema :meth:`to_config` output."""

        data = dict(payload)
        expected = {
            "type",
            "state_schema",
            "observation_dim",
            "n_actions",
            "hidden_dim",
            "step_size",
            "gradient_clip",
            "r_min",
            "r_max",
            "max_phase",
            "rtu_epsilon",
            "include_raw_observation",
            "rtrl_taylor_correction",
        }
        if set(data) != expected:
            missing = sorted(expected - set(data))
            extra = sorted(set(data) - expected)
            raise ValueError(
                "recurrent-trace-unit state-builder config manifest is not exact; "
                f"missing={missing}, extra={extra}"
            )
        if data.pop("type") != "RecurrentTraceUnitStateBuilder":
            raise ValueError(
                "recurrent-trace-unit state-builder config type is unsupported"
            )
        if data.pop("state_schema") != RECURRENT_TRACE_UNIT_STATE_BUILDER_STATE_SCHEMA:
            raise ValueError(
                "recurrent-trace-unit state-builder state schema is unsupported"
            )
        return cls(**data)


@chex.dataclass(frozen=True)
class RecurrentTraceUnitStateBuilderState:
    """RTU parameters, complex recurrence, and compressed online traces.

    Taylor-disabled states carry ``None`` in all four optional ownership
    slots.  Taylor-enabled states carry a compressed diagonal Taylor trace,
    the exact parameters that produced the incoming recurrent state, the
    actual accumulated parameter delta, and that source's uint64 update words.
    """

    parameters: Float[Array, " parameter_count"]
    rtu_state: RTUState
    sensitivities: RTUSensitivities
    taylor_trace: RTUSensitivities | None
    sensitivity_source_parameters: Float[Array, " parameter_count"] | None
    sensitivity_parameter_delta: Float[Array, " parameter_count"] | None
    sensitivity_source_update_words: UInt[Array, " 2"] | None
    step_count: Int[Array, ""]
    step_words: UInt[Array, " 2"]
    update_count: Int[Array, ""]
    update_words: UInt[Array, " 2"]
    last_gradient_norm: Float[Array, ""]


@chex.dataclass(frozen=True)
class RecurrentTraceUnitStateBuilderTransitionResult:
    """One fail-stop RTU observation transition and its exact identity."""

    state: RecurrentTraceUnitStateBuilderState
    representation: Float[Array, " feature_dim"]
    pre_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    state_valid: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    candidate_representation_valid: Bool[Array, ""]
    step_counter_valid: Bool[Array, ""]
    step_capacity_available: Bool[Array, ""]
    transition_applied: Bool[Array, ""]


class RecurrentTraceUnitStateBuilder:
    """Diagonal complex RTU state with compressed online RTRL.

    No dense hidden-by-parameter Jacobian is persisted.  For fixed recurrent
    parameters, :func:`rtu_step` propagates the exact unrolled Jacobian in
    ``O(H + H F)`` storage.  Online commits preserve the causal recurrent state
    and traces.  The default therefore has the documented moving-parameter
    eligibility approximation after a commit.  The optional diagonal Taylor
    correction is also approximate, but its input delta is exact because the
    state transaction owns both its source vector and accumulated delta.
    """

    def __init__(self, config: RecurrentTraceUnitStateBuilderConfig):
        if type(config) is not RecurrentTraceUnitStateBuilderConfig:
            raise TypeError(
                "config must be an exact RecurrentTraceUnitStateBuilderConfig"
            )
        self._config = config
        self._learning_fingerprint = _builder_learning_fingerprint(config.to_config())

    @property
    def config(self) -> RecurrentTraceUnitStateBuilderConfig:
        """Return the immutable RTU configuration."""

        return self._config

    def feature_dim(self) -> int:
        """Return the fixed emitted representation width."""

        return self.config.feature_dim()

    def observation_dim(self) -> int:
        """Return the raw observation dimension."""

        return self.config.observation_dim

    def resource_budget(self) -> StateBuilderBudget:
        """Return the exact persistent compressed-state budget."""

        cfg = self.config
        parameter_count = cfg.parameter_count()
        state_scalars = (
            parameter_count
            + 2 * cfg.hidden_dim
            + cfg.sensitivity_scalar_count()
            + 7
        )
        if cfg.rtrl_taylor_correction:
            state_scalars += (
                cfg.sensitivity_scalar_count()
                + 2 * parameter_count
                + 2
            )
        return StateBuilderBudget(
            output_scalars=self.feature_dim(),
            trainable_scalars=parameter_count,
            state_scalars=state_scalars,
            state_bytes=4 * state_scalars,
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize the exact RTU builder configuration."""

        return self.config.to_config()

    def _unpack_parameters(self, parameters: Array) -> RTUParameters:
        cfg = self.config
        hidden_dim = cfg.hidden_dim
        matrix_size = hidden_dim * cfg.event_dim()
        nu_log = parameters[:hidden_dim]
        theta_log = parameters[hidden_dim : 2 * hidden_dim]
        matrix_offset = 2 * hidden_dim
        b_real = parameters[matrix_offset : matrix_offset + matrix_size].reshape(
            (hidden_dim, cfg.event_dim())
        )
        matrix_offset += matrix_size
        b_imag = parameters[matrix_offset : matrix_offset + matrix_size].reshape(
            (hidden_dim, cfg.event_dim())
        )
        return RTUParameters(
            nu_log=nu_log,
            theta_log=theta_log,
            b_real=b_real,
            b_imag=b_imag,
        )

    @staticmethod
    def _pack_parameters(parameters: RTUParameters) -> Array:
        return jnp.concatenate(
            (
                parameters.nu_log,
                parameters.theta_log,
                parameters.b_real.reshape((-1,)),
                parameters.b_imag.reshape((-1,)),
            )
        ).astype(jnp.float32)

    def init(self, key: Array) -> RecurrentTraceUnitStateBuilderState:
        """Initialize stable polar dynamics and compressed zero traces."""

        cfg = self.config
        nu_key, theta_key, real_key, imaginary_key = jr.split(key, 4)
        radius_squared = jr.uniform(
            nu_key,
            (cfg.hidden_dim,),
            minval=jnp.asarray(max(cfg.r_min**2, _FLOAT32_TINY), dtype=jnp.float32),
            maxval=jnp.asarray(cfg.r_max**2, dtype=jnp.float32),
            dtype=jnp.float32,
        )
        radius_squared = jnp.maximum(
            radius_squared,
            jnp.asarray(_FLOAT32_TINY, dtype=jnp.float32),
        )
        nu_log = jnp.log(-0.5 * jnp.log(radius_squared))
        phase = cfg.max_phase * jr.uniform(
            theta_key,
            (cfg.hidden_dim,),
            dtype=jnp.float32,
        )
        phase = jnp.maximum(phase, jnp.asarray(_FLOAT32_TINY, dtype=jnp.float32))
        theta_log = jnp.log(phase)
        input_scale = jnp.sqrt(jnp.asarray(cfg.event_dim(), dtype=jnp.float32))
        params = RTUParameters(
            nu_log=nu_log,
            theta_log=theta_log,
            b_real=jr.normal(
                real_key,
                (cfg.hidden_dim, cfg.event_dim()),
                dtype=jnp.float32,
            )
            / input_scale,
            b_imag=jr.normal(
                imaginary_key,
                (cfg.hidden_dim, cfg.event_dim()),
                dtype=jnp.float32,
            )
            / input_scale,
        )
        parameters = self._pack_parameters(params)
        sensitivities = zero_rtu_sensitivities(cfg.hidden_dim, cfg.event_dim())
        taylor_trace = (
            zero_rtu_sensitivities(cfg.hidden_dim, cfg.event_dim())
            if cfg.rtrl_taylor_correction
            else None
        )
        return RecurrentTraceUnitStateBuilderState(
            parameters=parameters,
            rtu_state=zero_rtu_state(cfg.hidden_dim),
            sensitivities=sensitivities,
            taylor_trace=taylor_trace,
            sensitivity_source_parameters=(
                parameters if cfg.rtrl_taylor_correction else None
            ),
            sensitivity_parameter_delta=(
                jnp.zeros_like(parameters) if cfg.rtrl_taylor_correction else None
            ),
            sensitivity_source_update_words=(
                jnp.zeros((2,), dtype=jnp.uint32)
                if cfg.rtrl_taylor_correction
                else None
            ),
            step_count=jnp.asarray(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
            update_count=jnp.asarray(0, dtype=jnp.int32),
            update_words=jnp.zeros((2,), dtype=jnp.uint32),
            last_gradient_norm=jnp.asarray(0.0, dtype=jnp.float32),
        )

    def _transition(
        self,
        parameters: Array,
        state: RTUState,
        event: Array,
    ) -> RTUState:
        return rtu_forward(
            self._unpack_parameters(parameters),
            state,
            event,
            epsilon=self.config.rtu_epsilon,
        )

    def _event(
        self,
        raw_observation: Array,
        previous_action: Array | int,
        previous_reward: Array | float,
        previous_discount: Array | float,
    ) -> Array:
        """Construct the causal current-observation/previous-outcome event."""

        observation = jnp.asarray(raw_observation, dtype=jnp.float32).reshape(
            (self.config.observation_dim,)
        )
        return jnp.concatenate(
            (
                observation,
                _action_features(previous_action, self.config.n_actions),
                jnp.atleast_1d(jnp.asarray(previous_reward, dtype=jnp.float32)),
                jnp.atleast_1d(jnp.asarray(previous_discount, dtype=jnp.float32)),
            )
        )

    def _advance_recurrence(
        self,
        state: RecurrentTraceUnitStateBuilderState,
        parameters: Array,
        rtu_state: RTUState,
        sensitivities: RTUSensitivities,
        event: Array,
    ) -> tuple[RTUState, RTUSensitivities, RTUSensitivities | None]:
        params = self._unpack_parameters(parameters)
        if self.config.rtrl_taylor_correction:
            if state.taylor_trace is None or state.sensitivity_parameter_delta is None:
                raise ValueError("Taylor-enabled RTU state lacks exact delta ownership")
            next_state, next_sensitivities, next_taylor = rtu_taylor_step(
                params,
                rtu_state,
                sensitivities,
                state.taylor_trace,
                self._unpack_parameters(state.sensitivity_parameter_delta),
                event,
                epsilon=self.config.rtu_epsilon,
            )
            return next_state, next_sensitivities, next_taylor
        next_state, next_sensitivities = rtu_step(
            params,
            rtu_state,
            sensitivities,
            event,
            epsilon=self.config.rtu_epsilon,
        )
        return next_state, next_sensitivities, None

    @functools.partial(jax.jit, static_argnums=(0,))
    def encode(
        self,
        state: RecurrentTraceUnitStateBuilderState,
        raw_observation: Array,
    ) -> Float[Array, " feature_dim"]:
        """Pair raw input with the current complex state without advancing."""

        observation = jnp.asarray(raw_observation, dtype=jnp.float32).reshape(
            (self.config.observation_dim,)
        )
        recurrent = jnp.concatenate(
            (state.rtu_state.real, state.rtu_state.imaginary)
        )
        if self.config.include_raw_observation:
            return jnp.concatenate((observation, recurrent))
        return recurrent

    @functools.partial(jax.jit, static_argnums=(0,))
    def update_with_status(
        self,
        state: RecurrentTraceUnitStateBuilderState,
        raw_observation: Array,
        previous_action: Array | int,
        previous_reward: Array | float,
        previous_discount: Array | float,
    ) -> RecurrentTraceUnitStateBuilderTransitionResult:
        """Atomically advance the RTU recurrence and exact step clock."""

        previous_action = _canonical_start_action(previous_action)
        previous_reward = _canonical_start_outcome(
            previous_reward,
            name="previous_reward",
        )
        previous_discount = _canonical_start_outcome(
            previous_discount,
            name="previous_discount",
        )
        self._validate_state_static_contract(state)
        proposed_step_words, step_capacity_available = (
            _checked_step_words_increment(state.step_words)
        )
        step_counter_valid = _step_lifetime_counter_valid(
            state.step_words,
            state.step_count,
        )
        _static_array_contract(
            raw_observation,
            name="raw_observation",
            shape=(self.config.observation_dim,),
            dtype=jnp.float32,
        )
        _static_array_contract(
            previous_action,
            name="previous_action",
            shape=(),
            dtype=jnp.int32,
        )
        _static_array_contract(
            previous_reward,
            name="previous_reward",
            shape=(),
            dtype=jnp.float32,
        )
        _static_array_contract(
            previous_discount,
            name="previous_discount",
            shape=(),
            dtype=jnp.float32,
        )
        observation = jnp.asarray(raw_observation)
        action = jnp.asarray(previous_action)
        reward = jnp.asarray(previous_reward)
        discount = jnp.asarray(previous_discount)
        action_valid = jnp.asarray(True, dtype=jnp.bool_)
        if self.config.n_actions > 0:
            action_valid = (action >= -1) & (action < self.config.n_actions)
        input_valid = (
            jnp.all(jnp.isfinite(observation))
            & action_valid
            & jnp.isfinite(reward)
            & jnp.isfinite(discount)
            & (discount >= 0.0)
            & (discount <= 1.0)
        )
        state_valid = self._state_is_valid(state)
        safe_observation = jnp.where(jnp.isfinite(observation), observation, 0.0)
        safe_action = jnp.where(action_valid, action, jnp.asarray(0, dtype=jnp.int32))
        safe_reward = jnp.where(jnp.isfinite(reward), reward, 0.0)
        safe_discount = jnp.where(jnp.isfinite(discount), discount, 0.0)
        safe_parameters = jnp.where(jnp.isfinite(state.parameters), state.parameters, 0.0)
        safe_rtu_state = cast(
            RTUState,
            jax.tree.map(
                lambda value: jnp.where(jnp.isfinite(value), value, 0.0),
                state.rtu_state,
            ),
        )
        safe_sensitivities = cast(
            RTUSensitivities,
            jax.tree.map(
                lambda value: jnp.where(jnp.isfinite(value), value, 0.0),
                state.sensitivities,
            ),
        )
        safe_state = state
        if self.config.rtrl_taylor_correction:
            if state.taylor_trace is None or state.sensitivity_parameter_delta is None:
                raise ValueError("Taylor-enabled RTU state lacks exact delta ownership")
            safe_state = cast(
                RecurrentTraceUnitStateBuilderState,
                cast(Any, state).replace(
                    taylor_trace=cast(
                        RTUSensitivities,
                        jax.tree.map(
                            lambda value: jnp.where(jnp.isfinite(value), value, 0.0),
                            state.taylor_trace,
                        ),
                    ),
                    sensitivity_parameter_delta=jnp.where(
                        jnp.isfinite(state.sensitivity_parameter_delta),
                        state.sensitivity_parameter_delta,
                        0.0,
                    ),
                ),
            )
        event = self._event(
            safe_observation,
            safe_action,
            safe_reward,
            safe_discount,
        )
        next_rtu_state, next_sensitivities, next_taylor = self._advance_recurrence(
            safe_state,
            safe_parameters,
            safe_rtu_state,
            safe_sensitivities,
            event,
        )
        candidate_state = RecurrentTraceUnitStateBuilderState(
            parameters=state.parameters,
            rtu_state=next_rtu_state,
            sensitivities=next_sensitivities,
            taylor_trace=next_taylor,
            sensitivity_source_parameters=(
                state.parameters if self.config.rtrl_taylor_correction else None
            ),
            sensitivity_parameter_delta=(
                jnp.zeros_like(state.parameters)
                if self.config.rtrl_taylor_correction
                else None
            ),
            sensitivity_source_update_words=(
                state.update_words if self.config.rtrl_taylor_correction else None
            ),
            step_count=_saturating_int32_increment(state.step_count),
            step_words=proposed_step_words,
            update_count=state.update_count,
            update_words=state.update_words,
            last_gradient_norm=state.last_gradient_norm,
        )
        candidate_state_valid = self._state_is_valid(candidate_state)
        candidate_representation = self.encode(candidate_state, safe_observation)
        candidate_representation_valid = jnp.all(jnp.isfinite(candidate_representation))
        transition_applied = (
            state_valid
            & input_valid
            & step_counter_valid
            & step_capacity_available
            & candidate_state_valid
            & candidate_representation_valid
        )
        next_state = cast(
            RecurrentTraceUnitStateBuilderState,
            jax.lax.cond(
                transition_applied,
                lambda: candidate_state,
                lambda: state,
            ),
        )
        committed_representation = self.encode(state, safe_observation)
        finite_committed_representation = jnp.where(
            jnp.isfinite(committed_representation),
            committed_representation,
            0.0,
        )
        representation = jax.lax.cond(
            transition_applied,
            lambda: candidate_representation,
            lambda: finite_committed_representation,
        )
        return RecurrentTraceUnitStateBuilderTransitionResult(
            state=next_state,
            representation=representation,
            pre_step_words=state.step_words,
            post_step_words=next_state.step_words,
            state_valid=state_valid,
            input_valid=input_valid,
            candidate_state_valid=candidate_state_valid,
            candidate_representation_valid=candidate_representation_valid,
            step_counter_valid=step_counter_valid,
            step_capacity_available=step_capacity_available,
            transition_applied=transition_applied,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: RecurrentTraceUnitStateBuilderState,
        raw_observation: Array,
        previous_action: Array | int,
        previous_reward: Array | float,
        previous_discount: Array | float,
    ) -> tuple[RecurrentTraceUnitStateBuilderState, Float[Array, " feature_dim"]]:
        """Advance recurrence while retaining the common tuple API."""

        result = self.update_with_status(
            state,
            raw_observation,
            previous_action,
            previous_reward,
            previous_discount,
        )
        return result.state, result.representation

    def start(
        self,
        state: RecurrentTraceUnitStateBuilderState,
        raw_observation: Array,
        last_action: Array | int = -1,
        last_reward: Array | float = 0.0,
        last_discount: Array | float = 1.0,
    ) -> tuple[RecurrentTraceUnitStateBuilderState, Array]:
        """Consume an episode's initial causal observation event."""

        return cast(
            tuple[RecurrentTraceUnitStateBuilderState, Array],
            self.update(
                state,
                raw_observation,
                _canonical_start_action(last_action),
                _canonical_start_outcome(last_reward, name="last_reward"),
                _canonical_start_outcome(last_discount, name="last_discount"),
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset_episode(
        self,
        state: RecurrentTraceUnitStateBuilderState,
    ) -> RecurrentTraceUnitStateBuilderState:
        """Clear episode-local RTU history while preserving lifetime state."""

        self._validate_state_static_contract(state)
        zero_sensitivities = zero_rtu_sensitivities(
            self.config.hidden_dim,
            self.config.event_dim(),
        )
        candidate = RecurrentTraceUnitStateBuilderState(
            parameters=state.parameters,
            rtu_state=zero_rtu_state(self.config.hidden_dim),
            sensitivities=zero_sensitivities,
            taylor_trace=(
                zero_rtu_sensitivities(
                    self.config.hidden_dim,
                    self.config.event_dim(),
                )
                if self.config.rtrl_taylor_correction
                else None
            ),
            sensitivity_source_parameters=(
                state.parameters if self.config.rtrl_taylor_correction else None
            ),
            sensitivity_parameter_delta=(
                jnp.zeros_like(state.parameters)
                if self.config.rtrl_taylor_correction
                else None
            ),
            sensitivity_source_update_words=(
                state.update_words if self.config.rtrl_taylor_correction else None
            ),
            step_count=state.step_count,
            step_words=state.step_words,
            update_count=state.update_count,
            update_words=state.update_words,
            last_gradient_norm=state.last_gradient_norm,
        )
        valid = self._state_is_valid(state) & self._state_is_valid(candidate)
        return cast(
            RecurrentTraceUnitStateBuilderState,
            jax.lax.cond(valid, lambda: candidate, lambda: state),
        )

    def _validate_sensitivities_static_contract(
        self,
        sensitivities: RTUSensitivities,
        *,
        name: str,
    ) -> None:
        if not isinstance(sensitivities, RTUSensitivities):
            raise TypeError(f"{name} must be an RTUSensitivities tuple")
        hidden_dim = self.config.hidden_dim
        event_dim = self.config.event_dim()
        for field_name in ("nu_log", "theta_log"):
            _static_array_contract(
                getattr(sensitivities, field_name),
                name=f"{name}.{field_name}",
                shape=(2, hidden_dim),
                dtype=jnp.float32,
            )
        for field_name in ("b_real", "b_imag"):
            _static_array_contract(
                getattr(sensitivities, field_name),
                name=f"{name}.{field_name}",
                shape=(2, hidden_dim, event_dim),
                dtype=jnp.float32,
            )

    def _validate_state_static_contract(
        self,
        state: RecurrentTraceUnitStateBuilderState,
    ) -> None:
        """Reject structural RTU state corruption before tracing execution."""

        if not isinstance(state, RecurrentTraceUnitStateBuilderState):
            raise TypeError("state must be a RecurrentTraceUnitStateBuilderState")
        parameter_count = self.config.parameter_count()
        hidden_dim = self.config.hidden_dim
        _static_array_contract(
            state.parameters,
            name="state.parameters",
            shape=(parameter_count,),
            dtype=jnp.float32,
        )
        if not isinstance(state.rtu_state, RTUState):
            raise TypeError("state.rtu_state must be an RTUState tuple")
        for field_name in ("real", "imaginary"):
            _static_array_contract(
                getattr(state.rtu_state, field_name),
                name=f"state.rtu_state.{field_name}",
                shape=(hidden_dim,),
                dtype=jnp.float32,
            )
        self._validate_sensitivities_static_contract(
            state.sensitivities,
            name="state.sensitivities",
        )
        if self.config.rtrl_taylor_correction:
            if state.taylor_trace is None:
                raise TypeError("Taylor-enabled state.taylor_trace cannot be None")
            self._validate_sensitivities_static_contract(
                state.taylor_trace,
                name="state.taylor_trace",
            )
            if state.sensitivity_source_parameters is None:
                raise TypeError(
                    "Taylor-enabled state.sensitivity_source_parameters cannot be None"
                )
            if state.sensitivity_parameter_delta is None:
                raise TypeError(
                    "Taylor-enabled state.sensitivity_parameter_delta cannot be None"
                )
            if state.sensitivity_source_update_words is None:
                raise TypeError(
                    "Taylor-enabled state.sensitivity_source_update_words cannot be None"
                )
            _static_array_contract(
                state.sensitivity_source_parameters,
                name="state.sensitivity_source_parameters",
                shape=(parameter_count,),
                dtype=jnp.float32,
            )
            _static_array_contract(
                state.sensitivity_parameter_delta,
                name="state.sensitivity_parameter_delta",
                shape=(parameter_count,),
                dtype=jnp.float32,
            )
            _static_array_contract(
                state.sensitivity_source_update_words,
                name="state.sensitivity_source_update_words",
                shape=(2,),
                dtype=jnp.uint32,
            )
        elif any(
            value is not None
            for value in (
                state.taylor_trace,
                state.sensitivity_source_parameters,
                state.sensitivity_parameter_delta,
                state.sensitivity_source_update_words,
            )
        ):
            raise TypeError("Taylor-disabled RTU state must have empty ownership slots")
        for name, value, dtype, shape in (
            ("step_count", state.step_count, jnp.int32, ()),
            ("step_words", state.step_words, jnp.uint32, (2,)),
            ("update_count", state.update_count, jnp.int32, ()),
            ("update_words", state.update_words, jnp.uint32, (2,)),
            ("last_gradient_norm", state.last_gradient_norm, jnp.float32, ()),
        ):
            _static_array_contract(
                value,
                name=f"state.{name}",
                shape=shape,
                dtype=dtype,
            )

    def _state_is_valid(
        self,
        state: RecurrentTraceUnitStateBuilderState,
    ) -> Bool[Array, ""]:
        valid = (
            jnp.all(jnp.isfinite(state.parameters))
            & jnp.all(jnp.isfinite(state.rtu_state.real))
            & jnp.all(jnp.isfinite(state.rtu_state.imaginary))
            & jnp.all(jnp.isfinite(state.sensitivities.nu_log))
            & jnp.all(jnp.isfinite(state.sensitivities.theta_log))
            & jnp.all(jnp.isfinite(state.sensitivities.b_real))
            & jnp.all(jnp.isfinite(state.sensitivities.b_imag))
            & jnp.isfinite(state.last_gradient_norm)
            & (state.last_gradient_norm >= 0.0)
            & _step_lifetime_counter_valid(state.step_words, state.step_count)
            & _update_lifetime_counter_valid(state.update_words, state.update_count)
        )
        if self.config.rtrl_taylor_correction:
            if (
                state.taylor_trace is None
                or state.sensitivity_source_parameters is None
                or state.sensitivity_parameter_delta is None
                or state.sensitivity_source_update_words is None
            ):
                return jnp.asarray(False, dtype=jnp.bool_)
            expected_delta = state.parameters - state.sensitivity_source_parameters
            valid = (
                valid
                & jnp.all(jnp.isfinite(state.taylor_trace.nu_log))
                & jnp.all(jnp.isfinite(state.taylor_trace.theta_log))
                & jnp.all(jnp.isfinite(state.taylor_trace.b_real))
                & jnp.all(jnp.isfinite(state.taylor_trace.b_imag))
                & jnp.all(jnp.isfinite(state.sensitivity_source_parameters))
                & jnp.all(jnp.isfinite(state.sensitivity_parameter_delta))
                & _float32_vectors_bitwise_equal(
                    state.sensitivity_parameter_delta,
                    expected_delta,
                )
                & _uint64_words_not_after(
                    state.sensitivity_source_update_words,
                    state.update_words,
                )
            )
        return jnp.asarray(valid, dtype=jnp.bool_)

    def state_valid(
        self,
        state: RecurrentTraceUnitStateBuilderState,
    ) -> Bool[Array, ""]:
        """Validate the complete RTU state and exact clock contract."""

        self._validate_state_static_contract(state)
        return self._state_is_valid(state)

    def _validate_gradient_static_contract(self, representation_gradient: Array) -> None:
        _static_array_contract(
            representation_gradient,
            name="representation_gradient",
            shape=(self.feature_dim(),),
            dtype=jnp.float32,
        )

    def _compressed_parameter_gradient(
        self,
        sensitivities: RTUSensitivities,
        hidden_gradient: Array,
    ) -> Array:
        component_gradient = hidden_gradient.reshape((2, self.config.hidden_dim))
        return jnp.concatenate(
            (
                jnp.sum(sensitivities.nu_log * component_gradient, axis=0),
                jnp.sum(sensitivities.theta_log * component_gradient, axis=0),
                jnp.sum(
                    sensitivities.b_real * component_gradient[:, :, None],
                    axis=0,
                ).reshape((-1,)),
                jnp.sum(
                    sensitivities.b_imag * component_gradient[:, :, None],
                    axis=0,
                ).reshape((-1,)),
            )
        )

    def propose_learning_update(
        self,
        source_state: RecurrentTraceUnitStateBuilderState,
        dL_drepresentation: Array,  # noqa: N803 - mathematical dL notation
    ) -> StateBuilderLearningProposal:
        """Form a pure clipped proposal from the compressed sensitivity."""

        self._validate_state_static_contract(source_state)
        self._validate_gradient_static_contract(dL_drepresentation)
        return cast(
            StateBuilderLearningProposal,
            self._propose_learning_update_jit(source_state, dL_drepresentation),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _propose_learning_update_jit(
        self,
        source_state: RecurrentTraceUnitStateBuilderState,
        representation_gradient: Array,
    ) -> StateBuilderLearningProposal:
        gradient = jnp.asarray(representation_gradient, dtype=jnp.float32)
        hidden_gradient = gradient[-2 * self.config.hidden_dim :]
        source_state_valid = self._state_is_valid(source_state)
        input_valid = jnp.all(jnp.isfinite(gradient))
        raw_parameter_gradient = self._compressed_parameter_gradient(
            source_state.sensitivities,
            hidden_gradient,
        )
        raw_parameter_gradient_valid = jnp.all(jnp.isfinite(raw_parameter_gradient))
        safe_parameter_gradient = jnp.where(
            input_valid & raw_parameter_gradient_valid,
            raw_parameter_gradient,
            0.0,
        )
        clipped_parameter_gradient, safe_gradient_norm = _scale_safe_clip_by_l2_norm(
            safe_parameter_gradient,
            jnp.asarray(self.config.gradient_clip, dtype=jnp.float32),
        )
        gradient_norm = jnp.where(
            input_valid & raw_parameter_gradient_valid,
            safe_gradient_norm,
            jnp.asarray(_FLOAT32_MAX, dtype=jnp.float32),
        )
        clipped_parameter_gradient_valid = jnp.all(
            jnp.isfinite(clipped_parameter_gradient)
        )
        candidate_parameter_update = (
            -jnp.asarray(self.config.step_size, dtype=jnp.float32)
            * clipped_parameter_gradient
        )
        candidate_parameter_update_valid = jnp.all(
            jnp.isfinite(candidate_parameter_update)
        )
        candidate_parameters = source_state.parameters + candidate_parameter_update
        candidate_parameters_valid = jnp.all(jnp.isfinite(candidate_parameters))
        _, capacity_available = _checked_update_words_increment(
            source_state.update_words
        )
        valid = (
            source_state_valid
            & input_valid
            & raw_parameter_gradient_valid
            & clipped_parameter_gradient_valid
            & candidate_parameter_update_valid
            & candidate_parameters_valid
            & capacity_available
        )
        return StateBuilderLearningProposal(
            builder_fingerprint=self._learning_fingerprint,
            source_parameters=source_state.parameters,
            source_update_count=source_state.update_count,
            source_update_words=source_state.update_words,
            raw_parameter_gradient=raw_parameter_gradient,
            clipped_parameter_gradient=clipped_parameter_gradient,
            candidate_parameter_update=candidate_parameter_update,
            gradient_norm=gradient_norm,
            clipped_gradient_norm=_finite_or_max_norm(
                clipped_parameter_gradient,
                clipped_parameter_gradient_valid,
            ),
            parameter_update_norm=_finite_or_max_norm(
                candidate_parameter_update,
                candidate_parameter_update_valid,
            ),
            source_state_valid=source_state_valid,
            input_valid=input_valid,
            raw_parameter_gradient_valid=raw_parameter_gradient_valid,
            clipped_parameter_gradient_valid=clipped_parameter_gradient_valid,
            candidate_parameter_update_valid=candidate_parameter_update_valid,
            candidate_parameters_valid=candidate_parameters_valid,
            capacity_available=capacity_available,
            candidate_update_transformed=jnp.asarray(False),
            candidate_update_approved=jnp.asarray(True),
            fixed_noop=jnp.asarray(False),
            valid=valid,
            rejected=~valid,
        )

    def _proposal_has_integrity(
        self,
        proposal: StateBuilderLearningProposal,
    ) -> Bool[Array, ""]:
        """Recompute all proposal-derived diagnostics at RTU commit time."""

        raw_valid = jnp.all(jnp.isfinite(proposal.raw_parameter_gradient))
        safe_raw_gradient = jnp.where(
            proposal.input_valid & raw_valid,
            proposal.raw_parameter_gradient,
            0.0,
        )
        expected_clipped, safe_gradient_norm = _scale_safe_clip_by_l2_norm(
            safe_raw_gradient,
            jnp.asarray(self.config.gradient_clip, dtype=jnp.float32),
        )
        expected_gradient_norm = jnp.where(
            proposal.input_valid & raw_valid,
            safe_gradient_norm,
            jnp.asarray(_FLOAT32_MAX, dtype=jnp.float32),
        )
        clipped_valid = jnp.all(jnp.isfinite(proposal.clipped_parameter_gradient))
        expected_formed_update = (
            -jnp.asarray(self.config.step_size, dtype=jnp.float32)
            * proposal.clipped_parameter_gradient
        )
        formed_update_matches = proposal.candidate_update_transformed | (
            _float32_vectors_bitwise_equal(
                proposal.candidate_parameter_update,
                expected_formed_update,
            )
        )
        update_valid = jnp.all(jnp.isfinite(proposal.candidate_parameter_update))
        candidate_parameters = (
            proposal.source_parameters + proposal.candidate_parameter_update
        )
        candidate_parameters_valid = jnp.all(jnp.isfinite(candidate_parameters))
        _, capacity_available = _checked_update_words_increment(
            proposal.source_update_words
        )
        lifetime_counter_valid = _update_lifetime_counter_valid(
            proposal.source_update_words,
            proposal.source_update_count,
        )
        expected_valid = (
            proposal.source_state_valid
            & lifetime_counter_valid
            & proposal.input_valid
            & raw_valid
            & clipped_valid
            & update_valid
            & candidate_parameters_valid
            & capacity_available
            & proposal.candidate_update_approved
        )
        return (
            ~proposal.fixed_noop
            & (proposal.candidate_update_transformed | proposal.candidate_update_approved)
            & (
                ~proposal.source_state_valid
                | jnp.all(jnp.isfinite(proposal.source_parameters))
            )
            & (proposal.raw_parameter_gradient_valid == raw_valid)
            & _float32_vectors_bitwise_equal(
                proposal.clipped_parameter_gradient,
                expected_clipped,
            )
            & (proposal.clipped_parameter_gradient_valid == clipped_valid)
            & formed_update_matches
            & (proposal.candidate_parameter_update_valid == update_valid)
            & (proposal.candidate_parameters_valid == candidate_parameters_valid)
            & (proposal.capacity_available == capacity_available)
            & _diagnostic_scalar_matches(
                proposal.gradient_norm,
                expected_gradient_norm,
            )
            & _diagnostic_scalar_matches(
                proposal.clipped_gradient_norm,
                _finite_or_max_norm(
                    proposal.clipped_parameter_gradient,
                    clipped_valid,
                ),
            )
            & _diagnostic_scalar_matches(
                proposal.parameter_update_norm,
                _finite_or_max_norm(
                    proposal.candidate_parameter_update,
                    update_valid,
                ),
            )
            & (proposal.valid == expected_valid)
            & (proposal.rejected == ~expected_valid)
        )

    def commit_learning_update(
        self,
        destination_state: RecurrentTraceUnitStateBuilderState,
        proposal: StateBuilderLearningProposal,
    ) -> tuple[
        RecurrentTraceUnitStateBuilderState,
        StateBuilderLearningDiagnostics,
    ]:
        """Atomically commit a still-current source-bound RTU proposal."""

        self._validate_state_static_contract(destination_state)
        _validate_learning_proposal_static_contract(
            proposal,
            self.config.parameter_count(),
        )
        return cast(
            tuple[
                RecurrentTraceUnitStateBuilderState,
                StateBuilderLearningDiagnostics,
            ],
            self._commit_learning_update_jit(destination_state, proposal),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _commit_learning_update_jit(
        self,
        destination_state: RecurrentTraceUnitStateBuilderState,
        proposal: StateBuilderLearningProposal,
    ) -> tuple[
        RecurrentTraceUnitStateBuilderState,
        StateBuilderLearningDiagnostics,
    ]:
        proposal_integrity = self._proposal_has_integrity(proposal)
        fingerprint_matches = jnp.array_equal(
            proposal.builder_fingerprint,
            self._learning_fingerprint,
        )
        source_matches = (
            fingerprint_matches
            & _float32_vectors_bitwise_equal(
                destination_state.parameters,
                proposal.source_parameters,
            )
            & (destination_state.update_count == proposal.source_update_count)
            & jnp.array_equal(
                destination_state.update_words,
                proposal.source_update_words,
            )
        )
        destination_state_valid = self._state_is_valid(destination_state)
        proposed_update_words, destination_capacity_available = (
            _checked_update_words_increment(destination_state.update_words)
        )
        lifetime_counter_valid = _update_lifetime_counter_valid(
            destination_state.update_words,
            destination_state.update_count,
        )
        capacity_available = destination_capacity_available & proposal.capacity_available
        candidate_parameters = (
            destination_state.parameters + proposal.candidate_parameter_update
        )
        candidate_parameters_valid = jnp.all(jnp.isfinite(candidate_parameters))
        candidate_delta: Array | None = None
        if self.config.rtrl_taylor_correction:
            if destination_state.sensitivity_source_parameters is None:
                raise ValueError("Taylor-enabled destination lacks source ownership")
            candidate_delta = (
                candidate_parameters
                - destination_state.sensitivity_source_parameters
            )
        candidate_state = RecurrentTraceUnitStateBuilderState(
            parameters=candidate_parameters,
            rtu_state=destination_state.rtu_state,
            sensitivities=destination_state.sensitivities,
            taylor_trace=destination_state.taylor_trace,
            sensitivity_source_parameters=(
                destination_state.sensitivity_source_parameters
                if self.config.rtrl_taylor_correction
                else None
            ),
            sensitivity_parameter_delta=candidate_delta,
            sensitivity_source_update_words=(
                destination_state.sensitivity_source_update_words
                if self.config.rtrl_taylor_correction
                else None
            ),
            step_count=destination_state.step_count,
            step_words=destination_state.step_words,
            update_count=_saturating_int32_increment(destination_state.update_count),
            update_words=proposed_update_words,
            last_gradient_norm=proposal.gradient_norm,
        )
        candidate_state_valid = self._state_is_valid(candidate_state)
        proposal_valid = proposal_integrity & proposal.valid
        applied = (
            destination_state_valid
            & source_matches
            & proposal_valid
            & capacity_available
            & candidate_parameters_valid
            & candidate_state_valid
        )
        next_state = cast(
            RecurrentTraceUnitStateBuilderState,
            jax.lax.cond(
                applied,
                lambda: candidate_state,
                lambda: destination_state,
            ),
        )
        norms_valid = (
            jnp.isfinite(proposal.gradient_norm)
            & (proposal.gradient_norm >= 0.0)
            & jnp.isfinite(proposal.clipped_gradient_norm)
            & (proposal.clipped_gradient_norm >= 0.0)
            & jnp.isfinite(proposal.parameter_update_norm)
            & (proposal.parameter_update_norm >= 0.0)
        )
        maximum = jnp.asarray(_FLOAT32_MAX, dtype=jnp.float32)
        diagnostics = StateBuilderLearningDiagnostics(
            gradient_norm=jnp.where(norms_valid, proposal.gradient_norm, maximum),
            clipped_gradient_norm=jnp.where(
                norms_valid,
                proposal.clipped_gradient_norm,
                maximum,
            ),
            parameter_update_norm=jnp.where(
                norms_valid,
                proposal.parameter_update_norm,
                maximum,
            ),
            proposal_valid=proposal_valid,
            source_matches=source_matches,
            capacity_available=capacity_available,
            candidate_parameters_valid=(
                candidate_parameters_valid & candidate_state_valid
            ),
            applied=applied,
            fixed_noop=jnp.asarray(False),
            valid=applied,
            rejected=~applied,
            pre_update_words=destination_state.update_words,
            post_update_words=next_state.update_words,
            lifetime_counter_valid=lifetime_counter_valid,
            lifetime_capacity_available=capacity_available,
            update_applied=applied,
        )
        return next_state, diagnostics

    def learn(
        self,
        state: RecurrentTraceUnitStateBuilderState,
        representation_gradient: Array,
    ) -> tuple[
        RecurrentTraceUnitStateBuilderState,
        StateBuilderLearningDiagnostics,
    ]:
        """Propose and commit one RTU update against the same source."""

        proposal = self.propose_learning_update(state, representation_gradient)
        return self.commit_learning_update(state, proposal)


def fixed_state_builder_step_counter_nbytes() -> int:
    """Return bytes occupied by fixed-builder telemetry and exact words."""

    return FIXED_STATE_BUILDER_STEP_COUNTER_NBYTES


def migrate_legacy_identity_state_builder_state(
    legacy_state: Any,
) -> IdentityStateBuilderState:
    """Migrate an unambiguous pre-v2 identity state into exact words."""

    if isinstance(legacy_state, Mapping):
        state_fields = dict(legacy_state)
    elif dataclasses.is_dataclass(legacy_state) and not isinstance(legacy_state, type):
        state_fields = {
            state_field.name: getattr(legacy_state, state_field.name)
            for state_field in fields(legacy_state)
        }
    else:
        raise TypeError("legacy identity state must be a mapping or dataclass")
    if set(state_fields) != {"step_count"}:
        missing = sorted({"step_count"} - set(state_fields))
        extra = sorted(set(state_fields) - {"step_count"})
        raise ValueError(
            "legacy identity state field manifest is not exact; "
            f"missing={missing}, extra={extra}"
        )
    step_count = jnp.asarray(state_fields["step_count"])
    if step_count.shape != () or step_count.dtype != jnp.dtype(jnp.int32):
        raise TypeError("legacy identity step_count must be scalar int32")
    step = int(step_count)
    if step < 0:
        raise ValueError("negative legacy identity step_count indicates wrap")
    if step >= _INT32_MAX:
        raise ValueError("saturated legacy identity step_count is ambiguous")
    return IdentityStateBuilderState(
        step_count=step_count,
        step_words=jnp.asarray((0, step), dtype=jnp.uint32),
    )


def migrate_legacy_fixed_trace_state_builder_state(
    legacy_state: Any,
) -> WorkingMemoryState:
    """Migrate an unambiguous pre-v2 fixed-trace working-memory state."""

    return migrate_legacy_working_memory_state(legacy_state)


def online_gated_state_builder_step_counter_nbytes() -> int:
    """Return bytes occupied by step telemetry plus exact identity words."""

    return ONLINE_GATED_STATE_BUILDER_STEP_COUNTER_NBYTES


def online_gated_state_builder_update_counter_nbytes() -> int:
    """Return bytes occupied by update telemetry plus exact identity words."""

    return ONLINE_GATED_STATE_BUILDER_UPDATE_COUNTER_NBYTES


def measure_online_gated_state_builder_state_nbytes(
    state: OnlineGatedStateBuilderState,
) -> int:
    """Measure persistent JAX-array bytes in one online-gated builder state."""

    return sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(state)
        if isinstance(leaf, Array)
    )


def migrate_legacy_online_gated_state_builder_state(
    legacy_state: Any,
) -> OnlineGatedStateBuilderState:
    """Migrate exact v1/v2 clocks into the current two-clock v3 state."""

    if isinstance(legacy_state, Mapping):
        state_fields = dict(legacy_state)
    elif dataclasses.is_dataclass(legacy_state) and not isinstance(legacy_state, type):
        state_fields = {
            state_field.name: getattr(legacy_state, state_field.name)
            for state_field in fields(legacy_state)
        }
    else:
        raise TypeError("legacy online-gated state must be a mapping or dataclass")
    current_names = {
        state_field.name
        for state_field in fields(cast(Any, OnlineGatedStateBuilderState))
    }
    supplied_names = set(state_fields)
    v2_names = current_names - {"step_words"}
    v1_names = current_names - {"step_words", "update_words"}
    if supplied_names not in (v1_names, v2_names):
        missing = sorted(v2_names - supplied_names)
        extra = sorted(supplied_names - v2_names)
        raise ValueError(
            "legacy online-gated state field manifest is not exact; "
            f"missing={missing}, extra={extra}"
        )
    update_count = jnp.asarray(state_fields["update_count"])
    if update_count.shape != () or update_count.dtype != jnp.dtype(jnp.int32):
        raise TypeError("legacy online-gated update_count must be scalar int32")
    update = int(update_count)
    if update < 0:
        raise ValueError("negative legacy online-gated update_count indicates wrap")
    if supplied_names == v1_names:
        if update >= _INT32_MAX:
            raise ValueError("saturated legacy online-gated update_count is ambiguous")
        state_fields["update_words"] = jnp.asarray((0, update), dtype=jnp.uint32)
    else:
        update_words = jnp.asarray(state_fields["update_words"])
        if update_words.shape != (2,) or update_words.dtype != jnp.dtype(jnp.uint32):
            raise TypeError("legacy online-gated update_words must be uint32[2]")
        if not bool(_update_lifetime_counter_valid(update_words, update_count)):
            raise ValueError("legacy online-gated update clock is corrupt")
    step_count = jnp.asarray(state_fields["step_count"])
    if step_count.shape != () or step_count.dtype != jnp.dtype(jnp.int32):
        raise TypeError("legacy online-gated step_count must be scalar int32")
    step = int(step_count)
    if step < 0:
        raise ValueError("negative legacy online-gated step_count indicates wrap")
    if step >= _INT32_MAX:
        raise ValueError("saturated legacy online-gated step_count is ambiguous")
    state_fields["step_words"] = jnp.asarray((0, step), dtype=jnp.uint32)
    return OnlineGatedStateBuilderState(**state_fields)


StateBuilderConfig = (
    IdentityStateBuilderConfig
    | FixedTraceStateBuilderConfig
    | OnlineGatedStateBuilderConfig
    | LearnableGRUStateBuilderConfig
    | RecurrentTraceUnitStateBuilderConfig
)


def state_builder_config_from_config(payload: dict[str, Any]) -> StateBuilderConfig:
    """Parse one known state-builder configuration without creating state."""
    builder_type = payload.get("type")
    if builder_type == "IdentityStateBuilder":
        return IdentityStateBuilderConfig.from_config(payload)
    if builder_type == "FixedTraceStateBuilder":
        return FixedTraceStateBuilderConfig.from_config(payload)
    if builder_type == "OnlineGatedStateBuilder":
        return OnlineGatedStateBuilderConfig.from_config(payload)
    if builder_type == "LearnableGRUStateBuilder":
        return LearnableGRUStateBuilderConfig.from_config(payload)
    if builder_type == "RecurrentTraceUnitStateBuilder":
        return RecurrentTraceUnitStateBuilderConfig.from_config(payload)
    raise ValueError(f"unknown state builder type: {builder_type!r}")


def state_builder_from_config(payload: dict[str, Any]) -> StateBuilder[Any]:
    """Construct a known state builder from its serialized configuration."""
    config = state_builder_config_from_config(payload)
    if isinstance(config, IdentityStateBuilderConfig):
        return IdentityStateBuilder(config)
    if isinstance(config, FixedTraceStateBuilderConfig):
        return FixedTraceStateBuilder(config)
    if isinstance(config, LearnableGRUStateBuilderConfig):
        return LearnableGRUStateBuilder(config)
    if isinstance(config, RecurrentTraceUnitStateBuilderConfig):
        return RecurrentTraceUnitStateBuilder(config)
    return OnlineGatedStateBuilder(config)


def _state_builder_config_digest(config: dict[str, Any]) -> str:
    """Return the canonical SHA-256 digest of an exact serialized config."""
    canonical = json.dumps(
        config,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def save_state_builder_checkpoint(
    builder: StateBuilder[Any],
    state: Any,
    path: str | Path,
) -> None:
    """Save a builder's full configuration and PyTree state."""
    if not bool(jax.device_get(builder.state_valid(state))):
        raise ValueError("refusing to save an invalid state-builder state")
    config = builder.to_config()
    save_checkpoint(
        state,
        path,
        metadata={
            "schema": STATE_BUILDER_CHECKPOINT_SCHEMA,
            "builder_config": config,
            "config_sha256": _state_builder_config_digest(config),
            "resource_budget": builder.resource_budget().to_config(),
        },
    )


def load_state_builder_checkpoint(
    path: str | Path,
    *,
    template_key: Array | None = None,
) -> tuple[StateBuilder[Any], Any]:
    """Restore a state builder without requiring a caller-constructed template."""
    metadata = load_checkpoint_metadata(path)
    checkpoint_schema = metadata.get("schema")
    config = metadata.get("builder_config")
    if not isinstance(config, dict):
        raise ValueError("state-builder checkpoint is missing builder_config")
    legacy_v3_online = (
        checkpoint_schema == "alberta.state_builder.v3"
        and config.get("type") == "OnlineGatedStateBuilder"
        and config.get("state_schema") == ONLINE_GATED_STATE_BUILDER_STATE_SCHEMA
    )
    if checkpoint_schema in {
        "alberta.state_builder.v1",
        "alberta.state_builder.v2",
    }:
        raise ValueError(
            "legacy state-builder checkpoints lack exact step/update clocks or "
            "current fixed-builder step words and "
            "cannot be restored automatically; migrate an unambiguous raw state "
            "with the builder-specific migration helper and resave it"
        )
    if checkpoint_schema == "alberta.state_builder.v3" and not legacy_v3_online:
        raise ValueError(
            "legacy identity/fixed-trace v3 checkpoints lack exact step words; "
            "migrate an unambiguous raw state with the builder-specific helper "
            "and resave it"
        )
    if checkpoint_schema != STATE_BUILDER_CHECKPOINT_SCHEMA and not legacy_v3_online:
        raise ValueError("checkpoint is not a current Alberta state-builder v4 checkpoint")
    expected_config_digest = _state_builder_config_digest(config)
    if metadata.get("config_sha256") != expected_config_digest:
        raise ValueError("state-builder checkpoint config digest does not match config")
    try:
        builder = state_builder_from_config(config)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "state-builder checkpoint config is not canonical for its builder"
        ) from error
    if _state_builder_config_digest(builder.to_config()) != expected_config_digest:
        raise ValueError("state-builder checkpoint config is not canonical for its builder")
    key = jr.key(0) if template_key is None else template_key
    template = builder.init(key)
    state, restored_metadata = load_checkpoint(template, path)
    if restored_metadata != metadata:
        raise ValueError("state-builder checkpoint metadata changed between reads")
    if restored_metadata.get("resource_budget") != builder.resource_budget().to_config():
        raise ValueError("state-builder checkpoint resource budget does not match config")
    if not bool(jax.device_get(builder.state_valid(state))):
        raise ValueError("state-builder checkpoint restored an invalid state")
    return builder, state


__all__ = [
    "FIXED_STATE_BUILDER_STEP_COUNTER_DELTA_NBYTES",
    "FIXED_STATE_BUILDER_STEP_COUNTER_NBYTES",
    "FIXED_TRACE_STATE_BUILDER_STATE_SCHEMA",
    "IDENTITY_STATE_BUILDER_STATE_SCHEMA",
    "LEARNABLE_GRU_STATE_BUILDER_STATE_SCHEMA",
    "ONLINE_GATED_STATE_BUILDER_STATE_SCHEMA",
    "ONLINE_GATED_STATE_BUILDER_STEP_COUNTER_DELTA_NBYTES",
    "ONLINE_GATED_STATE_BUILDER_STEP_COUNTER_NBYTES",
    "ONLINE_GATED_STATE_BUILDER_UPDATE_COUNTER_DELTA_NBYTES",
    "ONLINE_GATED_STATE_BUILDER_UPDATE_COUNTER_NBYTES",
    "RECURRENT_TRACE_UNIT_STATE_BUILDER_STATE_SCHEMA",
    "STATE_BUILDER_CHECKPOINT_SCHEMA",
    "FixedTraceStateBuilder",
    "FixedTraceStateBuilderConfig",
    "IdentityStateBuilder",
    "IdentityStateBuilderConfig",
    "IdentityStateBuilderState",
    "LearnableGRUStateBuilder",
    "LearnableGRUStateBuilderConfig",
    "LearnableGRUStateBuilderState",
    "LearnableGRUStateBuilderTransitionResult",
    "OnlineGatedStateBuilder",
    "OnlineGatedStateBuilderConfig",
    "OnlineGatedStateBuilderState",
    "OnlineGatedStateBuilderTransitionResult",
    "RecurrentTraceUnitStateBuilder",
    "RecurrentTraceUnitStateBuilderConfig",
    "RecurrentTraceUnitStateBuilderState",
    "RecurrentTraceUnitStateBuilderTransitionResult",
    "StateBuilder",
    "StateBuilderBudget",
    "StateBuilderConfig",
    "StateBuilderLearningDiagnostics",
    "StateBuilderLearningProposal",
    "fixed_state_builder_step_counter_nbytes",
    "load_state_builder_checkpoint",
    "measure_online_gated_state_builder_state_nbytes",
    "migrate_legacy_fixed_trace_state_builder_state",
    "migrate_legacy_identity_state_builder_state",
    "migrate_legacy_online_gated_state_builder_state",
    "online_gated_state_builder_step_counter_nbytes",
    "online_gated_state_builder_update_counter_nbytes",
    "replace_state_builder_learning_proposal_update",
    "save_state_builder_checkpoint",
    "state_builder_config_from_config",
    "state_builder_from_config",
]
