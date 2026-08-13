# mypy: disable-error-code="attr-defined,call-arg,no-any-return"
"""Bounded forward screening for the paper-defined Kondo compute gate.

In *Does This Gradient Spark Joy?*, delight is the forward-pass quantity

``advantage * surprisal`` where ``surprisal = -action_log_probability``.

The Kondo gate uses that quantity to propose which samples' actor-gradient
contributions should enter an expensive backward pass.  This forward-only
module does not establish that execution; an actor consumer must actually
differentiate the admitted rows before they ``spark joy``.  It does not reuse
the name for a generic update-quality score and it does not claim that masking
a full-batch loss saves compute.

The boundary is deliberately split in two:

1. :meth:`KondoGate.screen` computes a detached gate from forward-pass values.
2. :meth:`KondoGate.gather_sparse` creates a fixed-capacity PyTree, smaller
   than the input only when configured capacity is below batch size.

Autodiff must be invoked on the gathered PyTree by the caller.  This makes the
backward input shape auditable and avoids the common non-implementation in
which every sample remains in the backward graph behind a zero loss mask.

Two development modes are available:

``top_k_rate``
    Deterministic fixed-rate screening.  Its target count follows the official
    small token implementation's ``max(1, round(rate * valid_count))`` rule,
    while Alberta deliberately resolves exact delight ties by lowest source
    index so the sparse capacity is never exceeded by threshold ties.

``bernoulli_price``
    The paper's finite-temperature gate
    ``Bernoulli(sigmoid((delight - price) / temperature))``.  If more samples
    are selected than the declared sparse capacity, the result requires an
    explicitly reported full-shape masked backward rather than silently
    dropping selected samples.

Caller-declared ``force_keep_mask`` values are never screened out.  They are
intended for independently detected safety/model-change events.  If they do
not fit the sparse capacity, the boundary similarly requires the full-shape
fallback.  This is L0 mechanism code, not evidence of wall-clock savings,
learning benefit, calibration, or safe deployment.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any, Literal, TypeVar, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

KONDO_GATE_SCHEMA = "alberta.kondo-gate.v2"
KONDO_GATE_LEGACY_V1_SCHEMA = "alberta.kondo-gate.v1"
KondoGateMode = Literal["top_k_rate", "bernoulli_price"]

_BACKWARD_ADMISSION_INTENT_SEMANTICS = (
    "detached-forward-selection-not-backward-execution"
)

_INT32_MAX = 2_147_483_647
_FLOAT32_MAX = float(np.finfo(np.float32).max)
_FLOAT32_TINY = float(np.finfo(np.float32).tiny)
_STATE_FIELDS = {
    "rng_key_data",
    "screen_count",
    "forward_slots_screened",
    "valid_examples_screened",
    "examples_selected",
    "sparse_batch_count",
    "full_fallback_count",
}


def _finite_float32_config(name: str, value: object, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number) or abs(number) > _FLOAT32_MAX:
        raise ValueError(f"{name} must be finite and representable as float32")
    if number != 0.0 and abs(number) < _FLOAT32_TINY:
        raise ValueError(f"{name} must be zero or a normal float32 value")
    if positive and number < _FLOAT32_TINY:
        raise ValueError(f"{name} must be a positive normal float32 value")
    return float(np.float32(number))


def _require_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> None:
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must expose array shape and dtype")
    actual_shape = tuple(cast(Any, value).shape)
    if actual_shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {actual_shape}")
    actual_dtype = jnp.dtype(cast(Any, value).dtype)
    expected_dtype = jnp.dtype(dtype)
    if actual_dtype != expected_dtype:
        raise TypeError(f"{name} must have dtype {expected_dtype}; got {actual_dtype}")


def _typed_key_valid(key: object) -> bool:
    if not (
        hasattr(key, "shape")
        and tuple(cast(Any, key).shape) == ()
        and jax.dtypes.issubdtype(cast(Any, key).dtype, jax.dtypes.prng_key)
    ):
        return False
    try:
        return (
            str(jr.key_impl(cast(Any, key))) == "threefry2x32"
            and tuple(jr.key_data(cast(Any, key)).shape) == (2,)
        )
    except (TypeError, ValueError):
        return False


@dataclasses.dataclass(frozen=True)
class KondoGateConfig:
    """Static screen and sparse-backward allocation.

    ``target_rate`` is used only by ``top_k_rate``.  Its fixed sparse allocation
    is ``max(1, round(target_rate * batch_size))`` using float32 ties-to-even,
    while the dynamic target uses the same rule on the current valid count and
    pads unused slots. ``sparse_capacity`` is used only by ``bernoulli_price``
    because a Bernoulli gate has a variable number of survivors.

    ``minimum_uniform_keep`` reserves a deterministic-size random subset as an
    independently diagnosed guardrail. In top-k mode it occupies fixed target
    slots where possible and expands the dynamic target only when necessary;
    in Bernoulli mode it is unioned with the sampled gate. It defaults to zero
    so the paper equations are exact. A nonzero value is an explicit continual-
    learning extension, not part of the paper's Kondo definition.

    Canonical v2 payloads call the forward mask ``backward_admission_intent``.
    The frozen v1 ``sparks_joy_semantics`` field is accepted only by the
    explicit legacy import path and is always normalized to v2 on emission.
    Only an executing actor consumer can establish that a gradient sparked joy.
    """

    batch_size: int
    mode: KondoGateMode = "top_k_rate"
    target_rate: float = 0.03
    sparse_capacity: int | None = None
    price: float = 0.0
    temperature: float = 1.0
    minimum_uniform_keep: int = 0
    max_screenings: int = 1_000_000

    def __post_init__(self) -> None:
        for name in ("batch_size", "minimum_uniform_keep", "max_screenings"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if type(self.mode) is not str or self.mode not in (
            "top_k_rate",
            "bernoulli_price",
        ):
            raise ValueError("mode must be 'top_k_rate' or 'bernoulli_price'")
        target_rate = _finite_float32_config("target_rate", self.target_rate)
        if not 0.0 < target_rate <= 1.0:
            raise ValueError("target_rate must be in (0, 1]")
        price = _finite_float32_config("price", self.price)
        if price < 0.0:
            raise ValueError("price must be nonnegative")
        if price == 0.0:
            price = 0.0
        temperature = _finite_float32_config(
            "temperature", self.temperature, positive=True
        )
        object.__setattr__(self, "target_rate", target_rate)
        object.__setattr__(self, "price", price)
        object.__setattr__(self, "temperature", temperature)
        if self.minimum_uniform_keep < 0:
            raise ValueError("minimum_uniform_keep must be nonnegative")
        if self.max_screenings <= 0:
            raise ValueError("max_screenings must be positive")
        if self.sparse_capacity is not None:
            if isinstance(self.sparse_capacity, bool) or not isinstance(
                self.sparse_capacity, int
            ):
                raise ValueError("sparse_capacity must be an integer or None")
            if not 1 <= self.sparse_capacity <= self.batch_size:
                raise ValueError("sparse_capacity must be in [1, batch_size]")
        if self.mode == "top_k_rate" and self.sparse_capacity is not None:
            raise ValueError("top_k_rate derives capacity; sparse_capacity must be None")
        if self.mode == "bernoulli_price" and self.sparse_capacity is None:
            raise ValueError("bernoulli_price requires sparse_capacity")
        if self.minimum_uniform_keep > self.backward_capacity:
            raise ValueError("minimum_uniform_keep cannot exceed backward capacity")
        if self.max_screenings > _INT32_MAX // self.batch_size:
            raise ValueError("max_screenings * batch_size exceeds int32 accounting")

    @property
    def backward_capacity(self) -> int:
        """Maximum examples accepted by the sparse backward shape."""
        if self.mode == "bernoulli_price":
            return cast(int, self.sparse_capacity)
        count = int(
            np.rint(
                np.float32(self.target_rate) * np.float32(self.batch_size)
            )
        )
        return max(1, min(self.batch_size, count))

    def _to_config_for_schema(self, schema: str) -> dict[str, object]:
        payload = dataclasses.asdict(self)
        payload["schema"] = schema
        payload["type"] = "KondoGateConfig"
        payload["backward_capacity"] = self.backward_capacity
        payload["delight_semantics"] = "advantage-times-action-surprisal"
        if schema == KONDO_GATE_SCHEMA:
            payload["backward_admission_intent_semantics"] = (
                _BACKWARD_ADMISSION_INTENT_SEMANTICS
            )
        elif schema == KONDO_GATE_LEGACY_V1_SCHEMA:
            payload["sparks_joy_semantics"] = "selected-for-backward-pass"
        else:  # pragma: no cover - private invariant
            raise ValueError("unsupported Kondo gate config schema")
        payload["generic_gradient_quality_audit"] = False
        payload["wall_clock_savings_claimed"] = False
        return payload

    def to_config(self) -> dict[str, object]:
        """Emit only the canonical v2 backward-admission-intent surface."""
        return self._to_config_for_schema(KONDO_GATE_SCHEMA)

    @classmethod
    def from_config(cls, payload: Mapping[str, object]) -> KondoGateConfig:
        schema = payload.get("schema")
        if type(schema) is not str:
            raise ValueError("Kondo gate config schema is invalid")
        if schema == KONDO_GATE_SCHEMA:
            version = "v2"
            semantic_field = "backward_admission_intent_semantics"
            semantic_value = _BACKWARD_ADMISSION_INTENT_SEMANTICS
        elif schema == KONDO_GATE_LEGACY_V1_SCHEMA:
            version = "legacy v1"
            semantic_field = "sparks_joy_semantics"
            semantic_value = "selected-for-backward-pass"
        else:
            raise ValueError("Kondo gate config schema is invalid")
        expected = {
            "batch_size",
            "mode",
            "target_rate",
            "sparse_capacity",
            "price",
            "temperature",
            "minimum_uniform_keep",
            "max_screenings",
            "schema",
            "type",
            "backward_capacity",
            "delight_semantics",
            semantic_field,
            "generic_gradient_quality_audit",
            "wall_clock_savings_claimed",
        }
        if set(payload) != expected:
            raise ValueError(f"Kondo gate config fields do not match {version}")
        fixed: dict[str, object] = {
            "schema": schema,
            "type": "KondoGateConfig",
            "delight_semantics": "advantage-times-action-surprisal",
            semantic_field: semantic_value,
            "generic_gradient_quality_audit": False,
            "wall_clock_savings_claimed": False,
        }
        for name, value in fixed.items():
            if type(payload.get(name)) is not type(value) or payload.get(name) != value:
                raise ValueError(f"Kondo gate {version} {name} is invalid")
        integer_fields = ("batch_size", "minimum_uniform_keep", "max_screenings")
        for name in integer_fields:
            if type(payload[name]) is not int:
                raise ValueError(f"Kondo gate {name} must be an integer")
        if payload["sparse_capacity"] is not None and type(
            payload["sparse_capacity"]
        ) is not int:
            raise ValueError("Kondo gate sparse_capacity must be an integer or null")
        for name in ("target_rate", "price", "temperature"):
            if type(payload[name]) is not float:
                raise ValueError(f"Kondo gate {name} must be a float")
        if type(payload["mode"]) is not str:
            raise ValueError("Kondo gate mode must be a string")
        result = cls(
            batch_size=cast(int, payload["batch_size"]),
            mode=cast(KondoGateMode, payload["mode"]),
            target_rate=cast(float, payload["target_rate"]),
            sparse_capacity=payload["sparse_capacity"],
            price=cast(float, payload["price"]),
            temperature=cast(float, payload["temperature"]),
            minimum_uniform_keep=cast(int, payload["minimum_uniform_keep"]),
            max_screenings=cast(int, payload["max_screenings"]),
        )
        if type(payload["backward_capacity"]) is not int or (
            payload["backward_capacity"] != result.backward_capacity
        ):
            raise ValueError("Kondo gate backward_capacity is noncanonical")
        if result._to_config_for_schema(schema) != dict(payload):
            raise ValueError("Kondo gate config is noncanonical")
        return result


@chex.dataclass(frozen=True)
class KondoGateState:
    """RNG ownership and exact logical accounting for forward screening.

    ``sparse_batch_count`` and ``full_fallback_count`` are frozen checkpoint
    fields counting the gate's planned screen outcomes. They do not certify
    actor-backward execution; the consuming actor reports that boundary.
    """

    rng_key: Array
    screen_count: Int[Array, ""]
    forward_slots_screened: Int[Array, ""]
    valid_examples_screened: Int[Array, ""]
    examples_selected: Int[Array, ""]
    sparse_batch_count: Int[Array, ""]
    full_fallback_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class KondoGateResult:
    """Detached forward gate and fixed sparse-gather plan."""

    state: KondoGateState
    screen_rng_key: Array
    config_id: UInt[Array, " 8"]
    delight: Float[Array, " batch"]
    action_surprisal: Float[Array, " batch"]
    gate_probability: Float[Array, " batch"]
    selected_by_delight_gate: Bool[Array, " batch"]
    uniformly_reserved: Bool[Array, " batch"]
    force_keep_mask: Bool[Array, " batch"]
    valid_mask: Bool[Array, " batch"]
    selected_mask: Bool[Array, " batch"]
    selected_indices: Int[Array, " capacity"]
    selected_slot_mask: Bool[Array, " capacity"]
    selected_count: Int[Array, ""]
    valid_count: Int[Array, ""]
    effective_price: Float[Array, ""]
    input_valid: Bool[Array, ""]
    state_valid: Bool[Array, ""]
    capacity_sufficient: Bool[Array, ""]
    sparse_backward_available: Bool[Array, ""]
    full_shape_masked_backward_required: Bool[Array, ""]
    random_draw_count: Int[Array, ""]
    transaction_applied: Bool[Array, ""]

    @property
    def backward_admission_intent(self) -> Bool[Array, " batch"]:
        """Forward plan for rows a caller may admit to an actor backward."""
        return self.selected_mask

    @property
    def sparks_joy(self) -> Bool[Array, " batch"]:
        """Compatibility shorthand for :attr:`backward_admission_intent`.

        This forward-only result does not establish that a gradient sparked
        joy because it does not itself execute autodiff.  The
        :mod:`kondo_sparse_actor` integration is the boundary that establishes
        that the admitted rows entered an actual actor backward pass.
        """
        return self.backward_admission_intent


@chex.dataclass(frozen=True)
class KondoSparseBatch:
    """A fixed-capacity PyTree on which the caller may perform autodiff.

    ``sample_mask`` is part of the loss contract and must be applied. When the
    dynamic survivor count is below capacity, inactive rows contain dummy
    gathered values and do not represent selected samples.
    """

    data: Any
    sample_mask: Bool[Array, " capacity"]
    source_indices: Int[Array, " capacity"]


@dataclasses.dataclass(frozen=True)
class KondoGateResourceDeclaration:
    """Exact logical allocation; not a measured FLOP or latency claim."""

    forward_slots_per_screen: int
    sparse_backward_capacity: int
    persistent_state_bytes: int
    maximum_random_draws_per_screen: int
    maximum_delight_products_per_screen: int
    full_shape_fallback_possible: bool
    wall_clock_savings_claimed: bool = False

    def to_config(self) -> dict[str, int | bool]:
        return dataclasses.asdict(self)


BatchTree = TypeVar("BatchTree")


class KondoGate:
    """Fixed-batch paper Kondo screen with a config-bound gather boundary."""

    def __init__(self, config: KondoGateConfig):
        if not isinstance(config, KondoGateConfig):
            raise TypeError("config must be KondoGateConfig")
        self._config = config
        config_bytes = json.dumps(
            config.to_config(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest = hashlib.sha256(config_bytes).digest()
        self._config_id = jnp.asarray(
            [
                int.from_bytes(digest[offset : offset + 4], "big")
                for offset in range(0, len(digest), 4)
            ],
            dtype=jnp.uint32,
        )

    @property
    def config(self) -> KondoGateConfig:
        return self._config

    def to_config(self) -> dict[str, object]:
        return self._config.to_config()

    @classmethod
    def from_config(cls, payload: Mapping[str, object]) -> KondoGate:
        return cls(KondoGateConfig.from_config(payload))

    def init(self, key: Array) -> KondoGateState:
        if not _typed_key_valid(key):
            raise TypeError("key must be a typed scalar threefry2x32 JAX PRNG key")
        zero = jnp.asarray(0, dtype=jnp.int32)
        return KondoGateState(
            rng_key=key,
            screen_count=zero,
            forward_slots_screened=zero,
            valid_examples_screened=zero,
            examples_selected=zero,
            sparse_batch_count=zero,
            full_fallback_count=zero,
        )

    def resource_declaration(
        self,
        state: KondoGateState | None = None,
    ) -> KondoGateResourceDeclaration:
        sample = (
            self.init(jr.key(0, impl="threefry2x32"))
            if state is None
            else state
        )
        self._validate_state_static(sample)
        state_bytes = sum(
            int(cast(Any, leaf).nbytes) for leaf in jax.tree_util.tree_leaves(sample)
        )
        random_streams = int(self._config.mode == "bernoulli_price") + int(
            self._config.minimum_uniform_keep > 0
        )
        random_draws = random_streams * self._config.batch_size
        return KondoGateResourceDeclaration(
            forward_slots_per_screen=self._config.batch_size,
            sparse_backward_capacity=self._config.backward_capacity,
            persistent_state_bytes=state_bytes,
            maximum_random_draws_per_screen=random_draws,
            maximum_delight_products_per_screen=self._config.batch_size,
            full_shape_fallback_possible=(
                self._config.backward_capacity < self._config.batch_size
            ),
        )

    def _validate_state_static(self, state: KondoGateState) -> None:
        if not isinstance(state, KondoGateState):
            raise TypeError("state must be KondoGateState")
        if not _typed_key_valid(state.rng_key):
            raise TypeError(
                "state.rng_key must be a typed scalar threefry2x32 JAX PRNG key"
            )
        for name in (
            "screen_count",
            "forward_slots_screened",
            "valid_examples_screened",
            "examples_selected",
            "sparse_batch_count",
            "full_fallback_count",
        ):
            _require_array(
                getattr(state, name), name=f"state.{name}", shape=(), dtype=jnp.int32
            )

    def _validate_result_static(self, result: KondoGateResult) -> None:
        """Reject result shape/dtype drift before host sparse orchestration."""
        if not isinstance(result, KondoGateResult):
            raise TypeError("result must be KondoGateResult")
        self._validate_state_static(result.state)
        if not _typed_key_valid(result.screen_rng_key):
            raise TypeError(
                "result.screen_rng_key must be a typed scalar threefry2x32 key"
            )
        batch_shape = (self._config.batch_size,)
        capacity_shape = (self._config.backward_capacity,)
        _require_array(
            result.config_id,
            name="result.config_id",
            shape=(8,),
            dtype=jnp.uint32,
        )
        for name in (
            "delight",
            "action_surprisal",
            "gate_probability",
        ):
            _require_array(
                getattr(result, name),
                name=f"result.{name}",
                shape=batch_shape,
                dtype=jnp.float32,
            )
        for name in (
            "selected_by_delight_gate",
            "uniformly_reserved",
            "force_keep_mask",
            "valid_mask",
            "selected_mask",
        ):
            _require_array(
                getattr(result, name),
                name=f"result.{name}",
                shape=batch_shape,
                dtype=jnp.bool_,
            )
        _require_array(
            result.selected_indices,
            name="result.selected_indices",
            shape=capacity_shape,
            dtype=jnp.int32,
        )
        _require_array(
            result.selected_slot_mask,
            name="result.selected_slot_mask",
            shape=capacity_shape,
            dtype=jnp.bool_,
        )
        for name in ("selected_count", "valid_count", "random_draw_count"):
            _require_array(
                getattr(result, name),
                name=f"result.{name}",
                shape=(),
                dtype=jnp.int32,
            )
        _require_array(
            result.effective_price,
            name="result.effective_price",
            shape=(),
            dtype=jnp.float32,
        )
        for name in (
            "input_valid",
            "state_valid",
            "capacity_sufficient",
            "sparse_backward_available",
            "full_shape_masked_backward_required",
            "transaction_applied",
        ):
            _require_array(
                getattr(result, name),
                name=f"result.{name}",
                shape=(),
                dtype=jnp.bool_,
            )

    def _state_valid(self, state: KondoGateState) -> Array:
        cfg = self._config
        return (
            (state.screen_count >= 0)
            & (state.screen_count <= cfg.max_screenings)
            & (
                state.forward_slots_screened
                == state.screen_count * cfg.batch_size
            )
            & (state.valid_examples_screened >= 0)
            & (state.valid_examples_screened <= state.forward_slots_screened)
            & (state.examples_selected >= 0)
            & (state.examples_selected <= state.valid_examples_screened)
            & (state.sparse_batch_count >= 0)
            & (state.full_fallback_count >= 0)
            & (
                state.sparse_batch_count + state.full_fallback_count
                == state.screen_count
            )
        )

    def screen(
        self,
        state: KondoGateState,
        advantage: Float[Array, " batch"],
        action_log_probability: Float[Array, " batch"],
        valid_mask: Bool[Array, " batch"],
        force_keep_mask: Bool[Array, " batch"],
    ) -> KondoGateResult:
        """Compute delight and select samples before any backward pass.

        ``force_keep_mask`` must be a subset of ``valid_mask``.  It is never a
        delight input and is reported independently.
        """

        self._validate_state_static(state)
        batch_shape = (self._config.batch_size,)
        _require_array(advantage, name="advantage", shape=batch_shape, dtype=jnp.float32)
        _require_array(
            action_log_probability,
            name="action_log_probability",
            shape=batch_shape,
            dtype=jnp.float32,
        )
        _require_array(valid_mask, name="valid_mask", shape=batch_shape, dtype=jnp.bool_)
        _require_array(
            force_keep_mask,
            name="force_keep_mask",
            shape=batch_shape,
            dtype=jnp.bool_,
        )
        return cast(
            KondoGateResult,
            self._screen_jit(
                state,
                advantage,
                action_log_probability,
                valid_mask,
                force_keep_mask,
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _screen_jit(
        self,
        state: KondoGateState,
        advantage: Array,
        action_log_probability: Array,
        valid_mask: Array,
        force_keep_mask: Array,
    ) -> KondoGateResult:
        cfg = self._config
        state_valid = self._state_valid(state)
        base_input_valid = (
            jnp.all(jnp.isfinite(advantage))
            & jnp.all(jnp.isfinite(action_log_probability))
            & jnp.all(action_log_probability <= 0.0)
            & jnp.all(~force_keep_mask | valid_mask)
        )
        capacity_available = state.screen_count < cfg.max_screenings

        safe_advantage = jnp.where(base_input_valid & valid_mask, advantage, 0.0)
        safe_log_probability = jnp.where(
            base_input_valid & valid_mask,
            action_log_probability,
            0.0,
        )
        candidate_surprisal = -safe_log_probability
        candidate_delight = safe_advantage * candidate_surprisal
        derived_finite = jnp.all(jnp.isfinite(candidate_delight))
        input_valid = base_input_valid & derived_finite
        transaction_allowed = state_valid & input_valid & capacity_available
        action_surprisal = jax.lax.stop_gradient(
            jnp.where(transaction_allowed, candidate_surprisal, 0.0)
        )
        delight = jax.lax.stop_gradient(
            jnp.where(transaction_allowed, candidate_delight, 0.0)
        )
        valid_count = jnp.sum(valid_mask.astype(jnp.int32))

        needs_gate_random = cfg.mode == "bernoulli_price"
        needs_reserve_random = cfg.minimum_uniform_keep > 0
        random_stream_count = int(needs_gate_random) + int(needs_reserve_random)
        if random_stream_count == 2:
            next_key, gate_draw_key, reserve_draw_key = jr.split(state.rng_key, 3)
        elif random_stream_count == 1:
            next_key, single_draw_key = jr.split(state.rng_key)
            gate_draw_key = single_draw_key
            reserve_draw_key = single_draw_key
        else:
            next_key = state.rng_key
            gate_draw_key = state.rng_key
            reserve_draw_key = state.rng_key
        if needs_gate_random:
            gate_uniforms = jr.uniform(
                gate_draw_key,
                shape=(cfg.batch_size,),
                dtype=jnp.float32,
                minval=0.0,
                maxval=1.0,
            )
        else:
            gate_uniforms = jnp.zeros((cfg.batch_size,), dtype=jnp.float32)
        if needs_reserve_random:
            reserve_uniforms = jr.uniform(
                reserve_draw_key,
                shape=(cfg.batch_size,),
                dtype=jnp.float32,
                minval=0.0,
                maxval=1.0,
            )
        else:
            reserve_uniforms = jnp.zeros((cfg.batch_size,), dtype=jnp.float32)
        random_draw_count = jnp.asarray(
            random_stream_count * cfg.batch_size,
            dtype=jnp.int32,
        )

        if cfg.mode == "bernoulli_price":
            effective_price = jnp.asarray(cfg.price, dtype=jnp.float32)
            temperature = jnp.asarray(cfg.temperature, dtype=jnp.float32)
            raw_gate_probability = jax.nn.sigmoid(
                (delight - effective_price) / temperature
            )
            gate_probability = jnp.where(
                valid_mask,
                raw_gate_probability,
                jnp.asarray(0.0, dtype=jnp.float32),
            )
            selected_by_delight_gate = valid_mask & (
                gate_uniforms < gate_probability
            )
        else:
            capacity = cfg.backward_capacity
            finite_scores = jnp.where(valid_mask, delight, -jnp.inf)
            sorted_valid_scores = jnp.sort(finite_scores)[::-1]
            target_count = jnp.maximum(
                jnp.asarray(1, dtype=jnp.int32),
                jnp.round(
                    jnp.asarray(cfg.target_rate, dtype=jnp.float32)
                    * valid_count.astype(jnp.float32)
                ).astype(jnp.int32),
            )
            price_index = jnp.clip(
                jnp.minimum(valid_count, target_count) - 1,
                0,
                cfg.batch_size - 1,
            )
            effective_price = jnp.where(
                valid_count > 0,
                sorted_valid_scores[price_index],
                jnp.asarray(0.0, dtype=jnp.float32),
            )
            delight_order = jnp.argsort(-finite_scores, stable=True)
            delight_indices = delight_order[:capacity]
            delight_slot_mask = valid_mask[delight_indices] & (
                jnp.arange(capacity, dtype=jnp.int32) < target_count
            )
            selected_by_delight_gate = jnp.zeros(
                (cfg.batch_size,), dtype=jnp.bool_
            ).at[delight_indices].set(delight_slot_mask)
            gate_probability = selected_by_delight_gate.astype(jnp.float32)

        random_order = jnp.argsort(
            jnp.where(valid_mask, reserve_uniforms, jnp.inf),
            stable=True,
        )
        random_rank = jnp.zeros((cfg.batch_size,), dtype=jnp.int32).at[
            random_order
        ].set(jnp.arange(cfg.batch_size, dtype=jnp.int32))
        uniformly_reserved = (
            valid_mask
            & (random_rank < jnp.asarray(cfg.minimum_uniform_keep, dtype=jnp.int32))
        )
        reserved = valid_mask & (force_keep_mask | uniformly_reserved)
        capacity = cfg.backward_capacity

        if cfg.mode == "top_k_rate":
            # Stable sort by delight first, then move reserved samples ahead
            # without changing within-class order. Exact ties therefore retain
            # ascending source-index order.
            delight_order = jnp.argsort(
                jnp.where(valid_mask, -delight, jnp.inf), stable=True
            )
            reserved_in_order = reserved[delight_order]
            valid_in_order = valid_mask[delight_order]
            class_key = jnp.where(
                reserved_in_order,
                0,
                jnp.where(valid_in_order, 1, 2),
            )
            class_order = jnp.argsort(class_key, stable=True)
            ordered_indices = delight_order[class_order]
            selected_indices_raw = ordered_indices[:capacity]
            reserved_count = jnp.sum(reserved.astype(jnp.int32))
            selected_target_count = jnp.maximum(target_count, reserved_count)
            slot_mask = valid_mask[selected_indices_raw] & (
                jnp.arange(capacity, dtype=jnp.int32) < selected_target_count
            )
            sparse_selected_mask = jnp.zeros((cfg.batch_size,), dtype=jnp.bool_).at[
                selected_indices_raw
            ].set(slot_mask)
            capacity_sufficient = reserved_count <= capacity
            # If caller-forced samples exceed sparse capacity, preserve every
            # forced/reserved selection and require the full-shape fallback.
            selected_mask = jnp.where(
                capacity_sufficient,
                sparse_selected_mask,
                selected_by_delight_gate | reserved,
            )
        else:
            selected_mask = valid_mask & (selected_by_delight_gate | reserved)
            selected_count_raw = jnp.sum(selected_mask.astype(jnp.int32))
            capacity_sufficient = selected_count_raw <= capacity
            selected_order = jnp.argsort(~selected_mask, stable=True)
            selected_indices_raw = selected_order[:capacity]
            slot_mask = selected_mask[selected_indices_raw]

        selected_count = jnp.sum(selected_mask.astype(jnp.int32))
        sparse_available = transaction_allowed & capacity_sufficient
        full_fallback = transaction_allowed & ~capacity_sufficient
        selected_indices = jnp.where(
            sparse_available & slot_mask,
            selected_indices_raw,
            jnp.zeros_like(selected_indices_raw),
        ).astype(jnp.int32)
        selected_slot_mask = jnp.where(
            sparse_available,
            slot_mask,
            jnp.zeros_like(slot_mask),
        )
        applied_selected_count = jnp.where(
            transaction_allowed,
            selected_count,
            jnp.asarray(0, dtype=jnp.int32),
        )
        committed_key = jax.lax.cond(
            transaction_allowed,
            lambda _: next_key,
            lambda _: state.rng_key,
            operand=None,
        )
        next_state = KondoGateState(
            rng_key=committed_key,
            screen_count=state.screen_count + transaction_allowed.astype(jnp.int32),
            forward_slots_screened=(
                state.forward_slots_screened
                + transaction_allowed.astype(jnp.int32) * cfg.batch_size
            ),
            valid_examples_screened=(
                state.valid_examples_screened
                + transaction_allowed.astype(jnp.int32) * valid_count
            ),
            examples_selected=state.examples_selected + applied_selected_count,
            sparse_batch_count=(
                state.sparse_batch_count + sparse_available.astype(jnp.int32)
            ),
            full_fallback_count=(
                state.full_fallback_count + full_fallback.astype(jnp.int32)
            ),
        )
        return KondoGateResult(
            state=next_state,
            screen_rng_key=state.rng_key,
            config_id=self._config_id,
            delight=jnp.where(transaction_allowed, delight, 0.0),
            action_surprisal=jnp.where(transaction_allowed, action_surprisal, 0.0),
            gate_probability=jnp.where(transaction_allowed, gate_probability, 0.0),
            selected_by_delight_gate=jnp.where(
                transaction_allowed,
                selected_by_delight_gate,
                False,
            ),
            uniformly_reserved=jnp.where(
                transaction_allowed, uniformly_reserved, False
            ),
            force_keep_mask=force_keep_mask,
            valid_mask=valid_mask,
            selected_mask=jnp.where(transaction_allowed, selected_mask, False),
            selected_indices=selected_indices,
            selected_slot_mask=selected_slot_mask,
            selected_count=applied_selected_count,
            valid_count=jnp.where(transaction_allowed, valid_count, 0),
            effective_price=jnp.where(transaction_allowed, effective_price, 0.0),
            input_valid=input_valid,
            state_valid=state_valid,
            capacity_sufficient=capacity_sufficient,
            sparse_backward_available=sparse_available,
            full_shape_masked_backward_required=full_fallback,
            random_draw_count=jnp.where(transaction_allowed, random_draw_count, 0),
            transaction_applied=transaction_allowed,
        )

    def gather_sparse(
        self,
        batch: BatchTree,
        result: KondoGateResult,
    ) -> KondoSparseBatch:
        """Gather the fixed-capacity PyTree selected for autodiff.

        This is a host orchestration boundary and is intentionally not JIT or
        scan compatible. A result that requires the caller-managed full-shape
        fallback is rejected rather than accidentally presented as a sparse
        compute saving. The returned ``sample_mask`` must be applied by the
        caller's loss because unused fixed-capacity slots contain dummy rows.
        """

        self._validate_result_static(result)
        if not np.array_equal(np.asarray(result.config_id), np.asarray(self._config_id)):
            raise ValueError("Kondo result belongs to a different gate configuration")
        if not bool(np.asarray(result.transaction_applied)):
            raise ValueError("cannot gather from a rejected Kondo transaction")
        transaction_applied = bool(np.asarray(result.transaction_applied))
        capacity_sufficient = bool(np.asarray(result.capacity_sufficient))
        sparse_backward_available = bool(
            np.asarray(result.sparse_backward_available)
        )
        full_shape_required = bool(
            np.asarray(result.full_shape_masked_backward_required)
        )
        if sparse_backward_available != (
            transaction_applied and capacity_sufficient
        ) or full_shape_required != (
            transaction_applied and not capacity_sufficient
        ):
            raise ValueError("Kondo sparse/full-shape decision is inconsistent")
        if not sparse_backward_available:
            raise ValueError("Kondo result requires a full-shape masked backward")
        if (
            not bool(np.asarray(result.input_valid))
            or not bool(np.asarray(result.state_valid))
            or not capacity_sufficient
            or full_shape_required
            or not bool(np.asarray(self._state_valid(result.state)))
        ):
            raise ValueError("Kondo sparse result invariants are invalid")
        needs_gate_random = self._config.mode == "bernoulli_price"
        needs_reserve_random = self._config.minimum_uniform_keep > 0
        random_stream_count = int(needs_gate_random) + int(needs_reserve_random)
        if random_stream_count == 2:
            expected_next_key, gate_draw_key, reserve_draw_key = jr.split(
                result.screen_rng_key,
                3,
            )
        elif random_stream_count == 1:
            expected_next_key, single_draw_key = jr.split(result.screen_rng_key)
            gate_draw_key = single_draw_key
            reserve_draw_key = single_draw_key
        else:
            expected_next_key = result.screen_rng_key
            gate_draw_key = result.screen_rng_key
            reserve_draw_key = result.screen_rng_key
        if not np.array_equal(
            np.asarray(jr.key_data(result.state.rng_key)),
            np.asarray(jr.key_data(expected_next_key)),
        ):
            raise ValueError("Kondo post-screen RNG key is inconsistent")
        prior_state = KondoGateState(
            rng_key=result.state.rng_key,
            screen_count=(
                result.state.screen_count - jnp.asarray(1, dtype=jnp.int32)
            ),
            forward_slots_screened=(
                result.state.forward_slots_screened
                - jnp.asarray(self._config.batch_size, dtype=jnp.int32)
            ),
            valid_examples_screened=(
                result.state.valid_examples_screened - result.valid_count
            ),
            examples_selected=(
                result.state.examples_selected - result.selected_count
            ),
            sparse_batch_count=(
                result.state.sparse_batch_count
                - jnp.asarray(1, dtype=jnp.int32)
            ),
            full_fallback_count=result.state.full_fallback_count,
        )
        if not bool(np.asarray(self._state_valid(prior_state))):
            raise ValueError("Kondo result has no valid predecessor accounting")
        indices = np.asarray(result.selected_indices, dtype=np.int64)
        slot_mask = np.asarray(result.selected_slot_mask, dtype=np.bool_)
        selected_mask = np.asarray(result.selected_mask, dtype=np.bool_)
        selected_by_delight_gate = np.asarray(
            result.selected_by_delight_gate,
            dtype=np.bool_,
        )
        uniformly_reserved = np.asarray(result.uniformly_reserved, dtype=np.bool_)
        force_keep_mask = np.asarray(result.force_keep_mask, dtype=np.bool_)
        valid_mask = np.asarray(result.valid_mask, dtype=np.bool_)
        selected_count = int(np.asarray(result.selected_count))
        valid_count = int(np.asarray(result.valid_count))
        random_draw_count = int(np.asarray(result.random_draw_count))
        expected_random_draw_count = self._config.batch_size * (
            random_stream_count
        )
        if (
            np.any(indices < 0)
            or np.any(indices >= self._config.batch_size)
            or selected_count < 0
            or valid_count < 0
            or selected_count > valid_count
            or valid_count > self._config.batch_size
            or int(np.sum(valid_mask)) != valid_count
            or int(np.sum(slot_mask)) != selected_count
            or int(np.sum(selected_mask)) != selected_count
            or np.any(indices[~slot_mask] != 0)
            or np.any(selected_by_delight_gate & ~valid_mask)
            or np.any(force_keep_mask & ~valid_mask)
            or np.any(uniformly_reserved & ~valid_mask)
            or np.any(selected_mask & ~valid_mask)
            or np.any(force_keep_mask & ~selected_mask)
            or np.any(uniformly_reserved & ~selected_mask)
            or random_draw_count != expected_random_draw_count
            or int(np.asarray(result.state.screen_count)) <= 0
            or int(np.asarray(result.state.sparse_batch_count)) <= 0
            or int(np.asarray(result.state.examples_selected)) < selected_count
            or int(np.asarray(result.state.valid_examples_screened)) < valid_count
            or not np.all(np.isfinite(np.asarray(result.delight)))
            or not np.all(np.isfinite(np.asarray(result.action_surprisal)))
            or not np.all(np.isfinite(np.asarray(result.gate_probability)))
            or not np.isfinite(float(np.asarray(result.effective_price)))
            or np.any(np.asarray(result.action_surprisal) < 0.0)
            or np.any(np.asarray(result.gate_probability) < 0.0)
            or np.any(np.asarray(result.gate_probability) > 1.0)
            or np.any(np.asarray(result.delight)[~valid_mask] != 0.0)
            or np.any(np.asarray(result.action_surprisal)[~valid_mask] != 0.0)
            or np.any(np.asarray(result.gate_probability)[~valid_mask] != 0.0)
        ):
            raise ValueError("Kondo sparse selection accounting is invalid")
        reserved = force_keep_mask | uniformly_reserved
        if needs_reserve_random:
            reserve_uniforms = np.asarray(
                jr.uniform(
                    reserve_draw_key,
                    shape=(self._config.batch_size,),
                    dtype=jnp.float32,
                    minval=0.0,
                    maxval=1.0,
                )
            )
            reserve_order = np.argsort(
                np.where(valid_mask, reserve_uniforms, np.inf),
                kind="stable",
            )
            reserve_rank = np.zeros(
                (self._config.batch_size,), dtype=np.int32
            )
            reserve_rank[reserve_order] = np.arange(
                self._config.batch_size,
                dtype=np.int32,
            )
            expected_uniformly_reserved = valid_mask & (
                reserve_rank < self._config.minimum_uniform_keep
            )
        else:
            expected_uniformly_reserved = np.zeros(
                (self._config.batch_size,), dtype=np.bool_
            )
        if not np.array_equal(
            uniformly_reserved,
            expected_uniformly_reserved,
        ):
            raise ValueError("Kondo uniform-reserve accounting is invalid")
        if self._config.mode == "bernoulli_price":
            expected_probability = np.asarray(
                jnp.where(
                    result.valid_mask,
                    jax.nn.sigmoid(
                        (
                            result.delight
                            - jnp.asarray(self._config.price, dtype=jnp.float32)
                        )
                        / jnp.asarray(
                            self._config.temperature,
                            dtype=jnp.float32,
                        )
                    ),
                    jnp.asarray(0.0, dtype=jnp.float32),
                )
            )
            gate_uniforms = np.asarray(
                jr.uniform(
                    gate_draw_key,
                    shape=(self._config.batch_size,),
                    dtype=jnp.float32,
                    minval=0.0,
                    maxval=1.0,
                )
            )
            expected_gate_mask = valid_mask & (
                gate_uniforms < expected_probability
            )
            expected_selected_mask = expected_gate_mask | reserved
            if (
                not np.array_equal(selected_mask, expected_selected_mask)
                or not np.array_equal(
                    selected_by_delight_gate,
                    expected_gate_mask,
                )
                or not np.array_equal(
                    np.asarray(result.gate_probability),
                    expected_probability,
                )
                or np.asarray(result.effective_price).tobytes()
                != np.asarray(self._config.price, dtype=np.float32).tobytes()
            ):
                raise ValueError("Kondo Bernoulli selection union is invalid")
            selected_order = np.argsort(~expected_selected_mask, kind="stable")
            expected_indices_raw = selected_order[: self._config.backward_capacity]
            expected_slot_mask = expected_selected_mask[expected_indices_raw]
            expected_indices = np.where(
                expected_slot_mask,
                expected_indices_raw,
                np.zeros_like(expected_indices_raw),
            ).astype(np.int32)
        else:
            dynamic_target = max(
                1,
                int(
                    np.rint(
                        np.float32(self._config.target_rate)
                        * np.float32(valid_count)
                    )
                ),
            )
            expected_gate_count = min(valid_count, dynamic_target)
            finite_scores = np.where(valid_mask, result.delight, -np.inf)
            delight_order = np.argsort(-finite_scores, kind="stable")
            price_index = int(
                np.clip(
                    min(valid_count, dynamic_target) - 1,
                    0,
                    self._config.batch_size - 1,
                )
            )
            expected_price = (
                np.sort(finite_scores)[::-1][price_index]
                if valid_count > 0
                else np.float32(0.0)
            )
            expected_gate_mask = np.zeros(
                (self._config.batch_size,), dtype=np.bool_
            )
            expected_gate_mask[delight_order[:expected_gate_count]] = True
            reserved_in_order = reserved[delight_order]
            valid_in_order = valid_mask[delight_order]
            class_key = np.where(
                reserved_in_order,
                0,
                np.where(valid_in_order, 1, 2),
            )
            class_order = np.argsort(class_key, kind="stable")
            ordered_indices = delight_order[class_order]
            expected_indices_raw = ordered_indices[
                : self._config.backward_capacity
            ]
            expected_selected_count = min(
                valid_count,
                max(dynamic_target, int(np.sum(reserved))),
            )
            expected_slot_mask = valid_mask[expected_indices_raw] & (
                np.arange(self._config.backward_capacity)
                < expected_selected_count
            )
            expected_selected_mask = np.zeros(
                (self._config.batch_size,), dtype=np.bool_
            )
            expected_selected_mask[expected_indices_raw] = expected_slot_mask
            expected_indices = np.where(
                expected_slot_mask,
                expected_indices_raw,
                np.zeros_like(expected_indices_raw),
            ).astype(np.int32)
            if (
                int(np.sum(selected_by_delight_gate)) != expected_gate_count
                or selected_count != expected_selected_count
                or not np.array_equal(
                    selected_by_delight_gate,
                    expected_gate_mask,
                )
                or not np.array_equal(selected_mask, expected_selected_mask)
                or not np.array_equal(
                    np.asarray(result.gate_probability),
                    expected_gate_mask.astype(np.float32),
                )
                or np.asarray(result.effective_price).tobytes()
                != np.asarray(expected_price, dtype=np.float32).tobytes()
            ):
                raise ValueError("Kondo top-k selection accounting is invalid")
        if not np.array_equal(indices.astype(np.int32), expected_indices) or not (
            np.array_equal(slot_mask, expected_slot_mask)
        ):
            raise ValueError("Kondo sparse index order is invalid")
        active_indices = indices[slot_mask]
        if np.unique(active_indices).size != active_indices.size:
            raise ValueError("Kondo sparse selection contains duplicate active rows")
        reconstructed = np.zeros((self._config.batch_size,), dtype=np.bool_)
        reconstructed[active_indices] = True
        if not np.array_equal(reconstructed, selected_mask):
            raise ValueError("Kondo sparse indices do not match selected_mask")
        leaves = jax.tree_util.tree_leaves(batch)
        if not leaves:
            raise ValueError("batch must be a non-empty PyTree")
        for leaf in leaves:
            if not hasattr(leaf, "shape") or len(cast(Any, leaf).shape) < 1:
                raise ValueError("every batch leaf must have a leading sample axis")
            if int(cast(Any, leaf).shape[0]) != self._config.batch_size:
                raise ValueError("every batch leaf must match configured batch_size")
        gathered = jax.tree_util.tree_map(
            lambda leaf: leaf[result.selected_indices],
            batch,
        )
        return KondoSparseBatch(
            data=gathered,
            sample_mask=result.selected_slot_mask,
            source_indices=result.selected_indices,
        )

    def _checkpoint_payload_for_schema(
        self,
        state: KondoGateState,
        schema: str,
    ) -> dict[str, object]:
        self._validate_state_static(state)
        if not bool(np.asarray(self._state_valid(state))):
            raise ValueError("Kondo gate state is invalid")

        def scalar(name: str) -> int:
            return int(np.asarray(getattr(state, name)))

        return {
            "schema": schema,
            "type": "KondoGateCheckpoint",
            "config": self._config._to_config_for_schema(schema),
            "state": {
                "rng_key_data": np.asarray(jr.key_data(state.rng_key), dtype=np.uint32).tolist(),
                "screen_count": scalar("screen_count"),
                "forward_slots_screened": scalar("forward_slots_screened"),
                "valid_examples_screened": scalar("valid_examples_screened"),
                "examples_selected": scalar("examples_selected"),
                "sparse_batch_count": scalar("sparse_batch_count"),
                "full_fallback_count": scalar("full_fallback_count"),
            },
        }

    def checkpoint_payload(self, state: KondoGateState) -> dict[str, object]:
        """Serialize a strict canonical v2 fixed-shape controller checkpoint."""
        return self._checkpoint_payload_for_schema(state, KONDO_GATE_SCHEMA)

    @classmethod
    def from_checkpoint_payload(
        cls,
        payload: Mapping[str, object],
    ) -> tuple[KondoGate, KondoGateState]:
        """Restore exact v2 or legacy-v1 checkpoints and normalize on emission."""
        schema = payload.get("schema")
        if type(schema) is not str:
            raise ValueError("Kondo checkpoint schema is invalid")
        if schema == KONDO_GATE_SCHEMA:
            version = "v2"
        elif schema == KONDO_GATE_LEGACY_V1_SCHEMA:
            version = "legacy v1"
        else:
            raise ValueError("Kondo checkpoint schema is invalid")
        if set(payload) != {"schema", "type", "config", "state"}:
            raise ValueError(f"Kondo checkpoint fields do not match {version}")
        if payload.get("type") != "KondoGateCheckpoint":
            raise ValueError("Kondo checkpoint type is invalid")
        config_payload = payload.get("config")
        state_payload = payload.get("state")
        if not isinstance(config_payload, Mapping) or not isinstance(state_payload, Mapping):
            raise ValueError("Kondo checkpoint config/state must be objects")
        if config_payload.get("schema") != schema:
            raise ValueError("Kondo checkpoint config schema does not match checkpoint")
        if set(state_payload) != _STATE_FIELDS:
            raise ValueError(f"Kondo checkpoint state fields do not match {version}")
        key_data = state_payload.get("rng_key_data")
        if (
            not isinstance(key_data, list)
            or len(key_data) != 2
            or any(type(item) is not int or not 0 <= item <= 0xFFFFFFFF for item in key_data)
        ):
            raise ValueError("Kondo checkpoint rng_key_data must be two uint32 integers")

        def integer(name: str) -> int:
            value = state_payload.get(name)
            if type(value) is not int or not 0 <= value <= _INT32_MAX:
                raise ValueError(f"Kondo checkpoint {name} must be a nonnegative int32")
            return value

        gate = cls.from_config(cast(Mapping[str, object], config_payload))
        state = KondoGateState(
            rng_key=jr.wrap_key_data(
                jnp.asarray(key_data, dtype=jnp.uint32),
                impl="threefry2x32",
            ),
            screen_count=jnp.asarray(integer("screen_count"), dtype=jnp.int32),
            forward_slots_screened=jnp.asarray(
                integer("forward_slots_screened"), dtype=jnp.int32
            ),
            valid_examples_screened=jnp.asarray(
                integer("valid_examples_screened"), dtype=jnp.int32
            ),
            examples_selected=jnp.asarray(integer("examples_selected"), dtype=jnp.int32),
            sparse_batch_count=jnp.asarray(integer("sparse_batch_count"), dtype=jnp.int32),
            full_fallback_count=jnp.asarray(integer("full_fallback_count"), dtype=jnp.int32),
        )
        if gate._checkpoint_payload_for_schema(state, schema) != dict(payload):
            raise ValueError("Kondo checkpoint is noncanonical")
        return gate, state


__all__ = [
    "KONDO_GATE_LEGACY_V1_SCHEMA",
    "KONDO_GATE_SCHEMA",
    "KondoGate",
    "KondoGateConfig",
    "KondoGateMode",
    "KondoGateResourceDeclaration",
    "KondoGateResult",
    "KondoGateState",
    "KondoSparseBatch",
]
