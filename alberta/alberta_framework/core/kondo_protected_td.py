# mypy: disable-error-code="attr-defined,call-arg"
"""Full-batch reward and cost TD learners protected from Kondo actor gating.

The Kondo actor consumes detached reward advantage, but it does not own the
baseline or target that define that advantage.  This module supplies two
minimal linear continuing-TD heads over one fixed batch: ``V`` for reward and
``C`` for cost.  Both heads evaluate every row, bootstrap from the same
pre-update parameters under ``stop_gradient``, and commit together or not at
all.  Action and decision identities are retained as lineage even though the
minimal heads are state-value functions.

The resulting :class:`KondoActorProtectedInputs` carries full-batch features,
baseline predictions, and return targets into the actor transaction.  This
module deliberately defines no joy or actor-backward execution alias.  It has
no dispatch, safety, evidence, or promotion authority.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core.kondo_sparse_actor import KondoActorProtectedInputs

KONDO_PROTECTED_TD_SCHEMA = "alberta.kondo-protected-td.v1"
KONDO_PROTECTED_TD_CHECKPOINT_SCHEMA = "alberta.kondo-protected-td-checkpoint.v1"
KONDO_PROTECTED_TD_RESOURCE_SCHEMA = "alberta.kondo-protected-td-resource.v1"

_TOKEN_NBYTES = 32
_INT32_MAX = np.iinfo(np.int32).max
_MAX_BATCH_SIZE = 4_096
_MAX_FEATURE_DIM = 16_384
_MAX_ACTION_COUNT = 4_096

__all__ = (
    "KONDO_PROTECTED_TD_CHECKPOINT_SCHEMA",
    "KONDO_PROTECTED_TD_RESOURCE_SCHEMA",
    "KONDO_PROTECTED_TD_SCHEMA",
    "KondoProtectedTDBackwardResult",
    "KondoProtectedTDBatch",
    "KondoProtectedTDConfig",
    "KondoProtectedTDLearner",
    "KondoProtectedTDParameters",
    "KondoProtectedTDResourceDeclaration",
    "KondoProtectedTDResult",
    "KondoProtectedTDState",
    "kondo_protected_td_backward_kernel",
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _token_bytes(payload: bytes) -> Array:
    return jnp.asarray(
        np.frombuffer(hashlib.sha256(payload).digest(), dtype=np.uint8).copy()
    )


def _require_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> None:
    if not isinstance(value, Array):
        raise TypeError(f"{name} must be a JAX array")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")


def _tree_nbytes(value: object) -> int:
    return sum(int(cast(Any, leaf).nbytes) for leaf in jax.tree_util.tree_leaves(value))


@dataclasses.dataclass(frozen=True, slots=True)
class KondoProtectedTDConfig:
    """One fixed full-batch, two-head continuing-TD contract."""

    batch_size: int
    feature_dim: int
    action_count: int
    learning_rate: float
    max_updates: int

    def __post_init__(self) -> None:
        for name, value, upper in (
            ("batch_size", self.batch_size, _MAX_BATCH_SIZE),
            ("feature_dim", self.feature_dim, _MAX_FEATURE_DIM),
            ("action_count", self.action_count, _MAX_ACTION_COUNT),
            ("max_updates", self.max_updates, _INT32_MAX),
        ):
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f"{name} must be an exact positive bounded integer")
        if type(self.learning_rate) is not float or not (
            math.isfinite(self.learning_rate) and self.learning_rate > 0.0
        ):
            raise ValueError("learning_rate must be a finite positive float")

    def to_config(self) -> dict[str, object]:
        return {
            "schema": KONDO_PROTECTED_TD_SCHEMA,
            "type": type(self).__name__,
            "batch_size": self.batch_size,
            "feature_dim": self.feature_dim,
            "action_count": self.action_count,
            "learning_rate": self.learning_rate,
            "max_updates": self.max_updates,
            "full_batch_rows": self.batch_size,
            "reward_head": "linear-state-value",
            "cost_head": "linear-state-value",
            "reward_target": "reward-plus-discount-times-detached-next-V",
            "cost_target": "cost-plus-discount-times-detached-next-C",
            "current_and_next_parameters": "same-pre-update-snapshot",
            "actor_gradient_gated": False,
            "cost_gradient_gated": False,
            "action_and_decision_lineage_retained": True,
            "random_draws_per_update": 0,
            "actor_backward_authority": False,
            "joy_alias": False,
            "dispatch_authority": False,
            "safety_authority": False,
            "evidence_authority": False,
            "promotion_authority": False,
            "scientific_promotion_allowed": False,
        }

    @classmethod
    def from_config(cls, payload: Mapping[str, object]) -> KondoProtectedTDConfig:
        if type(payload) is not dict:
            raise TypeError("protected TD config must be an exact dict")
        expected = {
            "schema",
            "type",
            "batch_size",
            "feature_dim",
            "action_count",
            "learning_rate",
            "max_updates",
            "full_batch_rows",
            "reward_head",
            "cost_head",
            "reward_target",
            "cost_target",
            "current_and_next_parameters",
            "actor_gradient_gated",
            "cost_gradient_gated",
            "action_and_decision_lineage_retained",
            "random_draws_per_update",
            "actor_backward_authority",
            "joy_alias",
            "dispatch_authority",
            "safety_authority",
            "evidence_authority",
            "promotion_authority",
            "scientific_promotion_allowed",
        }
        if set(payload) != expected:
            raise ValueError("protected TD config fields are noncanonical")
        for name in ("batch_size", "feature_dim", "action_count", "max_updates"):
            if type(payload[name]) is not int:
                raise ValueError(f"protected TD {name} must be an exact integer")
        if type(payload["learning_rate"]) is not float:
            raise ValueError("protected TD learning_rate must be an exact float")
        result = cls(
            batch_size=cast(int, payload["batch_size"]),
            feature_dim=cast(int, payload["feature_dim"]),
            action_count=cast(int, payload["action_count"]),
            learning_rate=payload["learning_rate"],
            max_updates=cast(int, payload["max_updates"]),
        )
        if result.to_config() != dict(payload):
            raise ValueError("protected TD config semantics are noncanonical")
        return result


@chex.dataclass(frozen=True)
class KondoProtectedTDParameters:
    """Linear reward-value and cost-value parameters."""

    reward_weight: Float[Array, " feature"]
    reward_bias: Float[Array, ""]
    cost_weight: Float[Array, " feature"]
    cost_bias: Float[Array, ""]


@chex.dataclass(frozen=True)
class KondoProtectedTDState:
    """Integrity-bound protected learner state."""

    config_token: UInt[Array, " 32"]
    content_token: UInt[Array, " 32"]
    parameters: KondoProtectedTDParameters
    update_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class KondoProtectedTDBatch:
    """One exact full batch with action and decision lineage retained."""

    current_features: Float[Array, "batch feature"]
    next_features: Float[Array, "batch feature"]
    actions: Int[Array, " batch"]
    decision_identities: UInt[Array, "batch 4"]
    rewards: Float[Array, " batch"]
    discounts: Float[Array, " batch"]
    costs: Float[Array, " batch"]


@chex.dataclass(frozen=True)
class KondoProtectedTDBackwardResult:
    """One real full-batch autodiff result over both protected heads."""

    total_loss: Float[Array, ""]
    reward_loss: Float[Array, ""]
    cost_loss: Float[Array, ""]
    reward_baseline: Float[Array, " batch"]
    reward_bootstrap: Float[Array, " batch"]
    return_targets: Float[Array, " batch"]
    cost_baseline: Float[Array, " batch"]
    cost_bootstrap: Float[Array, " batch"]
    cost_targets: Float[Array, " batch"]
    gradient: KondoProtectedTDParameters
    gradient_finite: Bool[Array, ""]


@chex.dataclass(frozen=True)
class KondoProtectedTDResult:
    """Candidate protected update and exact actor payload; no joy surface."""

    state: KondoProtectedTDState
    batch: KondoProtectedTDBatch
    actor_inputs: KondoActorProtectedInputs
    reward_baseline: Float[Array, " batch"]
    reward_bootstrap: Float[Array, " batch"]
    return_targets: Float[Array, " batch"]
    cost_baseline: Float[Array, " batch"]
    cost_bootstrap: Float[Array, " batch"]
    cost_targets: Float[Array, " batch"]
    reward_loss: Float[Array, ""]
    cost_loss: Float[Array, ""]
    total_loss: Float[Array, ""]
    gradient: KondoProtectedTDParameters
    full_batch_rows: Int[Array, ""]
    full_batch_backward_executed: Bool[Array, ""]
    input_finite: Bool[Array, ""]
    gradient_finite: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]


@dataclasses.dataclass(frozen=True, slots=True)
class KondoProtectedTDResourceDeclaration:
    """Exact storage and logical work, not wall-clock or FLOP evidence."""

    schema: str
    full_batch_rows: int
    parameter_count: int
    persistent_state_nbytes: int
    maximum_backwards_per_update: int
    maximum_reward_products_per_update: int
    maximum_cost_products_per_update: int
    random_draws_per_update: int
    checkpoint_supported: bool
    wall_clock_claimed: bool = False
    safety_claimed: bool = False
    efficacy_claimed: bool = False
    evidence_promotion_claimed: bool = False

    def to_config(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def _predictions(
    parameters: KondoProtectedTDParameters,
    batch: KondoProtectedTDBatch,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    reward_baseline = (
        batch.current_features @ parameters.reward_weight + parameters.reward_bias
    )
    reward_bootstrap = (
        batch.next_features @ parameters.reward_weight + parameters.reward_bias
    )
    return_targets = jax.lax.stop_gradient(
        batch.rewards + batch.discounts * reward_bootstrap
    )
    cost_baseline = batch.current_features @ parameters.cost_weight + parameters.cost_bias
    cost_bootstrap = batch.next_features @ parameters.cost_weight + parameters.cost_bias
    cost_targets = jax.lax.stop_gradient(batch.costs + batch.discounts * cost_bootstrap)
    return (
        reward_baseline,
        reward_bootstrap,
        return_targets,
        cost_baseline,
        cost_bootstrap,
        cost_targets,
    )


def _loss(
    parameters: KondoProtectedTDParameters,
    batch: KondoProtectedTDBatch,
) -> tuple[Array, tuple[Array, ...]]:
    predictions = _predictions(parameters, batch)
    reward_baseline, _, return_targets, cost_baseline, _, cost_targets = predictions
    reward_loss = jnp.asarray(0.5, dtype=jnp.float32) * jnp.mean(
        jnp.square(reward_baseline - return_targets)
    )
    cost_loss = jnp.asarray(0.5, dtype=jnp.float32) * jnp.mean(
        jnp.square(cost_baseline - cost_targets)
    )
    return reward_loss + cost_loss, (reward_loss, cost_loss, *predictions)


def kondo_protected_td_backward_kernel(
    parameters: KondoProtectedTDParameters,
    batch: KondoProtectedTDBatch,
) -> KondoProtectedTDBackwardResult:
    """Differentiate both heads over the complete fixed batch."""

    (total_loss, auxiliary), gradient = jax.value_and_grad(_loss, has_aux=True)(
        parameters,
        batch,
    )
    (
        reward_loss,
        cost_loss,
        reward_baseline,
        reward_bootstrap,
        return_targets,
        cost_baseline,
        cost_bootstrap,
        cost_targets,
    ) = auxiliary
    gradient_finite = jnp.isfinite(total_loss)
    for leaf in jax.tree_util.tree_leaves(gradient):
        gradient_finite = gradient_finite & jnp.all(jnp.isfinite(leaf))
    return KondoProtectedTDBackwardResult(
        total_loss=total_loss,
        reward_loss=reward_loss,
        cost_loss=cost_loss,
        reward_baseline=reward_baseline,
        reward_bootstrap=reward_bootstrap,
        return_targets=return_targets,
        cost_baseline=cost_baseline,
        cost_bootstrap=cost_bootstrap,
        cost_targets=cost_targets,
        gradient=cast(KondoProtectedTDParameters, gradient),
        gradient_finite=gradient_finite,
    )


class KondoProtectedTDLearner:
    """Integrity-bound owner of full-batch reward and cost TD updates."""

    def __init__(self, config: KondoProtectedTDConfig) -> None:
        if type(config) is not KondoProtectedTDConfig:
            raise TypeError("config must be exact KondoProtectedTDConfig")
        self._config = config
        self._config_token = _token_bytes(_canonical_json_bytes(config.to_config()))

    @property
    def config(self) -> KondoProtectedTDConfig:
        return self._config

    def to_config(self) -> dict[str, object]:
        return self._config.to_config()

    @classmethod
    def from_config(cls, payload: Mapping[str, object]) -> KondoProtectedTDLearner:
        return cls(KondoProtectedTDConfig.from_config(payload))

    def _parameter_contract(self, parameters: KondoProtectedTDParameters) -> None:
        if type(parameters) is not KondoProtectedTDParameters:
            raise TypeError("parameters must be exact KondoProtectedTDParameters")
        cfg = self._config
        _require_array(
            parameters.reward_weight,
            name="parameters.reward_weight",
            shape=(cfg.feature_dim,),
            dtype=jnp.dtype(jnp.float32),
        )
        _require_array(
            parameters.reward_bias,
            name="parameters.reward_bias",
            shape=(),
            dtype=jnp.dtype(jnp.float32),
        )
        _require_array(
            parameters.cost_weight,
            name="parameters.cost_weight",
            shape=(cfg.feature_dim,),
            dtype=jnp.dtype(jnp.float32),
        )
        _require_array(
            parameters.cost_bias,
            name="parameters.cost_bias",
            shape=(),
            dtype=jnp.dtype(jnp.float32),
        )

    def _state_token(self, state: KondoProtectedTDState) -> Array:
        digest = hashlib.sha256()
        digest.update(KONDO_PROTECTED_TD_SCHEMA.encode("ascii"))
        digest.update(np.asarray(state.config_token).tobytes(order="C"))
        for name in ("reward_weight", "reward_bias", "cost_weight", "cost_bias"):
            array = np.asarray(getattr(state.parameters, name), dtype=np.float32)
            digest.update(name.encode("ascii"))
            digest.update(array.tobytes(order="C"))
        digest.update(np.asarray(state.update_count, dtype=np.int32).tobytes(order="C"))
        return jnp.asarray(np.frombuffer(digest.digest(), dtype=np.uint8).copy())

    def reseal_state(self, state: KondoProtectedTDState) -> KondoProtectedTDState:
        """Recompute the content token after a deliberate state construction."""

        if type(state) is not KondoProtectedTDState:
            raise TypeError("state must be exact KondoProtectedTDState")
        return cast(
            KondoProtectedTDState,
            state.replace(content_token=self._state_token(state)),
        )

    def _state_contract(self, state: KondoProtectedTDState) -> None:
        if type(state) is not KondoProtectedTDState:
            raise TypeError("state must be exact KondoProtectedTDState")
        _require_array(
            state.config_token,
            name="state.config_token",
            shape=(_TOKEN_NBYTES,),
            dtype=jnp.dtype(jnp.uint8),
        )
        _require_array(
            state.content_token,
            name="state.content_token",
            shape=(_TOKEN_NBYTES,),
            dtype=jnp.dtype(jnp.uint8),
        )
        _require_array(
            state.update_count,
            name="state.update_count",
            shape=(),
            dtype=jnp.dtype(jnp.int32),
        )
        self._parameter_contract(state.parameters)

    def state_valid(self, state: KondoProtectedTDState) -> Bool[Array, ""]:
        self._state_contract(state)
        token_valid = np.array_equal(
            np.asarray(state.config_token),
            np.asarray(self._config_token),
        ) and np.array_equal(
            np.asarray(state.content_token),
            np.asarray(self._state_token(state)),
        )
        finite = all(
            bool(np.all(np.isfinite(np.asarray(leaf))))
            for leaf in jax.tree_util.tree_leaves(state.parameters)
        )
        count = int(state.update_count)
        return jnp.asarray(
            token_valid and finite and 0 <= count <= self._config.max_updates,
            dtype=jnp.bool_,
        )

    def init(
        self,
        parameters: KondoProtectedTDParameters | None = None,
    ) -> KondoProtectedTDState:
        selected = parameters
        if selected is None:
            zero = jnp.zeros((self._config.feature_dim,), dtype=jnp.float32)
            selected = KondoProtectedTDParameters(
                reward_weight=zero,
                reward_bias=jnp.asarray(0.0, dtype=jnp.float32),
                cost_weight=zero,
                cost_bias=jnp.asarray(0.0, dtype=jnp.float32),
            )
        self._parameter_contract(selected)
        if not all(
            bool(np.all(np.isfinite(np.asarray(leaf))))
            for leaf in jax.tree_util.tree_leaves(selected)
        ):
            raise ValueError("initial protected TD parameters must be finite")
        bare = KondoProtectedTDState(
            config_token=self._config_token,
            content_token=jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8),
            parameters=selected,
            update_count=jnp.asarray(0, dtype=jnp.int32),
        )
        return self.reseal_state(bare)

    def _batch_contract(self, batch: KondoProtectedTDBatch) -> None:
        if type(batch) is not KondoProtectedTDBatch:
            raise TypeError("batch must be exact KondoProtectedTDBatch")
        cfg = self._config
        for name in ("current_features", "next_features"):
            _require_array(
                getattr(batch, name),
                name=f"batch.{name}",
                shape=(cfg.batch_size, cfg.feature_dim),
                dtype=jnp.dtype(jnp.float32),
            )
        _require_array(
            batch.actions,
            name="batch.actions",
            shape=(cfg.batch_size,),
            dtype=jnp.dtype(jnp.int32),
        )
        _require_array(
            batch.decision_identities,
            name="batch.decision_identities",
            shape=(cfg.batch_size, 4),
            dtype=jnp.dtype(jnp.uint32),
        )
        for name in ("rewards", "discounts", "costs"):
            _require_array(
                getattr(batch, name),
                name=f"batch.{name}",
                shape=(cfg.batch_size,),
                dtype=jnp.dtype(jnp.float32),
            )
        actions = np.asarray(batch.actions)
        if not bool(np.all((actions >= 0) & (actions < cfg.action_count))):
            raise ValueError("batch.actions lie outside the configured action domain")
        for name in ("current_features", "next_features", "rewards", "discounts", "costs"):
            if not bool(np.all(np.isfinite(np.asarray(getattr(batch, name))))):
                raise ValueError(f"batch.{name} must be finite")
        discounts = np.asarray(batch.discounts)
        if not bool(np.all((discounts >= 0.0) & (discounts <= 1.0))):
            raise ValueError("batch.discounts must lie in [0, 1]")
        if not bool(np.all(np.asarray(batch.costs) >= 0.0)):
            raise ValueError("batch.costs must be nonnegative")

    def step(
        self,
        state: KondoProtectedTDState,
        batch: KondoProtectedTDBatch,
    ) -> KondoProtectedTDResult:
        """Run exactly one two-head full-batch backward and atomic update."""

        self._state_contract(state)
        if not bool(np.asarray(self.state_valid(state))):
            raise ValueError("protected TD step requires a valid source state")
        self._batch_contract(batch)
        if int(state.update_count) >= self._config.max_updates:
            raise OverflowError("protected TD max_updates is exhausted")

        backward = kondo_protected_td_backward_kernel(state.parameters, batch)
        rate = jnp.asarray(self._config.learning_rate, dtype=jnp.float32)
        candidate_parameters = cast(
            KondoProtectedTDParameters,
            jax.tree_util.tree_map(
                lambda parameter, gradient: parameter - rate * gradient,
                state.parameters,
                backward.gradient,
            ),
        )
        bare_candidate = KondoProtectedTDState(
            config_token=state.config_token,
            content_token=jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8),
            parameters=candidate_parameters,
            update_count=state.update_count + jnp.asarray(1, dtype=jnp.int32),
        )
        candidate = self.reseal_state(bare_candidate)
        candidate_valid = bool(np.asarray(self.state_valid(candidate)))
        applied = bool(np.asarray(backward.gradient_finite)) and candidate_valid
        final_state = candidate if applied else state
        actor_inputs = KondoActorProtectedInputs(
            critic_features=batch.current_features,
            baseline_predictions=backward.reward_baseline,
            return_targets=backward.return_targets,
            safety_features=batch.current_features,
        )
        return KondoProtectedTDResult(
            state=final_state,
            batch=batch,
            actor_inputs=actor_inputs,
            reward_baseline=backward.reward_baseline,
            reward_bootstrap=backward.reward_bootstrap,
            return_targets=backward.return_targets,
            cost_baseline=backward.cost_baseline,
            cost_bootstrap=backward.cost_bootstrap,
            cost_targets=backward.cost_targets,
            reward_loss=backward.reward_loss,
            cost_loss=backward.cost_loss,
            total_loss=backward.total_loss,
            gradient=backward.gradient,
            full_batch_rows=jnp.asarray(self._config.batch_size, dtype=jnp.int32),
            full_batch_backward_executed=jnp.asarray(True, dtype=jnp.bool_),
            input_finite=jnp.asarray(True, dtype=jnp.bool_),
            gradient_finite=backward.gradient_finite,
            candidate_state_valid=jnp.asarray(candidate_valid, dtype=jnp.bool_),
            transaction_applied=jnp.asarray(applied, dtype=jnp.bool_),
        )

    @staticmethod
    def _encode_float32(value: Array) -> dict[str, object]:
        array = np.asarray(value, dtype=np.float32)
        return {
            "shape": list(array.shape),
            "float32_bits": [
                int(item) for item in array.view(np.uint32).reshape(-1).tolist()
            ],
        }

    @staticmethod
    def _decode_float32(
        payload: object,
        *,
        name: str,
        shape: tuple[int, ...],
    ) -> Array:
        if type(payload) is not dict or set(payload) != {"shape", "float32_bits"}:
            raise ValueError(f"checkpoint {name} encoding is invalid")
        encoded_shape = payload["shape"]
        bits = payload["float32_bits"]
        if type(encoded_shape) is not list or encoded_shape != list(shape):
            raise ValueError(f"checkpoint {name} shape is invalid")
        if type(bits) is not list or len(bits) != int(np.prod(shape, dtype=np.int64)):
            raise ValueError(f"checkpoint {name} bit count is invalid")
        if any(type(item) is not int or not 0 <= item <= np.iinfo(np.uint32).max for item in bits):
            raise ValueError(f"checkpoint {name} bits are invalid")
        array = np.asarray(bits, dtype=np.uint32).view(np.float32).reshape(shape)
        return jnp.asarray(array)

    def checkpoint_payload(self, state: KondoProtectedTDState) -> dict[str, object]:
        """Return a strict bit-exact JSON-compatible learner checkpoint."""

        if not bool(np.asarray(self.state_valid(state))):
            raise ValueError("cannot checkpoint an invalid protected TD state")
        parameters = {
            name: self._encode_float32(getattr(state.parameters, name))
            for name in ("reward_weight", "reward_bias", "cost_weight", "cost_bias")
        }
        body: dict[str, object] = {
            "schema": KONDO_PROTECTED_TD_CHECKPOINT_SCHEMA,
            "config": self.to_config(),
            "state": {
                "parameters": parameters,
                "update_count": int(state.update_count),
            },
        }
        return {
            **body,
            "payload_sha256": hashlib.sha256(_canonical_json_bytes(body)).hexdigest(),
        }

    @classmethod
    def from_checkpoint_payload(
        cls,
        payload: Mapping[str, object],
    ) -> tuple[KondoProtectedTDLearner, KondoProtectedTDState]:
        """Restore only an exact current-schema protected learner checkpoint."""

        if type(payload) is not dict or set(payload) != {
            "schema",
            "config",
            "state",
            "payload_sha256",
        }:
            raise ValueError("protected TD checkpoint fields are invalid")
        digest = payload.get("payload_sha256")
        if type(digest) is not str or len(digest) != 64:
            raise ValueError("protected TD checkpoint digest is invalid")
        body = {name: payload[name] for name in ("schema", "config", "state")}
        if hashlib.sha256(_canonical_json_bytes(body)).hexdigest() != digest:
            raise ValueError("protected TD checkpoint digest is invalid")
        if body["schema"] != KONDO_PROTECTED_TD_CHECKPOINT_SCHEMA:
            raise ValueError("protected TD checkpoint schema is invalid")
        config_payload = body["config"]
        state_payload = body["state"]
        if type(config_payload) is not dict or type(state_payload) is not dict:
            raise ValueError("protected TD checkpoint objects are invalid")
        if set(state_payload) != {"parameters", "update_count"}:
            raise ValueError("protected TD checkpoint state fields are invalid")
        learner = cls.from_config(config_payload)
        parameter_payload = state_payload["parameters"]
        if type(parameter_payload) is not dict or set(parameter_payload) != {
            "reward_weight",
            "reward_bias",
            "cost_weight",
            "cost_bias",
        }:
            raise ValueError("protected TD checkpoint parameters are invalid")
        cfg = learner.config
        parameters = KondoProtectedTDParameters(
            reward_weight=learner._decode_float32(
                parameter_payload["reward_weight"],
                name="reward_weight",
                shape=(cfg.feature_dim,),
            ),
            reward_bias=learner._decode_float32(
                parameter_payload["reward_bias"],
                name="reward_bias",
                shape=(),
            ),
            cost_weight=learner._decode_float32(
                parameter_payload["cost_weight"],
                name="cost_weight",
                shape=(cfg.feature_dim,),
            ),
            cost_bias=learner._decode_float32(
                parameter_payload["cost_bias"],
                name="cost_bias",
                shape=(),
            ),
        )
        count = state_payload["update_count"]
        if type(count) is not int or not 0 <= count <= cfg.max_updates:
            raise ValueError("protected TD checkpoint update_count is invalid")
        bare = KondoProtectedTDState(
            config_token=learner._config_token,
            content_token=jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8),
            parameters=parameters,
            update_count=jnp.asarray(count, dtype=jnp.int32),
        )
        state = learner.reseal_state(bare)
        if not bool(np.asarray(learner.state_valid(state))):
            raise ValueError("protected TD checkpoint state is invalid")
        if learner.checkpoint_payload(state) != dict(payload):
            raise ValueError("protected TD checkpoint is noncanonical")
        return learner, state

    def resource_declaration(
        self,
        state: KondoProtectedTDState | None = None,
    ) -> KondoProtectedTDResourceDeclaration:
        selected = self.init() if state is None else state
        if not bool(np.asarray(self.state_valid(selected))):
            raise ValueError("protected TD resources require a valid state")
        cfg = self._config
        return KondoProtectedTDResourceDeclaration(
            schema=KONDO_PROTECTED_TD_RESOURCE_SCHEMA,
            full_batch_rows=cfg.batch_size,
            parameter_count=2 * (cfg.feature_dim + 1),
            persistent_state_nbytes=_tree_nbytes(selected),
            maximum_backwards_per_update=1,
            maximum_reward_products_per_update=2 * cfg.batch_size * cfg.feature_dim,
            maximum_cost_products_per_update=2 * cfg.batch_size * cfg.feature_dim,
            random_draws_per_update=0,
            checkpoint_supported=True,
        )
