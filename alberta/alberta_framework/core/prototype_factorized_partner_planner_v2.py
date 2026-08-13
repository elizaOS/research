# mypy: disable-error-code="attr-defined,call-arg,no-any-return"
"""Source-update-first feature routing for the paired Prototype planner.

This L0 owner composes the existing :class:`BehaviorModel`,
:class:`GroundedJointWorldModel`, and v2 HCCL semantic-birth route.  For each
agent it performs exactly one Behavior update and one Grounded update on the
executed transition under the source 35-coordinate representation, routes the
updated feature columns, rebuilds a destination-bound planning cache, and only
then prepares a read-only action proposal ``P``.  Planning performs no second
model update and consumes no post-initialization randomness.

The Grounded head order is fixed to physical16, task score, safety cost,
message charge, net reward, and discount.  ``GroundedJointWorldModel`` exposes
``[next_observation, reward, discount]``; this owner therefore uses a
19-coordinate target observation containing physical16 plus the first three
named signals and assigns net reward to its reward head.

Representation/ledger composition is also checked exactly.  Every live pair
coordinate must be the bit-exact float32 product of its two physical parents;
inactive pair and context coordinates must be positive zero.  Values of live
context coordinates and all fast coordinates remain outer-owned: this L0 unit
checks their identity/activity envelope but does not claim to construct them.

All composition checks are deliberately host/eager-only.  SHA-256 tokens bind
supplied in-memory state but are unkeyed integrity checks, not authentication
or replay protection.  The returned proposal has no dispatch, evidence, or
promotion authority.  Composition does not modify either donor implementation
or mutate caller-supplied donor states in place; updates return new functional
state values.
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
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core.behavior_model import (
    BehaviorModel,
    BehaviorModelConfig,
    BehaviorModelState,
    BehaviorModelUpdateResult,
)
from alberta_framework.core.grounded_joint_world_model import (
    GroundedJointWorldModel,
    GroundedJointWorldModelConfig,
    GroundedJointWorldModelState,
    GroundedJointWorldUpdateResult,
)
from alberta_framework.core.hccl_feature_consumer_route import (
    HCCL_FEATURE_CONTEXT_START,
    HCCL_FEATURE_FAST_START,
    HCCL_FEATURE_PAIR_SLOTS,
    HCCL_FEATURE_PAIR_START,
    HCCL_FEATURE_TOTAL_DIM,
    HCCLFeatureBirthLedger,
    HCCLFeatureConsumerRoute,
    HCCLFeatureConsumerRouteMap,
    HCCLFeatureConsumerRouteResult,
)

PROTOTYPE_FACTORIZED_PARTNER_PLANNER_V2_SCHEMA = (
    "alberta.prototype-factorized-partner-planner-v2.config.v1"
)
PROTOTYPE_FACTORIZED_PARTNER_PLANNER_V2_STATE_SCHEMA = (
    "alberta.prototype-factorized-partner-planner-v2.state.v1"
)
PROTOTYPE_FACTORIZED_PARTNER_PLANNER_V2_STATUS = (
    "l0-development-source-update-route-plan-only"
)
PROTOTYPE_FACTORIZED_PARTNER_PLANNER_V2_SCIENTIFIC_PROMOTION_ALLOWED = False
PROTOTYPE_FACTORIZED_PARTNER_PLANNER_V2_DISPATCH_AUTHORITY = False

N_AGENTS = 2
N_ACTIONS = 2
PHYSICAL_OUTPUT_DIM = 16
GROUNDED_TARGET_OBSERVATION_DIM = 19
GROUNDED_TARGET_DIM = 21
TASK_SCORE_OUTPUT_INDEX = 16
SAFETY_COST_OUTPUT_INDEX = 17
MESSAGE_CHARGE_OUTPUT_INDEX = 18
NET_REWARD_OUTPUT_INDEX = 19
DISCOUNT_OUTPUT_INDEX = 20
GROUNDED_OUTPUT_ORDER = tuple(
    [f"physical_{index}" for index in range(PHYSICAL_OUTPUT_DIM)]
    + ["task_score", "safety_cost", "message_charge", "net_reward", "discount"]
)

_TOKEN_NBYTES = 32
_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _contains_tracer(value: object) -> bool:
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(value))


def _digest_tree(schema: str, *values: object) -> UInt[Array, " 32"]:
    if _contains_tracer(values):
        raise TypeError("planner-v2 integrity is host/eager-only")
    digest = hashlib.sha256(schema.encode("ascii"))
    for value in values:
        digest.update(type(value).__module__.encode("utf-8"))
        digest.update(type(value).__qualname__.encode("utf-8"))
        leaves, structure = jax.tree.flatten(value)
        digest.update(repr(structure).encode("utf-8"))
        digest.update(len(leaves).to_bytes(8, "big"))
        for leaf in leaves:
            array = jnp.asarray(leaf)
            if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
                array = jr.key_data(array)
            host = np.ascontiguousarray(np.asarray(jax.device_get(array)))
            digest.update(str(host.dtype).encode("ascii"))
            digest.update(np.asarray(host.shape, dtype=np.int64).tobytes())
            digest.update(host.tobytes(order="C"))
    return jnp.asarray(tuple(digest.digest()), dtype=jnp.uint8)


def _require_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must expose exact array metadata")
    array = cast(Array, value)
    if tuple(array.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}; got {tuple(array.shape)}")
    if jnp.dtype(array.dtype) != jnp.dtype(dtype):
        raise TypeError(f"{name} must have dtype {jnp.dtype(dtype)}; got {array.dtype}")
    return array


def _require_key(value: object, *, name: str) -> Array:
    if not hasattr(value, "shape") or tuple(cast(Array, value).shape) != ():
        raise TypeError(f"{name} must be a scalar typed PRNG key")
    key = cast(Array, value)
    if not jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key):
        raise TypeError(f"{name} must be a scalar typed PRNG key")
    return key


def _host(value: Array) -> np.ndarray[Any, Any]:
    return np.asarray(jax.device_get(value))


def _host_bool(value: object) -> bool:
    return bool(np.asarray(jax.device_get(value)))


def _array_exact_equal(left: Array, right: Array) -> bool:
    left_array = jnp.asarray(left)
    right_array = jnp.asarray(right)
    if jax.dtypes.issubdtype(left_array.dtype, jax.dtypes.prng_key):
        if not jax.dtypes.issubdtype(right_array.dtype, jax.dtypes.prng_key):
            return False
        left_array = jr.key_data(left_array)
        right_array = jr.key_data(right_array)
    left_host = np.ascontiguousarray(np.asarray(jax.device_get(left_array)))
    right_host = np.ascontiguousarray(np.asarray(jax.device_get(right_array)))
    return (
        left_host.dtype == right_host.dtype
        and left_host.shape == right_host.shape
        and left_host.tobytes(order="C") == right_host.tobytes(order="C")
    )


def _tree_exact_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    if cast(object, left_tree) != cast(object, right_tree):
        return False
    if len(left_leaves) != len(right_leaves):
        return False
    return all(
        _array_exact_equal(jnp.asarray(left_leaf), jnp.asarray(right_leaf))
        for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True)
    )


def _counter_valid(words: Array, telemetry: Array) -> bool:
    high, low = (int(item) for item in _host(words))
    exact = (high << 32) | low
    return int(_host(telemetry)) == min(exact, _INT32_MAX)


def _words_successor(source: Array, destination: Array) -> bool:
    high, low = (int(item) for item in _host(source))
    if high == _UINT32_MAX and low == _UINT32_MAX:
        return False
    expected = (high + 1, 0) if low == _UINT32_MAX else (high, low + 1)
    return expected == tuple(int(item) for item in _host(destination))


def _probabilities_valid(probabilities: Array) -> bool:
    values = _host(probabilities)
    return bool(
        np.all(np.isfinite(values))
        and np.all(values >= 0.0)
        and np.all(values <= 1.0)
        and np.isclose(np.sum(values), 1.0, atol=1.0e-5, rtol=0.0)
    )


def _finite_positive(value: object, *, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a real non-boolean scalar")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return numeric


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeFactorizedPartnerPlannerV2Config:
    """Fixed two-agent, two-action R35 construction."""

    behavior_step_size: float = 0.05
    grounded_step_size: float = 0.02
    grounded_initialization_scale: float = 0.01
    max_input_magnitude: float = 1_000.0
    max_parameter_magnitude: float = 1_000.0
    schema: str = PROTOTYPE_FACTORIZED_PARTNER_PLANNER_V2_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "behavior_step_size",
            "grounded_step_size",
            "grounded_initialization_scale",
            "max_input_magnitude",
            "max_parameter_magnitude",
        ):
            object.__setattr__(self, name, _finite_positive(getattr(self, name), name=name))
        if self.grounded_initialization_scale > self.max_parameter_magnitude:
            raise ValueError("grounded_initialization_scale exceeds the parameter bound")
        if self.schema != PROTOTYPE_FACTORIZED_PARTNER_PLANNER_V2_SCHEMA:
            raise ValueError("planner-v2 schema is unsupported")

    def to_config(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "schema": self.schema,
            "status": PROTOTYPE_FACTORIZED_PARTNER_PLANNER_V2_STATUS,
            "development_only": True,
            "scientific_promotion_allowed": False,
            "dispatch_authority": False,
            "history_owned": False,
            "replay_detection_claimed": False,
            "n_agents": N_AGENTS,
            "n_actions": N_ACTIONS,
            "representation_dim": HCCL_FEATURE_TOTAL_DIM,
            "grounded_output_order": list(GROUNDED_OUTPUT_ORDER),
            "behavior_step_size": self.behavior_step_size,
            "grounded_step_size": self.grounded_step_size,
            "grounded_initialization_scale": self.grounded_initialization_scale,
            "max_input_magnitude": self.max_input_magnitude,
            "max_parameter_magnitude": self.max_parameter_magnitude,
        }

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, object],
    ) -> PrototypeFactorizedPartnerPlannerV2Config:
        if type(payload) is not dict:
            raise TypeError("planner-v2 config must be an exact dict")
        values = dict(payload)
        expected_fixed = {
            "type": cls.__name__,
            "status": PROTOTYPE_FACTORIZED_PARTNER_PLANNER_V2_STATUS,
            "development_only": True,
            "scientific_promotion_allowed": False,
            "dispatch_authority": False,
            "history_owned": False,
            "replay_detection_claimed": False,
            "n_agents": N_AGENTS,
            "n_actions": N_ACTIONS,
            "representation_dim": HCCL_FEATURE_TOTAL_DIM,
            "grounded_output_order": list(GROUNDED_OUTPUT_ORDER),
        }
        for name, expected in expected_fixed.items():
            if values.pop(name, None) != expected:
                raise ValueError(f"planner-v2 config field {name!r} is noncanonical")
        for name in (
            "behavior_step_size",
            "grounded_step_size",
            "grounded_initialization_scale",
            "max_input_magnitude",
            "max_parameter_magnitude",
        ):
            if type(values.get(name)) is not float:
                raise ValueError(f"serialized planner-v2 field {name!r} must be an exact float")
        restored = cls(**values)
        if _canonical_json_bytes(restored.to_config()) != _canonical_json_bytes(payload):
            raise ValueError("planner-v2 config is noncanonical")
        return restored


@chex.dataclass(frozen=True)
class PrototypeFactorizedPartnerPlannerV2Cache:
    """One destination-ledger-bound, read-only P preparation."""

    ledger_content_token: UInt[Array, " 32"]
    representation: Float[Array, " 35"]
    behavior_step_words: UInt[Array, " 2"]
    grounded_update_words: UInt[Array, " 2"]
    partner_belief: Float[Array, " 2"]
    partner_belief_by_own_action: Float[Array, "2 2"]
    world_raw_predictions: Float[Array, "2 2 21"]
    world_cell_valid: Bool[Array, "2 2"]
    expected_net_rewards: Float[Array, " 2"]
    prepared_action: Int[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeFactorizedPartnerPlannerV2AgentState:
    """One ledger, two donor models, and its destination cache."""

    ledger: HCCLFeatureBirthLedger
    behavior: BehaviorModelState
    grounded: GroundedJointWorldModelState
    cache: PrototypeFactorizedPartnerPlannerV2Cache


@chex.dataclass(frozen=True)
class PrototypeFactorizedPartnerPlannerV2State:
    """Integrity-bound paired planner state."""

    config_token: UInt[Array, " 32"]
    content_token: UInt[Array, " 32"]
    agent_0: PrototypeFactorizedPartnerPlannerV2AgentState
    agent_1: PrototypeFactorizedPartnerPlannerV2AgentState


@chex.dataclass(frozen=True)
class PrototypeFactorizedPartnerPlannerV2SourceUpdateReceipt:
    """Predict-before-update receipt under the source R35 bank."""

    source_state_content_token: UInt[Array, " 32"]
    source_ledger_content_tokens: UInt[Array, "2 32"]
    source_representations: Float[Array, "2 35"]
    executed_actions: Int[Array, " 2"]
    observed_partner_actions: Int[Array, " 2"]
    grounded_targets: Float[Array, "2 21"]
    behavior_pre_step_words: UInt[Array, "2 2"]
    behavior_post_step_words: UInt[Array, "2 2"]
    grounded_pre_update_words: UInt[Array, "2 2"]
    grounded_post_update_words: UInt[Array, "2 2"]
    source_cache_matches: Bool[Array, " 2"]
    source_representation_matches_ledger: Bool[Array, " 2"]
    behavior_update_applied: Bool[Array, " 2"]
    grounded_update_applied: Bool[Array, " 2"]
    grounded_target_order_valid: Bool[Array, " 2"]
    clocks_advanced_once: Bool[Array, " 2"]
    phase_valid: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeFactorizedPartnerPlannerV2RouteReceipt:
    """Route-result and post-update column-preservation receipt."""

    route_witness_content_tokens: UInt[Array, "2 32"]
    destination_ledger_content_tokens: UInt[Array, "2 32"]
    route_result_integrity_valid: Bool[Array, " 2"]
    route_transaction_applied: Bool[Array, " 2"]
    destination_representation_matches_ledger: Bool[Array, " 2"]
    post_update_clock_matches_destination: Bool[Array, " 2"]
    survivor_columns_bit_exact: Bool[Array, " 2"]
    newborn_columns_positive_zero: Bool[Array, " 2"]
    inactive_columns_positive_zero: Bool[Array, " 2"]
    nonfeature_fields_bit_exact: Bool[Array, " 2"]
    source_update_preceded_route: Bool[Array, " 2"]
    phase_valid: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeFactorizedPartnerPlannerV2PlanReceipt:
    """Destination-cache and four-cell marginalization receipt."""

    destination_representations: Float[Array, "2 35"]
    partner_belief: Float[Array, "2 2"]
    partner_belief_by_own_action: Float[Array, "2 2 2"]
    world_raw_predictions: Float[Array, "2 2 2 21"]
    world_cell_valid: Bool[Array, "2 2 2"]
    expected_net_rewards: Float[Array, "2 2"]
    proposed_actions: Int[Array, " 2"]
    identical_partner_belief_across_own_rows: Bool[Array, " 2"]
    joint_cells_evaluated_per_agent: Int[Array, " 2"]
    destination_cache_bound: Bool[Array, " 2"]
    model_clocks_unchanged_by_planning: Bool[Array, " 2"]
    phase_valid: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeFactorizedPartnerPlannerV2Receipt:
    """Complete three-phase attempted transaction receipt."""

    source_update: PrototypeFactorizedPartnerPlannerV2SourceUpdateReceipt
    feature_route: PrototypeFactorizedPartnerPlannerV2RouteReceipt
    plan: PrototypeFactorizedPartnerPlannerV2PlanReceipt
    source_state_valid: Bool[Array, ""]
    event_inputs_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]
    complete_source_returned: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeFactorizedPartnerPlannerV2Work:
    """Fixed logical calls/products in one public transition attempt."""

    source_state_validations: Int[Array, ""]
    candidate_state_validations: Int[Array, ""]
    behavior_model_updates: Int[Array, ""]
    grounded_model_updates: Int[Array, ""]
    route_result_integrity_evaluations: Int[Array, ""]
    behavior_column_routes: Int[Array, ""]
    grounded_column_routes: Int[Array, ""]
    behavior_probability_evaluations: Int[Array, ""]
    grounded_joint_cell_evaluations: Int[Array, ""]
    cache_validation_behavior_probability_evaluations: Int[Array, ""]
    cache_validation_grounded_joint_cell_evaluations: Int[Array, ""]
    partner_belief_marginalization_products: Int[Array, ""]
    planner_action_preparations: Int[Array, ""]
    second_model_updates: Int[Array, ""]
    post_init_rng_draws: Int[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeFactorizedPartnerPlannerV2Result:
    """Selected source-or-candidate state and auditable attempted P."""

    state: PrototypeFactorizedPartnerPlannerV2State
    candidate_state: PrototypeFactorizedPartnerPlannerV2State
    prepared_actions: Int[Array, " 2"]
    receipt: PrototypeFactorizedPartnerPlannerV2Receipt
    work: PrototypeFactorizedPartnerPlannerV2Work
    transaction_applied: Bool[Array, ""]
    complete_source_returned: Bool[Array, ""]


def _fixed_work() -> PrototypeFactorizedPartnerPlannerV2Work:
    return PrototypeFactorizedPartnerPlannerV2Work(
        source_state_validations=jnp.asarray(1, dtype=jnp.int32),
        candidate_state_validations=jnp.asarray(1, dtype=jnp.int32),
        behavior_model_updates=jnp.asarray(N_AGENTS, dtype=jnp.int32),
        grounded_model_updates=jnp.asarray(N_AGENTS, dtype=jnp.int32),
        route_result_integrity_evaluations=jnp.asarray(N_AGENTS, dtype=jnp.int32),
        behavior_column_routes=jnp.asarray(N_AGENTS, dtype=jnp.int32),
        grounded_column_routes=jnp.asarray(N_AGENTS, dtype=jnp.int32),
        behavior_probability_evaluations=jnp.asarray(N_AGENTS, dtype=jnp.int32),
        grounded_joint_cell_evaluations=jnp.asarray(
            N_AGENTS * N_ACTIONS**2,
            dtype=jnp.int32,
        ),
        cache_validation_behavior_probability_evaluations=jnp.asarray(
            3 * N_AGENTS,
            dtype=jnp.int32,
        ),
        cache_validation_grounded_joint_cell_evaluations=jnp.asarray(
            3 * N_AGENTS * N_ACTIONS**2,
            dtype=jnp.int32,
        ),
        partner_belief_marginalization_products=jnp.asarray(
            N_AGENTS * N_ACTIONS**2,
            dtype=jnp.int32,
        ),
        planner_action_preparations=jnp.asarray(N_AGENTS, dtype=jnp.int32),
        second_model_updates=jnp.asarray(0, dtype=jnp.int32),
        post_init_rng_draws=jnp.asarray(0, dtype=jnp.int32),
    )


class PrototypeFactorizedPartnerPlannerV2:
    """Paired source-update, full-birth route, and four-cell P owner."""

    def __init__(self, config: PrototypeFactorizedPartnerPlannerV2Config) -> None:
        if type(config) is not PrototypeFactorizedPartnerPlannerV2Config:
            raise TypeError("config must be an exact planner-v2 config")
        self._config = config
        self._behavior = BehaviorModel(
            BehaviorModelConfig(
                n_actions=N_ACTIONS,
                step_size=config.behavior_step_size,
            )
        )
        self._grounded = GroundedJointWorldModel(
            GroundedJointWorldModelConfig(
                representation_dim=HCCL_FEATURE_TOTAL_DIM,
                target_observation_dim=GROUNDED_TARGET_OBSERVATION_DIM,
                n_focal_actions=N_ACTIONS,
                n_partner_actions=N_ACTIONS,
                step_size=config.grounded_step_size,
                initialization_scale=config.grounded_initialization_scale,
                max_input_magnitude=config.max_input_magnitude,
                max_parameter_magnitude=config.max_parameter_magnitude,
            )
        )
        self._routes = (
            HCCLFeatureConsumerRoute(agent_index=0),
            HCCLFeatureConsumerRoute(agent_index=1),
        )
        payload = {
            "planner": config.to_config(),
            "behavior": self._behavior.to_config(),
            "grounded": self._grounded.to_config(),
        }
        self._config_token = jnp.asarray(
            tuple(hashlib.sha256(_canonical_json_bytes(payload)).digest()),
            dtype=jnp.uint8,
        )

    @property
    def config(self) -> PrototypeFactorizedPartnerPlannerV2Config:
        return self._config

    @property
    def behavior_model(self) -> BehaviorModel:
        return self._behavior

    @property
    def grounded_world_model(self) -> GroundedJointWorldModel:
        return self._grounded

    def feature_route(self, agent_index: int) -> HCCLFeatureConsumerRoute:
        if type(agent_index) is not int or agent_index not in (0, 1):
            raise ValueError("agent_index must be the exact dyad index 0 or 1")
        return self._routes[agent_index]

    def to_config(self) -> dict[str, object]:
        return self._config.to_config()

    def _state_content_token(
        self,
        state: PrototypeFactorizedPartnerPlannerV2State,
    ) -> Array:
        return _digest_tree(
            PROTOTYPE_FACTORIZED_PARTNER_PLANNER_V2_STATE_SCHEMA,
            state.config_token,
            state.agent_0,
            state.agent_1,
        )

    def _seal_state(
        self,
        state: PrototypeFactorizedPartnerPlannerV2State,
    ) -> PrototypeFactorizedPartnerPlannerV2State:
        return cast(
            PrototypeFactorizedPartnerPlannerV2State,
            cast(Any, state).replace(content_token=self._state_content_token(state)),
        )

    def _require_behavior_contract(self, state: BehaviorModelState, *, name: str) -> None:
        if type(state) is not BehaviorModelState:
            raise TypeError(f"{name} must be an exact BehaviorModelState")
        _require_array(
            state.weights,
            name=f"{name}.weights",
            shape=(N_ACTIONS, HCCL_FEATURE_TOTAL_DIM),
            dtype=jnp.float32,
        )
        _require_array(
            state.bias,
            name=f"{name}.bias",
            shape=(N_ACTIONS,),
            dtype=jnp.float32,
        )
        _require_key(state.rng_key, name=f"{name}.rng_key")
        _require_array(
            state.step_count,
            name=f"{name}.step_count",
            shape=(),
            dtype=jnp.int32,
        )
        _require_array(
            state.step_words,
            name=f"{name}.step_words",
            shape=(2,),
            dtype=jnp.uint32,
        )
        for field_name in ("nll_ema", "accuracy_ema", "confidence_ema"):
            _require_array(
                getattr(state, field_name),
                name=f"{name}.{field_name}",
                shape=(),
                dtype=jnp.float32,
            )

    def _require_grounded_contract(
        self,
        state: GroundedJointWorldModelState,
        *,
        name: str,
    ) -> None:
        if type(state) is not GroundedJointWorldModelState:
            raise TypeError(f"{name} must be an exact GroundedJointWorldModelState")
        _require_array(
            state.weights,
            name=f"{name}.weights",
            shape=(N_ACTIONS**2, GROUNDED_TARGET_DIM, HCCL_FEATURE_TOTAL_DIM),
            dtype=jnp.float32,
        )
        _require_array(
            state.bias,
            name=f"{name}.bias",
            shape=(N_ACTIONS**2, GROUNDED_TARGET_DIM),
            dtype=jnp.float32,
        )
        _require_array(
            state.update_count,
            name=f"{name}.update_count",
            shape=(),
            dtype=jnp.int32,
        )
        _require_array(
            state.update_words,
            name=f"{name}.update_words",
            shape=(2,),
            dtype=jnp.uint32,
        )

    def _require_cache_contract(
        self,
        cache: PrototypeFactorizedPartnerPlannerV2Cache,
        *,
        name: str,
    ) -> None:
        if type(cache) is not PrototypeFactorizedPartnerPlannerV2Cache:
            raise TypeError(f"{name} must be an exact planner-v2 cache")
        checks = (
            (cache.ledger_content_token, (32,), jnp.uint8, "ledger_content_token"),
            (cache.representation, (35,), jnp.float32, "representation"),
            (cache.behavior_step_words, (2,), jnp.uint32, "behavior_step_words"),
            (cache.grounded_update_words, (2,), jnp.uint32, "grounded_update_words"),
            (cache.partner_belief, (2,), jnp.float32, "partner_belief"),
            (
                cache.partner_belief_by_own_action,
                (2, 2),
                jnp.float32,
                "partner_belief_by_own_action",
            ),
            (
                cache.world_raw_predictions,
                (2, 2, 21),
                jnp.float32,
                "world_raw_predictions",
            ),
            (cache.world_cell_valid, (2, 2), jnp.bool_, "world_cell_valid"),
            (cache.expected_net_rewards, (2,), jnp.float32, "expected_net_rewards"),
            (cache.prepared_action, (), jnp.int32, "prepared_action"),
        )
        for value, shape, dtype, field_name in checks:
            _require_array(
                value,
                name=f"{name}.{field_name}",
                shape=shape,
                dtype=dtype,
            )

    def _require_agent_contract(
        self,
        agent: PrototypeFactorizedPartnerPlannerV2AgentState,
        *,
        name: str,
    ) -> None:
        if type(agent) is not PrototypeFactorizedPartnerPlannerV2AgentState:
            raise TypeError(f"{name} must be an exact planner-v2 agent state")
        self._routes[int(name[-1])]._require_ledger_contract(agent.ledger)
        self._require_behavior_contract(agent.behavior, name=f"{name}.behavior")
        self._require_grounded_contract(agent.grounded, name=f"{name}.grounded")
        self._require_cache_contract(agent.cache, name=f"{name}.cache")

    def _require_state_contract(
        self,
        state: PrototypeFactorizedPartnerPlannerV2State,
    ) -> None:
        if type(state) is not PrototypeFactorizedPartnerPlannerV2State:
            raise TypeError("state must be an exact PrototypeFactorizedPartnerPlannerV2State")
        _require_array(
            state.config_token,
            name="state.config_token",
            shape=(_TOKEN_NBYTES,),
            dtype=jnp.uint8,
        )
        _require_array(
            state.content_token,
            name="state.content_token",
            shape=(_TOKEN_NBYTES,),
            dtype=jnp.uint8,
        )
        self._require_agent_contract(state.agent_0, name="agent_0")
        self._require_agent_contract(state.agent_1, name="agent_1")

    def _behavior_semantics_valid(self, state: BehaviorModelState) -> bool:
        diagnostics = np.asarray(
            (
                float(_host(state.nll_ema)),
                float(_host(state.accuracy_ema)),
                float(_host(state.confidence_ema)),
            ),
            dtype=np.float64,
        )
        return bool(
            np.all(np.isfinite(_host(state.weights)))
            and np.all(np.isfinite(_host(state.bias)))
            and np.all(np.isfinite(diagnostics))
            and diagnostics[0] >= 0.0
            and 0.0 <= diagnostics[1] <= 1.0
            and 0.0 <= diagnostics[2] <= 1.0
            and _counter_valid(state.step_words, state.step_count)
        )

    def _grounded_semantics_valid(self, state: GroundedJointWorldModelState) -> bool:
        weights = _host(state.weights)
        bias = _host(state.bias)
        bound = self._config.max_parameter_magnitude
        return bool(
            np.all(np.isfinite(weights))
            and np.all(np.abs(weights) <= bound)
            and np.all(np.isfinite(bias))
            and np.all(np.abs(bias) <= bound)
            and _counter_valid(state.update_words, state.update_count)
        )

    @staticmethod
    def _representation_matches_ledger(
        representation: Array,
        ledger: HCCLFeatureBirthLedger,
    ) -> bool:
        """Check the representation values whose semantics the ledger fixes.

        Physical, live-context, and fast values are supplied by the outer
        representation owner.  This planner owns only the exact pair
        realization and positive-zero scrubbing of inactive context/pair
        coordinates.
        """

        values = np.ascontiguousarray(_host(representation))
        active = _host(ledger.active).astype(np.bool_)
        descriptors = _host(ledger.descriptor)
        value_bits = values.view(np.uint32)
        valid = True
        for slot in range(HCCL_FEATURE_CONTEXT_START, HCCL_FEATURE_FAST_START):
            if not active[slot]:
                valid &= bool(value_bits[slot] == np.uint32(0))
        for local in range(HCCL_FEATURE_PAIR_SLOTS):
            slot = HCCL_FEATURE_PAIR_START + local
            if not active[slot]:
                valid &= bool(value_bits[slot] == np.uint32(0))
                continue
            left, right = (int(item) for item in descriptors[slot])
            parents_valid = (
                0 <= left < HCCL_FEATURE_CONTEXT_START
                and 0 <= right < HCCL_FEATURE_CONTEXT_START
                and left < right
            )
            if not parents_valid:
                valid = False
                continue
            expected = jnp.asarray(
                representation[left] * representation[right],
                dtype=jnp.float32,
            )
            valid &= _array_exact_equal(representation[slot], expected)
        return bool(valid)

    def _cache_semantics_valid(
        self,
        agent: PrototypeFactorizedPartnerPlannerV2AgentState,
    ) -> bool:
        cache = agent.cache
        expected = self._build_cache(
            agent.ledger,
            agent.behavior,
            agent.grounded,
            cache.representation,
        )
        representation_values = _host(cache.representation)
        representation_valid = bool(
            np.all(np.isfinite(representation_values))
            and np.all(
                np.abs(representation_values) <= self._config.max_input_magnitude
            )
            and self._representation_matches_ledger(
                cache.representation,
                agent.ledger,
            )
        )
        expected_outputs_valid = bool(
            _probabilities_valid(expected.partner_belief)
            and np.all(np.isfinite(_host(expected.world_raw_predictions)))
            and np.all(_host(expected.world_cell_valid))
        )
        return bool(
            representation_valid
            and expected_outputs_valid
            and _tree_exact_equal(cache, expected)
        )

    def _agent_semantics_valid(
        self,
        agent: PrototypeFactorizedPartnerPlannerV2AgentState,
        *,
        index: int,
    ) -> bool:
        ledger_valid = _host_bool(self._routes[index].ledger_valid(agent.ledger))
        behavior_valid = self._behavior_semantics_valid(agent.behavior)
        grounded_valid = self._grounded_semantics_valid(agent.grounded)
        model_clocks_match = _array_exact_equal(
            agent.behavior.step_words,
            agent.grounded.update_words,
        )
        ledger_clock_matches = _array_exact_equal(
            agent.behavior.step_words,
            agent.ledger.source_clock_words,
        )
        cache_valid = self._cache_semantics_valid(agent)
        return bool(
            all(
                (
                    ledger_valid,
                    behavior_valid,
                    grounded_valid,
                    model_clocks_match,
                    ledger_clock_matches,
                    cache_valid,
                )
            )
        )

    def state_valid(
        self,
        state: PrototypeFactorizedPartnerPlannerV2State,
    ) -> Bool[Array, ""]:
        """Validate exact composition, model clocks, cache binding, and token."""

        self._require_state_contract(state)
        if _contains_tracer(state):
            raise TypeError("planner-v2 state validity is host/eager-only")
        config_valid = _array_exact_equal(state.config_token, self._config_token)
        content_valid = _array_exact_equal(
            state.content_token,
            self._state_content_token(state),
        )
        agent_0_valid = self._agent_semantics_valid(state.agent_0, index=0)
        agent_1_valid = self._agent_semantics_valid(state.agent_1, index=1)
        valid = bool(all((config_valid, content_valid, agent_0_valid, agent_1_valid)))
        return jnp.asarray(valid, dtype=jnp.bool_)

    def _build_cache(
        self,
        ledger: HCCLFeatureBirthLedger,
        behavior: BehaviorModelState,
        grounded: GroundedJointWorldModelState,
        representation: Array,
    ) -> PrototypeFactorizedPartnerPlannerV2Cache:
        belief = self._behavior.predict_probabilities(behavior, representation)
        predictions = tuple(
            self._grounded.predict(
                grounded,
                representation,
                jnp.asarray(own_action, dtype=jnp.int32),
                jnp.asarray(partner_action, dtype=jnp.int32),
            )
            for own_action in range(N_ACTIONS)
            for partner_action in range(N_ACTIONS)
        )
        raw = jnp.stack(tuple(item.raw_predictions for item in predictions)).reshape(
            (N_ACTIONS, N_ACTIONS, GROUNDED_TARGET_DIM)
        )
        cell_valid = jnp.stack(tuple(item.valid for item in predictions)).reshape(
            (N_ACTIONS, N_ACTIONS)
        )
        repeated_belief = jnp.broadcast_to(belief, (N_ACTIONS, N_ACTIONS))
        expected = raw[:, :, NET_REWARD_OUTPUT_INDEX] @ belief
        action = jnp.argmax(expected).astype(jnp.int32)
        return PrototypeFactorizedPartnerPlannerV2Cache(
            ledger_content_token=ledger.content_token,
            representation=representation,
            behavior_step_words=behavior.step_words,
            grounded_update_words=grounded.update_words,
            partner_belief=belief,
            partner_belief_by_own_action=repeated_belief,
            world_raw_predictions=raw,
            world_cell_valid=cell_valid,
            expected_net_rewards=expected,
            prepared_action=action,
        )

    def init(
        self,
        key: Array,
        *,
        ledger_agent_0: HCCLFeatureBirthLedger,
        ledger_agent_1: HCCLFeatureBirthLedger,
        representations: Array,
    ) -> PrototypeFactorizedPartnerPlannerV2State:
        """Initialize both donor pairs and bind their first four-cell caches."""

        _require_key(key, name="key")
        reps = _require_array(
            representations,
            name="representations",
            shape=(N_AGENTS, HCCL_FEATURE_TOTAL_DIM),
            dtype=jnp.float32,
        )
        ledgers = (ledger_agent_0, ledger_agent_1)
        for index, ledger in enumerate(ledgers):
            self._routes[index]._require_ledger_contract(ledger)
            if not _host_bool(self._routes[index].ledger_valid(ledger)):
                raise ValueError(f"ledger_agent_{index} is invalid")
        if _contains_tracer((key, ledgers, reps)):
            raise TypeError("planner-v2 initialization is host/eager-only")
        keys = jr.split(key, 4)
        behaviors = (
            self._behavior.init(HCCL_FEATURE_TOTAL_DIM, keys[0]),
            self._behavior.init(HCCL_FEATURE_TOTAL_DIM, keys[2]),
        )
        grounded = (
            self._grounded.init(keys[1]),
            self._grounded.init(keys[3]),
        )
        agents = tuple(
            PrototypeFactorizedPartnerPlannerV2AgentState(
                ledger=ledgers[index],
                behavior=behaviors[index],
                grounded=grounded[index],
                cache=self._build_cache(
                    ledgers[index],
                    behaviors[index],
                    grounded[index],
                    reps[index],
                ),
            )
            for index in range(N_AGENTS)
        )
        unsigned = PrototypeFactorizedPartnerPlannerV2State(
            config_token=self._config_token,
            content_token=jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8),
            agent_0=agents[0],
            agent_1=agents[1],
        )
        state = self._seal_state(unsigned)
        if not _host_bool(self.state_valid(state)):
            raise ValueError("initial planner-v2 composition is invalid")
        return state

    @staticmethod
    def _route_feature_axis(values: Array, route_map: HCCLFeatureConsumerRouteMap) -> Array:
        source_slots = jnp.clip(route_map.source_slots, 0, HCCL_FEATURE_TOTAL_DIM - 1)
        gathered = jnp.take(values, source_slots, axis=-1)
        mask_shape = (1,) * (values.ndim - 1) + (HCCL_FEATURE_TOTAL_DIM,)
        survivor = jnp.reshape(route_map.survivor_mask, mask_shape)
        return jnp.where(survivor, gathered, jnp.zeros_like(gathered))

    @staticmethod
    def _masked_positive_zero(values: Array, mask: Array) -> bool:
        host_values = _host(values)
        host_mask = _host(mask).astype(np.bool_)
        selected = host_values[..., host_mask]
        return bool(np.all(selected.view(np.uint32) == np.uint32(0)))

    @staticmethod
    def _behavior_nonfeature_equal(
        routed: BehaviorModelState,
        updated: BehaviorModelState,
    ) -> bool:
        return all(
            (
                _array_exact_equal(routed.bias, updated.bias),
                _array_exact_equal(routed.rng_key, updated.rng_key),
                _array_exact_equal(routed.step_count, updated.step_count),
                _array_exact_equal(routed.step_words, updated.step_words),
                _array_exact_equal(routed.nll_ema, updated.nll_ema),
                _array_exact_equal(routed.accuracy_ema, updated.accuracy_ema),
                _array_exact_equal(routed.confidence_ema, updated.confidence_ema),
            )
        )

    @staticmethod
    def _grounded_nonfeature_equal(
        routed: GroundedJointWorldModelState,
        updated: GroundedJointWorldModelState,
    ) -> bool:
        return all(
            (
                _array_exact_equal(routed.bias, updated.bias),
                _array_exact_equal(routed.update_count, updated.update_count),
                _array_exact_equal(routed.update_words, updated.update_words),
            )
        )

    def _route_models(
        self,
        behavior: BehaviorModelState,
        grounded: GroundedJointWorldModelState,
        route_map: HCCLFeatureConsumerRouteMap,
    ) -> tuple[BehaviorModelState, GroundedJointWorldModelState]:
        routed_behavior = cast(
            BehaviorModelState,
            cast(Any, behavior).replace(
                weights=self._route_feature_axis(behavior.weights, route_map)
            ),
        )
        routed_grounded = cast(
            GroundedJointWorldModelState,
            cast(Any, grounded).replace(
                weights=self._route_feature_axis(grounded.weights, route_map)
            ),
        )
        return routed_behavior, routed_grounded

    def _require_transition_inputs(
        self,
        state: PrototypeFactorizedPartnerPlannerV2State,
        route_results: tuple[HCCLFeatureConsumerRouteResult, ...],
        source_representations: Array,
        destination_representations: Array,
        executed_actions: Array,
        next_physical_observations: Array,
        task_score: Array,
        safety_costs: Array,
        message_charges: Array,
        net_rewards: Array,
        discount: Array,
    ) -> None:
        self._require_state_contract(state)
        if len(route_results) != N_AGENTS or any(
            type(result) is not HCCLFeatureConsumerRouteResult for result in route_results
        ):
            raise TypeError("both route results must be exact HCCLFeatureConsumerRouteResult")
        checks = (
            (source_representations, (2, 35), jnp.float32, "source_representations"),
            (
                destination_representations,
                (2, 35),
                jnp.float32,
                "destination_representations",
            ),
            (executed_actions, (2,), jnp.int32, "executed_actions"),
            (
                next_physical_observations,
                (2, 16),
                jnp.float32,
                "next_physical_observations",
            ),
            (task_score, (), jnp.float32, "task_score"),
            (safety_costs, (2,), jnp.float32, "safety_costs"),
            (message_charges, (2,), jnp.float32, "message_charges"),
            (net_rewards, (2,), jnp.float32, "net_rewards"),
            (discount, (), jnp.float32, "discount"),
        )
        for value, shape, dtype, name in checks:
            _require_array(value, name=name, shape=shape, dtype=dtype)

    def observe_route_and_plan(
        self,
        state: PrototypeFactorizedPartnerPlannerV2State,
        *,
        route_result_agent_0: HCCLFeatureConsumerRouteResult,
        route_result_agent_1: HCCLFeatureConsumerRouteResult,
        source_representations: Array,
        destination_representations: Array,
        executed_actions: Array,
        next_physical_observations: Array,
        task_score: Array,
        safety_costs: Array,
        message_charges: Array,
        net_rewards: Array,
        discount: Array,
    ) -> PrototypeFactorizedPartnerPlannerV2Result:
        """Update source models once, route, cache four cells, and prepare P."""

        route_results = (route_result_agent_0, route_result_agent_1)
        self._require_transition_inputs(
            state,
            route_results,
            source_representations,
            destination_representations,
            executed_actions,
            next_physical_observations,
            task_score,
            safety_costs,
            message_charges,
            net_rewards,
            discount,
        )
        if _contains_tracer(
            (
                state,
                route_results,
                source_representations,
                destination_representations,
                executed_actions,
                next_physical_observations,
                task_score,
                safety_costs,
                message_charges,
                net_rewards,
                discount,
            )
        ):
            raise TypeError("planner-v2 transition is host/eager-only")

        agents = (state.agent_0, state.agent_1)
        source_valid = _host_bool(self.state_valid(state))
        source_cache_matches = np.asarray(
            [
                _array_exact_equal(
                    agents[index].cache.representation,
                    source_representations[index],
                )
                for index in range(N_AGENTS)
            ],
            dtype=np.bool_,
        )
        source_representation_matches_ledger = np.asarray(
            [
                self._representation_matches_ledger(
                    source_representations[index],
                    agents[index].ledger,
                )
                for index in range(N_AGENTS)
            ],
            dtype=np.bool_,
        )
        destination_representation_matches_ledger = np.asarray(
            [
                self._representation_matches_ledger(
                    destination_representations[index],
                    route_results[index].ledger,
                )
                for index in range(N_AGENTS)
            ],
            dtype=np.bool_,
        )
        finite_bound = self._config.max_input_magnitude
        expected_net = task_score - message_charges - safety_costs
        event_inputs_valid = bool(
            np.all(np.isfinite(_host(source_representations)))
            and np.all(np.abs(_host(source_representations)) <= finite_bound)
            and np.all(np.isfinite(_host(destination_representations)))
            and np.all(np.abs(_host(destination_representations)) <= finite_bound)
            and np.all(np.isfinite(_host(next_physical_observations)))
            and np.all(np.abs(_host(next_physical_observations)) <= finite_bound)
            and np.all(np.isfinite(_host(safety_costs)))
            and np.all(_host(safety_costs) >= 0.0)
            and np.all(np.isfinite(_host(message_charges)))
            and np.all(_host(message_charges) >= 0.0)
            and np.all(np.isfinite(_host(net_rewards)))
            and np.isfinite(float(_host(task_score)))
            and np.isfinite(float(_host(discount)))
            and 0.0 <= float(_host(discount)) <= 1.0
            and np.all(_host(executed_actions) >= 0)
            and np.all(_host(executed_actions) < N_ACTIONS)
            and _array_exact_equal(net_rewards, expected_net)
            and np.all(source_cache_matches)
            and np.all(source_representation_matches_ledger)
            and np.all(destination_representation_matches_ledger)
        )
        partner_actions = jnp.stack((executed_actions[1], executed_actions[0])).astype(
            jnp.int32
        )
        grounded_targets = jnp.stack(
            tuple(
                jnp.concatenate(
                    (
                        next_physical_observations[index],
                        jnp.reshape(task_score, (1,)),
                        jnp.reshape(safety_costs[index], (1,)),
                        jnp.reshape(message_charges[index], (1,)),
                        jnp.reshape(net_rewards[index], (1,)),
                        jnp.reshape(discount, (1,)),
                    )
                )
                for index in range(N_AGENTS)
            )
        )

        behavior_updates: tuple[BehaviorModelUpdateResult, ...] = tuple(
            self._behavior.update(
                agents[index].behavior,
                source_representations[index],
                partner_actions[index],
            )
            for index in range(N_AGENTS)
        )
        grounded_updates: tuple[GroundedJointWorldUpdateResult, ...] = tuple(
            self._grounded.update(
                agents[index].grounded,
                source_representations[index],
                executed_actions[index],
                partner_actions[index],
                grounded_targets[index, :GROUNDED_TARGET_OBSERVATION_DIM],
                grounded_targets[index, NET_REWARD_OUTPUT_INDEX],
                grounded_targets[index, DISCOUNT_OUTPUT_INDEX],
            )
            for index in range(N_AGENTS)
        )
        behavior_applied = np.asarray(
            [_host_bool(update.update_applied) for update in behavior_updates],
            dtype=np.bool_,
        )
        grounded_applied = np.asarray(
            [_host_bool(update.update_applied) for update in grounded_updates],
            dtype=np.bool_,
        )
        target_order_valid = np.asarray(
            [
                _array_exact_equal(grounded_updates[index].targets, grounded_targets[index])
                for index in range(N_AGENTS)
            ],
            dtype=np.bool_,
        )
        clocks_advanced_once = np.asarray(
            [
                _words_successor(
                    agents[index].behavior.step_words,
                    behavior_updates[index].state.step_words,
                )
                and _words_successor(
                    agents[index].grounded.update_words,
                    grounded_updates[index].state.update_words,
                )
                and _array_exact_equal(
                    behavior_updates[index].state.step_words,
                    grounded_updates[index].state.update_words,
                )
                for index in range(N_AGENTS)
            ],
            dtype=np.bool_,
        )
        source_update_phase_valid = bool(
            source_valid
            and event_inputs_valid
            and np.all(behavior_applied)
            and np.all(grounded_applied)
            and np.all(target_order_valid)
            and np.all(clocks_advanced_once)
            and np.all(source_representation_matches_ledger)
        )
        source_update_receipt = PrototypeFactorizedPartnerPlannerV2SourceUpdateReceipt(
            source_state_content_token=state.content_token,
            source_ledger_content_tokens=jnp.stack(
                tuple(agent.ledger.content_token for agent in agents)
            ),
            source_representations=source_representations,
            executed_actions=executed_actions,
            observed_partner_actions=partner_actions,
            grounded_targets=grounded_targets,
            behavior_pre_step_words=jnp.stack(
                tuple(agent.behavior.step_words for agent in agents)
            ),
            behavior_post_step_words=jnp.stack(
                tuple(update.state.step_words for update in behavior_updates)
            ),
            grounded_pre_update_words=jnp.stack(
                tuple(agent.grounded.update_words for agent in agents)
            ),
            grounded_post_update_words=jnp.stack(
                tuple(update.state.update_words for update in grounded_updates)
            ),
            source_cache_matches=jnp.asarray(source_cache_matches, dtype=jnp.bool_),
            source_representation_matches_ledger=jnp.asarray(
                source_representation_matches_ledger,
                dtype=jnp.bool_,
            ),
            behavior_update_applied=jnp.asarray(behavior_applied, dtype=jnp.bool_),
            grounded_update_applied=jnp.asarray(grounded_applied, dtype=jnp.bool_),
            grounded_target_order_valid=jnp.asarray(target_order_valid, dtype=jnp.bool_),
            clocks_advanced_once=jnp.asarray(clocks_advanced_once, dtype=jnp.bool_),
            phase_valid=jnp.asarray(source_update_phase_valid, dtype=jnp.bool_),
        )

        route_integrity = np.asarray(
            [
                _host_bool(
                    self._routes[index].result_integrity_valid(
                        agents[index].ledger,
                        route_results[index],
                    )
                )
                for index in range(N_AGENTS)
            ],
            dtype=np.bool_,
        )
        route_applied = np.asarray(
            [
                _host_bool(route_results[index].witness.transaction_applied)
                for index in range(N_AGENTS)
            ],
            dtype=np.bool_,
        )
        routed_models = tuple(
            self._route_models(
                behavior_updates[index].state,
                grounded_updates[index].state,
                route_results[index].witness.route_map,
            )
            for index in range(N_AGENTS)
        )
        clock_matches_destination = np.asarray(
            [
                _array_exact_equal(
                    routed_models[index][0].step_words,
                    route_results[index].ledger.source_clock_words,
                )
                and _array_exact_equal(
                    routed_models[index][1].update_words,
                    route_results[index].ledger.source_clock_words,
                )
                for index in range(N_AGENTS)
            ],
            dtype=np.bool_,
        )
        survivor_exact: list[bool] = []
        newborn_zero: list[bool] = []
        inactive_zero: list[bool] = []
        nonfeature_exact: list[bool] = []
        for index in range(N_AGENTS):
            route_map = route_results[index].witness.route_map
            updated_behavior = behavior_updates[index].state
            updated_grounded = grounded_updates[index].state
            routed_behavior, routed_grounded = routed_models[index]
            expected_behavior_weights = self._route_feature_axis(
                updated_behavior.weights,
                route_map,
            )
            expected_grounded_weights = self._route_feature_axis(
                updated_grounded.weights,
                route_map,
            )
            survivor_exact.append(
                _array_exact_equal(routed_behavior.weights, expected_behavior_weights)
                and _array_exact_equal(routed_grounded.weights, expected_grounded_weights)
            )
            newborn_zero.append(
                self._masked_positive_zero(
                    routed_behavior.weights,
                    route_map.newborn_mask,
                )
                and self._masked_positive_zero(
                    routed_grounded.weights,
                    route_map.newborn_mask,
                )
            )
            inactive_zero.append(
                self._masked_positive_zero(
                    routed_behavior.weights,
                    route_map.inactive_mask,
                )
                and self._masked_positive_zero(
                    routed_grounded.weights,
                    route_map.inactive_mask,
                )
            )
            nonfeature_exact.append(
                self._behavior_nonfeature_equal(routed_behavior, updated_behavior)
                and self._grounded_nonfeature_equal(routed_grounded, updated_grounded)
            )
        source_update_preceded_route = np.asarray(
            [
                survivor_exact[index]
                and _array_exact_equal(
                    routed_models[index][0].step_words,
                    behavior_updates[index].post_step_words,
                )
                and _array_exact_equal(
                    routed_models[index][1].update_words,
                    grounded_updates[index].post_update_words,
                )
                for index in range(N_AGENTS)
            ],
            dtype=np.bool_,
        )
        route_phase_valid = bool(
            np.all(route_integrity)
            and np.all(route_applied)
            and np.all(clock_matches_destination)
            and all(survivor_exact)
            and all(newborn_zero)
            and all(inactive_zero)
            and all(nonfeature_exact)
            and np.all(source_update_preceded_route)
            and np.all(destination_representation_matches_ledger)
        )
        route_receipt = PrototypeFactorizedPartnerPlannerV2RouteReceipt(
            route_witness_content_tokens=jnp.stack(
                tuple(result.witness.content_token for result in route_results)
            ),
            destination_ledger_content_tokens=jnp.stack(
                tuple(result.ledger.content_token for result in route_results)
            ),
            route_result_integrity_valid=jnp.asarray(route_integrity, dtype=jnp.bool_),
            route_transaction_applied=jnp.asarray(route_applied, dtype=jnp.bool_),
            destination_representation_matches_ledger=jnp.asarray(
                destination_representation_matches_ledger,
                dtype=jnp.bool_,
            ),
            post_update_clock_matches_destination=jnp.asarray(
                clock_matches_destination,
                dtype=jnp.bool_,
            ),
            survivor_columns_bit_exact=jnp.asarray(survivor_exact, dtype=jnp.bool_),
            newborn_columns_positive_zero=jnp.asarray(newborn_zero, dtype=jnp.bool_),
            inactive_columns_positive_zero=jnp.asarray(inactive_zero, dtype=jnp.bool_),
            nonfeature_fields_bit_exact=jnp.asarray(nonfeature_exact, dtype=jnp.bool_),
            source_update_preceded_route=jnp.asarray(
                source_update_preceded_route,
                dtype=jnp.bool_,
            ),
            phase_valid=jnp.asarray(route_phase_valid, dtype=jnp.bool_),
        )

        candidate_agents = tuple(
            PrototypeFactorizedPartnerPlannerV2AgentState(
                ledger=route_results[index].ledger,
                behavior=routed_models[index][0],
                grounded=routed_models[index][1],
                cache=self._build_cache(
                    route_results[index].ledger,
                    routed_models[index][0],
                    routed_models[index][1],
                    destination_representations[index],
                ),
            )
            for index in range(N_AGENTS)
        )
        destination_cache_bound = np.asarray(
            [self._cache_semantics_valid(candidate_agents[index]) for index in range(N_AGENTS)],
            dtype=np.bool_,
        )
        identical_belief = np.asarray(
            [
                _array_exact_equal(
                    candidate_agents[index].cache.partner_belief_by_own_action,
                    jnp.broadcast_to(
                        candidate_agents[index].cache.partner_belief,
                        (N_ACTIONS, N_ACTIONS),
                    ),
                )
                for index in range(N_AGENTS)
            ],
            dtype=np.bool_,
        )
        clocks_unchanged_by_planning = np.asarray(
            [
                _array_exact_equal(
                    candidate_agents[index].behavior.step_words,
                    routed_models[index][0].step_words,
                )
                and _array_exact_equal(
                    candidate_agents[index].grounded.update_words,
                    routed_models[index][1].update_words,
                )
                for index in range(N_AGENTS)
            ],
            dtype=np.bool_,
        )
        plan_phase_valid = bool(
            np.all(destination_cache_bound)
            and np.all(identical_belief)
            and np.all(clocks_unchanged_by_planning)
            and all(
                np.all(_host(candidate_agents[index].cache.world_cell_valid))
                for index in range(N_AGENTS)
            )
        )
        plan_receipt = PrototypeFactorizedPartnerPlannerV2PlanReceipt(
            destination_representations=destination_representations,
            partner_belief=jnp.stack(
                tuple(agent.cache.partner_belief for agent in candidate_agents)
            ),
            partner_belief_by_own_action=jnp.stack(
                tuple(
                    agent.cache.partner_belief_by_own_action for agent in candidate_agents
                )
            ),
            world_raw_predictions=jnp.stack(
                tuple(agent.cache.world_raw_predictions for agent in candidate_agents)
            ),
            world_cell_valid=jnp.stack(
                tuple(agent.cache.world_cell_valid for agent in candidate_agents)
            ),
            expected_net_rewards=jnp.stack(
                tuple(agent.cache.expected_net_rewards for agent in candidate_agents)
            ),
            proposed_actions=jnp.stack(
                tuple(agent.cache.prepared_action for agent in candidate_agents)
            ),
            identical_partner_belief_across_own_rows=jnp.asarray(
                identical_belief,
                dtype=jnp.bool_,
            ),
            joint_cells_evaluated_per_agent=jnp.full(
                (N_AGENTS,),
                N_ACTIONS**2,
                dtype=jnp.int32,
            ),
            destination_cache_bound=jnp.asarray(
                destination_cache_bound,
                dtype=jnp.bool_,
            ),
            model_clocks_unchanged_by_planning=jnp.asarray(
                clocks_unchanged_by_planning,
                dtype=jnp.bool_,
            ),
            phase_valid=jnp.asarray(plan_phase_valid, dtype=jnp.bool_),
        )
        unsigned_candidate = PrototypeFactorizedPartnerPlannerV2State(
            config_token=state.config_token,
            content_token=jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8),
            agent_0=candidate_agents[0],
            agent_1=candidate_agents[1],
        )
        candidate = self._seal_state(unsigned_candidate)
        candidate_valid = _host_bool(self.state_valid(candidate))
        transaction_applied = bool(
            source_valid
            and event_inputs_valid
            and source_update_phase_valid
            and route_phase_valid
            and plan_phase_valid
            and candidate_valid
        )
        selected = candidate if transaction_applied else state
        proposed_actions = plan_receipt.proposed_actions
        prepared_actions = (
            proposed_actions
            if transaction_applied
            else jnp.full((N_AGENTS,), -1, dtype=jnp.int32)
        )
        receipt = PrototypeFactorizedPartnerPlannerV2Receipt(
            source_update=source_update_receipt,
            feature_route=route_receipt,
            plan=plan_receipt,
            source_state_valid=jnp.asarray(source_valid, dtype=jnp.bool_),
            event_inputs_valid=jnp.asarray(event_inputs_valid, dtype=jnp.bool_),
            candidate_state_valid=jnp.asarray(candidate_valid, dtype=jnp.bool_),
            transaction_applied=jnp.asarray(transaction_applied, dtype=jnp.bool_),
            complete_source_returned=jnp.asarray(
                not transaction_applied,
                dtype=jnp.bool_,
            ),
        )
        return PrototypeFactorizedPartnerPlannerV2Result(
            state=selected,
            candidate_state=candidate,
            prepared_actions=prepared_actions,
            receipt=receipt,
            work=_fixed_work(),
            transaction_applied=jnp.asarray(transaction_applied, dtype=jnp.bool_),
            complete_source_returned=jnp.asarray(
                not transaction_applied,
                dtype=jnp.bool_,
            ),
        )


__all__ = [
    "DISCOUNT_OUTPUT_INDEX",
    "GROUNDED_OUTPUT_ORDER",
    "GROUNDED_TARGET_DIM",
    "MESSAGE_CHARGE_OUTPUT_INDEX",
    "NET_REWARD_OUTPUT_INDEX",
    "N_ACTIONS",
    "N_AGENTS",
    "PHYSICAL_OUTPUT_DIM",
    "PROTOTYPE_FACTORIZED_PARTNER_PLANNER_V2_DISPATCH_AUTHORITY",
    "PROTOTYPE_FACTORIZED_PARTNER_PLANNER_V2_SCHEMA",
    "PROTOTYPE_FACTORIZED_PARTNER_PLANNER_V2_SCIENTIFIC_PROMOTION_ALLOWED",
    "PROTOTYPE_FACTORIZED_PARTNER_PLANNER_V2_STATE_SCHEMA",
    "PROTOTYPE_FACTORIZED_PARTNER_PLANNER_V2_STATUS",
    "PrototypeFactorizedPartnerPlannerV2",
    "PrototypeFactorizedPartnerPlannerV2AgentState",
    "PrototypeFactorizedPartnerPlannerV2Cache",
    "PrototypeFactorizedPartnerPlannerV2Config",
    "PrototypeFactorizedPartnerPlannerV2PlanReceipt",
    "PrototypeFactorizedPartnerPlannerV2Receipt",
    "PrototypeFactorizedPartnerPlannerV2Result",
    "PrototypeFactorizedPartnerPlannerV2RouteReceipt",
    "PrototypeFactorizedPartnerPlannerV2SourceUpdateReceipt",
    "PrototypeFactorizedPartnerPlannerV2State",
    "PrototypeFactorizedPartnerPlannerV2Work",
    "SAFETY_COST_OUTPUT_INDEX",
    "TASK_SCORE_OUTPUT_INDEX",
]
