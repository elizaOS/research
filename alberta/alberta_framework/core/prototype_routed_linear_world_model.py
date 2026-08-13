# mypy: disable-error-code="attr-defined,call-arg,no-any-return"
"""Bank-routed linear world learning with fixed physical prediction targets.

This module is a deliberately narrow L0 composition.  A linear learner may
condition on the complete Prototype representation
``phi_g(x) = [x, pair-products_g(x)]``, while its outputs remain the stable
physical targets ``[delta x, reward, discount]``.  When the feature bank is
replaced, only learned *input* columns are migrated:

* stable base and action columns are preserved bit-for-bit;
* descriptor-identical survivor columns move bit-for-bit;
* newborn and inactive columns are exact positive float32 zero; and
* evicted columns are deliberately forgotten.

Generated-feature output heads are unsupported.  An evicted output target and
a newborn descriptor denote different functions, so there is no lossless
remap of the just-observed old-bank target.  Predicting the stable base and
re-augmenting it under the live destination bank also keeps imagined states on
the exact pair-product manifold.

One real transition has a causal two-phase boundary.  Before the outcome is
known, ``prepare_transition`` snapshots the complete source world/router,
base/action, augmented input, and prediction.  After the outcome arrives,
``observe_and_route`` exact-tree revalidates the authoritative current world
and router, recomputes the cache/prediction, updates against the fixed physical
target, adds the physical successor to the bounded anchor cache, then applies
the authenticated source-to-destination input-column route.  Planning uses the
same split: ``prepare_plan`` binds complete source world/OaK/router state plus
the indexed physical anchor/action/proposal, and ``plan_one`` revalidates it
before carrying only OaK's base learner subtree.  Stale identity, clock, cache,
non-finite value, route, or consumer failures are atomic no-ops.

The mechanism has no calibrated gate, benefit result, retention result,
evidence, or scientific-promotion authority.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any, ClassVar, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core.dreaming import (
    RecentObservationBuffer,
    RecentObservationBufferState,
)
from alberta_framework.core.feature_bank_router import (
    FeatureBankRouteDiagnostics,
    FeatureBankRouter,
    FeatureBankRouterConfig,
    FeatureBankRouterState,
)
from alberta_framework.core.multi_head_learner import (
    MultiHeadMLPLearner,
    MultiHeadMLPState,
)
from alberta_framework.core.normalizers import (
    _checked_lifetime_words_increment,
    _lifetime_counter_valid,
    _saturating_int32_counter_increment,
)
from alberta_framework.core.oak import (
    OaKAgent,
    OaKConfig,
    OaKState,
    _oak_outer_state_validity,
)
from alberta_framework.core.optimizers import LMSState
from alberta_framework.core.prototype_feature_lifecycle import (
    PrototypeFeatureConsumerBinding,
    PrototypeFeatureLifecycle,
    PrototypeFeatureLifecycleConfig,
)
from alberta_framework.core.types import MLPParams
from alberta_framework.core.world_model import (
    ActionConditionedWorldModel,
    ActionConditionedWorldModelConfig,
)

PROTOTYPE_ROUTED_LINEAR_WORLD_CONFIG_SCHEMA = "alberta.prototype-routed-linear-world.config.v1"
PROTOTYPE_ROUTED_LINEAR_WORLD_STATE_SCHEMA = "alberta.prototype-routed-linear-world.state.v1"
PROTOTYPE_ROUTED_LINEAR_WORLD_MECHANISM_STATUS = "l0-mechanism-only-not-assessed"
PROTOTYPE_ROUTED_LINEAR_WORLD_SCIENTIFIC_PROMOTION_ALLOWED = False
PROTOTYPE_ROUTED_LINEAR_WORLD_EVIDENCE_LEVEL = "L0"

_INT32_MAX = 2_147_483_647
_UINT32_MAX = 4_294_967_295
_MAX_ANCHOR_CAPACITY = 4_096
_SCHEMA_DIGEST_NBYTES = 32
_FIXED_OUTPUT_SEMANTICS = "[normalized-delta-stable-base,reward,discount]"
_GENERATED_OUTPUT_SEMANTICS = "unsupported-non-remappable"
_ROUTE_SEMANTICS = (
    "source-update-first;stable-and-action-exact;survivor-by-descriptor-exact;"
    "newborn-and-inactive-positive-zero;evictee-dropped"
)
_PLANNING_SEMANTICS = (
    "physical-anchor;live-bank-input;physical-successor;live-bank-reaugmentation;"
    "one-oak-base-backup"
)


def _tree_nbytes(tree: object) -> int:
    """Return exact bytes in persistent array leaves."""

    total = 0
    for leaf in jax.tree.leaves(tree):
        if isinstance(leaf, Array):
            total += int(leaf.size) * int(leaf.dtype.itemsize)
    return total


def _require_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    """Reject shape or dtype drift without coercion."""

    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must be an array with exact shape and dtype")
    array = cast(Array, value)
    if tuple(array.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}; got {tuple(array.shape)}")
    expected = jnp.dtype(dtype)
    if jnp.dtype(array.dtype) != expected:
        raise TypeError(f"{name} must have dtype {expected}; got {array.dtype}")
    return array


def _tree_static_contract_matches(value: object, template: object) -> bool:
    """Compare exact PyTree structure plus array shape and dtype."""

    value_leaves, value_structure = jax.tree.flatten(value)
    template_leaves, template_structure = jax.tree.flatten(template)
    if cast(Any, value_structure) != cast(Any, template_structure) or len(value_leaves) != len(
        template_leaves
    ):
        return False
    for value_leaf, template_leaf in zip(
        value_leaves,
        template_leaves,
        strict=True,
    ):
        if isinstance(template_leaf, Array) or isinstance(value_leaf, Array):
            if not isinstance(template_leaf, Array):
                if not isinstance(template_leaf, (bool, int, float, np.generic)):
                    return False
                template_leaf = jnp.asarray(
                    template_leaf,
                    dtype=cast(Array, value_leaf).dtype,
                )
            if not isinstance(value_leaf, Array):
                if not isinstance(value_leaf, (bool, int, float, np.generic)):
                    return False
                value_leaf = jnp.asarray(value_leaf, dtype=template_leaf.dtype)
            if not isinstance(template_leaf, Array) or not isinstance(value_leaf, Array):
                return False
            if value_leaf.shape != template_leaf.shape or value_leaf.dtype != template_leaf.dtype:
                return False
        elif type(value_leaf) is not type(template_leaf):
            return False
    return True


def _tree_exactly_equal(left: object, right: object) -> Bool[Array, ""]:
    """Return bit-sensitive equality for two statically identical trees."""

    left_leaves, left_structure = jax.tree.flatten(left)
    right_leaves, right_structure = jax.tree.flatten(right)
    if cast(Any, left_structure) != cast(Any, right_structure) or len(left_leaves) != len(
        right_leaves
    ):
        return jnp.asarray(False, dtype=jnp.bool_)
    equal = jnp.asarray(True, dtype=jnp.bool_)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = jnp.asarray(left_leaf)
        right_array = jnp.asarray(right_leaf)
        if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
            return jnp.asarray(False, dtype=jnp.bool_)
        if jax.dtypes.issubdtype(left_array.dtype, jax.dtypes.prng_key):
            equal = equal & jnp.array_equal(
                jr.key_data(left_array),
                jr.key_data(right_array),
            )
        elif jnp.issubdtype(left_array.dtype, jnp.floating):
            if left_array.dtype == jnp.float32:
                equal = equal & jnp.array_equal(
                    jax.lax.bitcast_convert_type(left_array, jnp.uint32),
                    jax.lax.bitcast_convert_type(right_array, jnp.uint32),
                )
            else:
                equal = equal & jnp.array_equal(left_array, right_array)
        else:
            equal = equal & jnp.array_equal(left_array, right_array)
    return equal


def _floating_tree_is_finite(tree: object) -> Bool[Array, ""]:
    """Return whether every inexact array leaf is finite."""

    valid = jnp.asarray(True, dtype=jnp.bool_)
    for leaf in jax.tree.leaves(tree):
        array = jnp.asarray(leaf)
        if jnp.issubdtype(array.dtype, jnp.inexact):
            valid = valid & jnp.all(jnp.isfinite(array))
    return valid


def _telemetry_from_words(words: Array) -> Int[Array, ""]:
    """Project an exact uint64 identity to saturating int32 telemetry."""

    below = (words[0] == jnp.uint32(0)) & (words[1] < jnp.uint32(_INT32_MAX))
    return jnp.where(
        below,
        words[1].astype(jnp.int32),
        jnp.int32(_INT32_MAX),
    )


def _words_less_equal(left: Array, right: Array) -> Bool[Array, ""]:
    """Compare big-endian uint32 word pairs."""

    return (left[0] < right[0]) | ((left[0] == right[0]) & (left[1] <= right[1]))


def _words_successor(source: Array, destination: Array) -> Bool[Array, ""]:
    """Return whether destination is the non-wrapping source successor."""

    available = ~jnp.all(source == jnp.uint32(_UINT32_MAX))
    low = source[1] + jnp.uint32(1)
    carry = (low == jnp.uint32(0)).astype(jnp.uint32)
    candidate = jnp.stack((source[0] + carry, low)).astype(jnp.uint32)
    return available & jnp.all(destination == candidate)


def _checked_words_add(left: Array, right: Array) -> tuple[Array, Bool[Array, ""]]:
    """Add two uint64 word pairs and report non-wrapping capacity."""

    low = left[1] + right[1]
    carry = (low < left[1]).astype(jnp.uint32)
    high_without_carry = left[0] + right[0]
    overflow_high = high_without_carry < left[0]
    high = high_without_carry + carry
    overflow_carry = (carry != 0) & (high == jnp.uint32(0))
    return (
        jnp.stack((high, low)).astype(jnp.uint32),
        cast(Bool[Array, ""], ~(overflow_high | overflow_carry)),
    )


def _words_mod_small(words: Array, modulus: int) -> Int[Array, ""]:
    """Return one uint64 word pair modulo a small positive integer."""

    modulus_u = jnp.asarray(modulus, dtype=jnp.uint32)
    two32_mod = jnp.asarray((1 << 32) % modulus, dtype=jnp.uint32)
    high_term = (words[0] % modulus_u) * two32_mod
    return ((high_term + (words[1] % modulus_u)) % modulus_u).astype(jnp.int32)


def _words_at_least_small(words: Array, threshold: int) -> Bool[Array, ""]:
    """Compare an exact word pair with one int32-bounded threshold."""

    return cast(
        Bool[Array, ""],
        (words[0] != jnp.uint32(0)) | (words[1] >= jnp.asarray(threshold, dtype=jnp.uint32)),
    )


def _positive_int(value: object, *, name: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be a strict integer in [1, {maximum}]")
    return value


def _nonnegative_int(value: object, *, name: str, maximum: int) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{name} must be a strict integer in [0, {maximum}]")
    return value


def _finite_nonnegative_float32(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real non-boolean scalar")
    parsed = float(value)
    narrowed = float(np.float32(parsed))
    if not math.isfinite(parsed) or not math.isfinite(narrowed) or narrowed < 0.0:
        raise ValueError(f"{name} must be finite, non-negative, and float32-safe")
    return narrowed


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeRoutedLinearWorldConfig:
    """Exact fixed-output model, feature bank, OaK, cache, and guard contract."""

    feature_lifecycle: PrototypeFeatureLifecycleConfig
    world_model: ActionConditionedWorldModelConfig
    oak: OaKConfig
    anchor_capacity: int = 8
    planning_enabled: bool = False
    planning_warmup_steps: int = 1
    max_generation_model_error: float = 1_000_000.0
    max_planned_backups: int = _INT32_MAX
    carry_survivors: bool = True

    SCHEMA_VERSION: ClassVar[str] = PROTOTYPE_ROUTED_LINEAR_WORLD_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if type(self.feature_lifecycle) is not PrototypeFeatureLifecycleConfig:
            raise TypeError("feature_lifecycle must be an exact PrototypeFeatureLifecycleConfig")
        if type(self.world_model) is not ActionConditionedWorldModelConfig:
            raise TypeError("world_model must be an exact ActionConditionedWorldModelConfig")
        if type(self.oak) is not OaKConfig:
            raise TypeError("oak must be an exact OaKConfig")
        # Reuse the production model's scalar validation, then impose this
        # composition's narrower fixed-output linear contract.
        ActionConditionedWorldModel(self.world_model)
        lifecycle = PrototypeFeatureLifecycle(self.feature_lifecycle)
        lifecycle.require_compatible_oak_config(self.oak)
        base = self.feature_lifecycle.base_feature_dim
        total = self.feature_lifecycle.total_feature_dim
        if self.world_model.observation_dim != base:
            if self.world_model.observation_dim == total:
                raise ValueError(
                    "generated-feature output heads are unsupported and non-remappable; "
                    "world_model.observation_dim must equal the stable base width"
                )
            raise ValueError(
                "world_model.observation_dim must equal feature_lifecycle.base_feature_dim"
            )
        if self.world_model.n_actions != self.feature_lifecycle.n_primitive_actions:
            raise ValueError("world_model.n_actions must equal the primitive-action count")
        if self.world_model.hidden_sizes != ():
            raise ValueError("routed feature input requires an exact linear world model")
        if self.world_model.include_action_interactions:
            raise ValueError("action interactions are unsupported in the v1 routed lane")
        if not self.world_model.predict_delta:
            raise ValueError("fixed physical outputs must be stable-base deltas")
        if self.oak.stomp.option_planning_backups_per_step != 0:
            raise ValueError("the bounded routed lane requires OaK option planning disabled")
        _positive_int(
            self.anchor_capacity,
            name="anchor_capacity",
            maximum=_MAX_ANCHOR_CAPACITY,
        )
        if type(self.planning_enabled) is not bool:
            raise ValueError("planning_enabled must be an exact bool")
        _nonnegative_int(
            self.planning_warmup_steps,
            name="planning_warmup_steps",
            maximum=_INT32_MAX,
        )
        _positive_int(
            self.max_planned_backups,
            name="max_planned_backups",
            maximum=_INT32_MAX,
        )
        object.__setattr__(
            self,
            "max_generation_model_error",
            _finite_nonnegative_float32(
                self.max_generation_model_error,
                name="max_generation_model_error",
            ),
        )
        if type(self.carry_survivors) is not bool:
            raise ValueError("carry_survivors must be an exact bool")

    def _digest_payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA_VERSION,
            "state_schema": PROTOTYPE_ROUTED_LINEAR_WORLD_STATE_SCHEMA,
            "type": type(self).__name__,
            "mechanism_status": PROTOTYPE_ROUTED_LINEAR_WORLD_MECHANISM_STATUS,
            "evidence_level": PROTOTYPE_ROUTED_LINEAR_WORLD_EVIDENCE_LEVEL,
            "scientific_promotion_allowed": False,
            "feature_lifecycle": self.feature_lifecycle.to_config(),
            "world_model": self.world_model.to_config(),
            "oak": self.oak.to_config(),
            "anchor_capacity": self.anchor_capacity,
            "planning_enabled": self.planning_enabled,
            "planning_warmup_steps": self.planning_warmup_steps,
            "max_generation_model_error": self.max_generation_model_error,
            "max_planned_backups": self.max_planned_backups,
            "carry_survivors": self.carry_survivors,
            "fixed_output_semantics": _FIXED_OUTPUT_SEMANTICS,
            "generated_output_semantics": _GENERATED_OUTPUT_SEMANTICS,
            "route_semantics": _ROUTE_SEMANTICS,
            "planning_semantics": _PLANNING_SEMANTICS,
        }

    @property
    def schema_digest(self) -> bytes:
        encoded = json.dumps(
            self._digest_payload(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).digest()

    @property
    def schema_digest_hex(self) -> str:
        return self.schema_digest.hex()

    def to_config(self) -> dict[str, object]:
        return {
            **self._digest_payload(),
            "schema_digest_sha256": self.schema_digest_hex,
        }

    @classmethod
    def from_config(cls, payload: Mapping[str, object]) -> PrototypeRoutedLinearWorldConfig:
        """Restore only the exact current schema."""

        if not isinstance(payload, Mapping):
            raise TypeError("routed linear world config must be a mapping")
        raw = dict(payload)
        expected = {
            "schema",
            "state_schema",
            "type",
            "mechanism_status",
            "evidence_level",
            "scientific_promotion_allowed",
            "feature_lifecycle",
            "world_model",
            "oak",
            "anchor_capacity",
            "planning_enabled",
            "planning_warmup_steps",
            "max_generation_model_error",
            "max_planned_backups",
            "carry_survivors",
            "fixed_output_semantics",
            "generated_output_semantics",
            "route_semantics",
            "planning_semantics",
            "schema_digest_sha256",
        }
        if set(raw) != expected:
            raise ValueError("routed linear world config fields differ from v1")
        fixed = {
            "schema": cls.SCHEMA_VERSION,
            "state_schema": PROTOTYPE_ROUTED_LINEAR_WORLD_STATE_SCHEMA,
            "type": cls.__name__,
            "mechanism_status": PROTOTYPE_ROUTED_LINEAR_WORLD_MECHANISM_STATUS,
            "evidence_level": PROTOTYPE_ROUTED_LINEAR_WORLD_EVIDENCE_LEVEL,
            "scientific_promotion_allowed": False,
            "fixed_output_semantics": _FIXED_OUTPUT_SEMANTICS,
            "generated_output_semantics": _GENERATED_OUTPUT_SEMANTICS,
            "route_semantics": _ROUTE_SEMANTICS,
            "planning_semantics": _PLANNING_SEMANTICS,
        }
        if any(raw.get(name) != value for name, value in fixed.items()):
            raise ValueError("routed linear world fixed semantics differ")
        feature_raw = raw["feature_lifecycle"]
        world_raw = raw["world_model"]
        oak_raw = raw["oak"]
        if not isinstance(feature_raw, Mapping):
            raise ValueError("feature_lifecycle must be a config mapping")
        if not isinstance(world_raw, dict) or not isinstance(oak_raw, dict):
            raise ValueError("world_model and oak must be exact config dictionaries")
        result = cls(
            feature_lifecycle=PrototypeFeatureLifecycleConfig.from_config(feature_raw),
            world_model=ActionConditionedWorldModelConfig.from_config(world_raw),
            oak=OaKConfig.from_config(oak_raw),
            anchor_capacity=cast(int, raw["anchor_capacity"]),
            planning_enabled=cast(bool, raw["planning_enabled"]),
            planning_warmup_steps=cast(int, raw["planning_warmup_steps"]),
            max_generation_model_error=cast(float, raw["max_generation_model_error"]),
            max_planned_backups=cast(int, raw["max_planned_backups"]),
            carry_survivors=cast(bool, raw["carry_survivors"]),
        )
        if raw["schema_digest_sha256"] != result.schema_digest_hex:
            raise ValueError("routed linear world schema digest differs")
        return result


@chex.dataclass(frozen=True)
class PrototypeFixedPhysicalWorldCoreState:
    """Linear model with full feature input and fixed physical output heads."""

    learner_state: MultiHeadMLPState
    observation_min: Float[Array, " base_feature_dim"]
    observation_max: Float[Array, " base_feature_dim"]
    reward_min: Float[Array, ""]
    reward_max: Float[Array, ""]
    model_error_ema: Float[Array, ""]
    step_count: Int[Array, ""]
    step_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class PrototypeRoutedLinearWorldState:
    """World model, physical anchor cache, live bank, and exact guard clocks."""

    model_state: PrototypeFixedPhysicalWorldCoreState
    buffer_state: RecentObservationBufferState
    consumer_binding: PrototypeFeatureConsumerBinding
    generation_update_count: Int[Array, ""]
    generation_update_words: UInt[Array, " 2"]
    generation_birth_model_step_words: UInt[Array, " 2"]
    generation_error_ema: Float[Array, ""]
    generation_error_valid: Bool[Array, ""]
    planned_backup_count: Int[Array, ""]
    planned_backup_words: UInt[Array, " 2"]
    schema_digest: UInt[Array, " 32"]


@chex.dataclass(frozen=True)
class PrototypeFixedPhysicalWorldPrediction:
    """Decoded fixed-output prediction."""

    next_base_observation: Float[Array, " base_feature_dim"]
    reward: Float[Array, ""]
    discount: Float[Array, ""]
    raw_predictions: Float[Array, " physical_heads"]


@chex.dataclass(frozen=True)
class PrototypeRoutedLinearWorldPreparedTransition:
    """Source-owned pre-outcome model/bank/cache/action/prediction snapshot."""

    source_state: PrototypeRoutedLinearWorldState
    source_router_state: FeatureBankRouterState
    base_observation: Float[Array, " base_feature_dim"]
    cached_augmented_observation: Float[Array, " total_feature_dim"]
    input_features: Float[Array, " model_input_dim"]
    primitive_action: Int[Array, ""]
    prediction: PrototypeFixedPhysicalWorldPrediction
    prepared: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeRoutedLinearWorldPrepareDiagnostics:
    """Audit for a source-owned pre-outcome prediction cache."""

    source_state_valid: Bool[Array, ""]
    source_router_valid: Bool[Array, ""]
    source_router_matches_binding: Bool[Array, ""]
    base_observation_valid: Bool[Array, ""]
    primitive_action_valid: Bool[Array, ""]
    prediction_values_finite: Bool[Array, ""]
    prepared: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeRoutedLinearWorldPrepareResult:
    """One immutable pre-outcome cache plus its audit."""

    prepared: PrototypeRoutedLinearWorldPreparedTransition
    diagnostics: PrototypeRoutedLinearWorldPrepareDiagnostics


@chex.dataclass(frozen=True)
class PrototypeRoutedLinearWorldTransition:
    """Prepared source transition plus outcome and destination receipt."""

    prepared: PrototypeRoutedLinearWorldPreparedTransition
    reward: Float[Array, ""]
    discount: Float[Array, ""]
    next_base_observation: Float[Array, " base_feature_dim"]
    destination_router_state: FeatureBankRouterState
    destination_binding: PrototypeFeatureConsumerBinding


@chex.dataclass(frozen=True)
class PrototypeRoutedLinearWorldDiagnostics:
    """Audit of one real update and optional input-column migration."""

    source_state_valid: Bool[Array, ""]
    source_router_valid: Bool[Array, ""]
    source_router_matches_binding: Bool[Array, ""]
    destination_binding_valid: Bool[Array, ""]
    destination_router_valid: Bool[Array, ""]
    destination_router_matches_binding: Bool[Array, ""]
    descriptors_changed: Bool[Array, ""]
    generation_changed: Bool[Array, ""]
    generation_is_successor: Bool[Array, ""]
    bank_transition_consistent: Bool[Array, ""]
    prepared_source_state_matches: Bool[Array, ""]
    prepared_source_router_matches: Bool[Array, ""]
    source_cache_matches: Bool[Array, ""]
    source_prediction_matches: Bool[Array, ""]
    event_values_valid: Bool[Array, ""]
    source_clock_receipt_valid: Bool[Array, ""]
    learner_update_applied: Bool[Array, ""]
    generation_counter_capacity_available: Bool[Array, ""]
    route_attempted: Bool[Array, ""]
    route_candidate_valid: Bool[Array, ""]
    route_state_matches_destination: Bool[Array, ""]
    candidate_values_finite: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]
    physical_output_head_count: Int[Array, ""]
    generated_output_head_count: Int[Array, ""]
    cache_entries_after: Int[Array, ""]
    model_step_words_before: UInt[Array, " 2"]
    model_step_words_after: UInt[Array, " 2"]
    generation_update_words_after: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class PrototypeRoutedLinearWorldResult:
    """Atomically selected state plus predict-before-update evidence."""

    state: PrototypeRoutedLinearWorldState
    prediction: PrototypeFixedPhysicalWorldPrediction
    targets: Float[Array, " physical_heads"]
    prediction_error: Float[Array, ""]
    route_diagnostics: FeatureBankRouteDiagnostics
    diagnostics: PrototypeRoutedLinearWorldDiagnostics


@chex.dataclass(frozen=True)
class PrototypeRoutedLinearWorldPreparedAdoption:
    """One source-bank successor and routed candidate from one real update."""

    source_state: PrototypeRoutedLinearWorldState
    source_router_state: FeatureBankRouterState
    transition: PrototypeRoutedLinearWorldTransition
    ordinary_result: PrototypeRoutedLinearWorldResult
    destination_result: PrototypeRoutedLinearWorldResult
    ordinary_valid: Bool[Array, ""]
    destination_valid: Bool[Array, ""]
    preparation_learner_update_evaluations: Int[Array, ""]
    preparation_router_evaluations: Int[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeRoutedLinearWorldExternalReadinessReceipt:
    """Exact content-bound external verdict over a prepared world adoption."""

    prepared_adoption: PrototypeRoutedLinearWorldPreparedAdoption
    all_consumers_ready: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeRoutedLinearWorldAdoptionDiagnostics:
    """Source, receipt, external-veto, and exact-work adoption facts."""

    source_state_matches: Bool[Array, ""]
    source_router_state_matches: Bool[Array, ""]
    receipt_matches_preparation: Bool[Array, ""]
    ordinary_successor_valid: Bool[Array, ""]
    destination_candidate_valid: Bool[Array, ""]
    preparation_internally_valid: Bool[Array, ""]
    all_consumers_ready: Bool[Array, ""]
    destination_adopted: Bool[Array, ""]
    ordinary_update_retained: Bool[Array, ""]
    external_route_rolled_back: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]
    rejected: Bool[Array, ""]
    preparation_learner_update_evaluations: Int[Array, ""]
    adoption_learner_update_evaluations: Int[Array, ""]
    total_learner_update_evaluations: Int[Array, ""]
    preparation_router_evaluations: Int[Array, ""]
    adoption_router_evaluations: Int[Array, ""]
    total_router_evaluations: Int[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeRoutedLinearWorldAdoptionResult:
    """Legacy-shaped result plus external readiness adoption audit."""

    result: PrototypeRoutedLinearWorldResult
    diagnostics: PrototypeRoutedLinearWorldAdoptionDiagnostics


@chex.dataclass(frozen=True)
class PrototypeRoutedLinearWorldPlanRequest:
    """Inputs to the source-owned pre-plan snapshot boundary."""

    anchor_index: Int[Array, ""]
    primitive_action: Int[Array, ""]
    consumer_binding: PrototypeFeatureConsumerBinding
    router_state: FeatureBankRouterState
    expected_model_step_words: UInt[Array, " 2"]
    expected_planned_backup_words: UInt[Array, " 2"]
    expected_oak_step_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class PrototypeRoutedLinearWorldPreparedPlan:
    """Source-owned full world/OaK snapshot and deterministic plan cache."""

    source_state: PrototypeRoutedLinearWorldState
    source_oak_state: OaKState
    source_router_state: FeatureBankRouterState
    request: PrototypeRoutedLinearWorldPlanRequest
    anchor_base_observation: Float[Array, " base_feature_dim"]
    anchor_augmented_observation: Float[Array, " total_feature_dim"]
    predicted_next_augmented_observation: Float[Array, " total_feature_dim"]
    prediction: PrototypeFixedPhysicalWorldPrediction
    prepared: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeRoutedLinearWorldPlanPrepareDiagnostics:
    """Audit for a full source-world/source-OaK planning snapshot."""

    source_state_valid: Bool[Array, ""]
    source_oak_state_valid: Bool[Array, ""]
    router_valid: Bool[Array, ""]
    router_matches_binding: Bool[Array, ""]
    request_router_matches_source: Bool[Array, ""]
    consumer_binding_valid: Bool[Array, ""]
    consumer_binding_matches: Bool[Array, ""]
    clock_receipts_valid: Bool[Array, ""]
    anchor_available: Bool[Array, ""]
    action_valid: Bool[Array, ""]
    prediction_values_finite: Bool[Array, ""]
    prepared: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeRoutedLinearWorldPlanPrepareResult:
    """One immutable pre-plan cache plus its audit."""

    prepared: PrototypeRoutedLinearWorldPreparedPlan
    diagnostics: PrototypeRoutedLinearWorldPlanPrepareDiagnostics


@chex.dataclass(frozen=True)
class PrototypeRoutedLinearWorldPlanDiagnostics:
    """Audit of one bounded model-to-OaK base backup."""

    source_state_valid: Bool[Array, ""]
    source_oak_state_valid: Bool[Array, ""]
    prepared_source_state_matches: Bool[Array, ""]
    prepared_source_oak_state_matches: Bool[Array, ""]
    prepared_source_router_matches: Bool[Array, ""]
    prepared_cache_matches: Bool[Array, ""]
    router_valid: Bool[Array, ""]
    router_matches_binding: Bool[Array, ""]
    consumer_binding_valid: Bool[Array, ""]
    consumer_binding_matches: Bool[Array, ""]
    clock_receipts_valid: Bool[Array, ""]
    anchor_available: Bool[Array, ""]
    action_valid: Bool[Array, ""]
    planning_enabled: Bool[Array, ""]
    generation_warm: Bool[Array, ""]
    generation_error_ready: Bool[Array, ""]
    generation_error_below_limit: Bool[Array, ""]
    planning_capacity_available: Bool[Array, ""]
    prediction_values_finite: Bool[Array, ""]
    oak_update_applied: Bool[Array, ""]
    candidate_base_learner_finite: Bool[Array, ""]
    candidate_oak_state_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    base_learner_changed: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]
    pair_products_evaluated: Int[Array, ""]
    planned_backup_words_before: UInt[Array, " 2"]
    planned_backup_words_after: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class PrototypeRoutedLinearWorldPlanResult:
    """Read-only model plus atomically selected one-backup OaK state."""

    state: PrototypeRoutedLinearWorldState
    oak_state: OaKState
    anchor_base_observation: Float[Array, " base_feature_dim"]
    anchor_augmented_observation: Float[Array, " total_feature_dim"]
    predicted_next_augmented_observation: Float[Array, " total_feature_dim"]
    prediction: PrototypeFixedPhysicalWorldPrediction
    td_error: Float[Array, ""]
    diagnostics: PrototypeRoutedLinearWorldPlanDiagnostics


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeRoutedLinearWorldResourceBudget:
    """Exact bytes and history-independent two-phase transaction work maxima."""

    mechanism_status: str
    scientific_promotion_allowed: bool
    evidence_level: str
    base_feature_dim: int
    active_pair_slots: int
    total_feature_dim: int
    n_primitive_actions: int
    physical_output_heads: int
    generated_output_heads: int
    model_input_dim: int
    anchor_capacity: int
    planning_enabled: bool
    learner_state_nbytes: int
    model_core_state_nbytes: int
    buffer_state_nbytes: int
    consumer_binding_nbytes: int
    schema_digest_nbytes: int
    wrapper_metadata_nbytes: int
    persistent_state_nbytes: int
    source_oak_state_nbytes: int
    prepared_transition_cache_nbytes: int
    prepared_plan_cache_nbytes: int
    incremental_dynamic_input_nbytes: int
    routed_input_feature_groups: int
    routed_input_scalars: int
    routed_dynamic_input_scalars: int
    preserved_action_input_scalars: int
    max_router_calls_per_real_transition: int
    max_pair_products_per_transition_prepare: int
    max_pair_products_per_transition_consume: int
    max_pair_products_per_real_transition: int
    max_pair_products_per_plan_prepare: int
    max_pair_products_per_plan_consume: int
    max_pair_products_per_planning_call: int
    max_pair_products_per_planning_transaction: int
    max_world_forwards_per_transition_prepare: int
    max_world_forwards_per_transition_consume: int
    max_world_forwards_per_real_transition: int
    max_world_backwards_per_real_transition: int
    max_world_forwards_per_plan_prepare: int
    max_world_forwards_per_plan_consume: int
    max_world_forwards_per_planning_call: int
    max_oak_updates_per_planning_call: int
    max_oak_base_backups_per_planning_call: int
    persistent_capacity_growth: int

    def to_config(self) -> dict[str, str | int | bool]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeRoutedLinearWorldExternalTransactionResourceBudget:
    """Serialized logical PyTree bytes and fixed prepare/adopt work.

    The receipt embeds its preparation, so shared leaf references are counted
    once per serialized occurrence.  These are not physical allocator peaks.
    """

    persistent_state_nbytes_before: int
    persistent_state_nbytes_after: int
    source_router_state_nbytes: int
    prepared_adoption_logical_nbytes: int
    readiness_receipt_logical_nbytes: int
    simultaneous_logical_transient_nbytes: int
    learner_update_evaluations_per_prepare: int
    learner_update_evaluations_per_adopt: int
    learner_update_evaluations_per_transaction: int
    router_evaluations_per_prepare: int
    router_evaluations_per_adopt: int
    router_evaluations_per_transaction: int
    persistent_capacity_growth: int

    def to_config(self) -> dict[str, int]:
        return dataclasses.asdict(self)


class PrototypeRoutedLinearWorldModel:
    """Fixed-output linear world learner with authenticated feature routing."""

    def __init__(self, config: PrototypeRoutedLinearWorldConfig):
        if type(config) is not PrototypeRoutedLinearWorldConfig:
            raise TypeError("config must be an exact PrototypeRoutedLinearWorldConfig")
        self._config = config
        self._feature = config.feature_lifecycle
        self._world = config.world_model
        self._base_dim = self._feature.base_feature_dim
        self._active_slots = self._feature.active_pair_slots
        self._total_dim = self._feature.total_feature_dim
        self._n_actions = self._feature.n_primitive_actions
        self._n_heads = self._base_dim + 2
        self._input_dim = self._total_dim + self._n_actions
        self._router = FeatureBankRouter(
            FeatureBankRouterConfig(
                base_dim=self._base_dim,
                active_slots=self._active_slots,
            )
        )
        self._learner = MultiHeadMLPLearner(
            n_heads=self._n_heads,
            hidden_sizes=(),
            step_size=self._world.step_size,
            gamma=0.0,
            lamda=0.0,
            sparsity=self._world.sparsity,
            leaky_relu_slope=self._world.leaky_relu_slope,
            use_layer_norm=self._world.use_layer_norm,
            trace_mode=self._world.trace_mode,
            utility_decay=self._world.utility_decay,
        )
        self._buffer = RecentObservationBuffer(
            capacity=config.anchor_capacity,
            observation_dim=self._base_dim,
        )
        self._oak = OaKAgent(config.oak)
        oak_template = self._oak.init(jr.key(1))
        oak_base_template = oak_template.stomp_state.base_learner_state.replace(
            birth_timestamp=jnp.asarray(0.0, dtype=jnp.float32),
            uptime_s=jnp.asarray(0.0, dtype=jnp.float32),
        )
        self._oak_template = cast(
            OaKState,
            oak_template.replace(
                stomp_state=oak_template.stomp_state.replace(
                    base_learner_state=oak_base_template,
                )
            ),
        )
        self._schema_digest = jnp.asarray(
            tuple(config.schema_digest),
            dtype=jnp.uint8,
        )
        learner_template = self._learner.init(self._input_dim, jr.key(0))
        self._learner_template = cast(
            MultiHeadMLPState,
            learner_template.replace(
                birth_timestamp=jnp.asarray(0.0, dtype=jnp.float32),
                uptime_s=jnp.asarray(0.0, dtype=jnp.float32),
            ),
        )

    @property
    def config(self) -> PrototypeRoutedLinearWorldConfig:
        return self._config

    @property
    def learner(self) -> MultiHeadMLPLearner:
        """Expose the exact linear learner for direct parity witnesses."""

        return self._learner

    @property
    def oak(self) -> OaKAgent:
        """Expose the exact bounded planning consumer."""

        return self._oak

    @property
    def router(self) -> FeatureBankRouter:
        return self._router

    @property
    def schema_digest(self) -> UInt[Array, " 32"]:
        return self._schema_digest

    def to_config(self) -> dict[str, object]:
        return self._config.to_config()

    @classmethod
    def from_config(cls, payload: Mapping[str, object]) -> PrototypeRoutedLinearWorldModel:
        return cls(PrototypeRoutedLinearWorldConfig.from_config(payload))

    def _validate_binding_static_contract(
        self,
        binding: PrototypeFeatureConsumerBinding,
        *,
        name: str,
    ) -> None:
        if type(binding) is not PrototypeFeatureConsumerBinding:
            raise TypeError(f"{name} must be an exact PrototypeFeatureConsumerBinding")
        _require_array(
            binding.semantic_generation,
            name=f"{name}.semantic_generation",
            shape=(),
            dtype=jnp.int32,
        )
        _require_array(
            binding.semantic_generation_words,
            name=f"{name}.semantic_generation_words",
            shape=(2,),
            dtype=jnp.uint32,
        )
        _require_array(
            binding.descriptors,
            name=f"{name}.descriptors",
            shape=(self._active_slots, 2),
            dtype=jnp.int32,
        )

    def _validate_router_static_contract(
        self,
        state: FeatureBankRouterState,
        *,
        name: str,
    ) -> None:
        if type(state) is not FeatureBankRouterState:
            raise TypeError(f"{name} must be an exact FeatureBankRouterState")
        _require_array(
            state.descriptors,
            name=f"{name}.descriptors",
            shape=(self._active_slots, 2),
            dtype=jnp.int32,
        )
        for field_name in ("route_count", "generation_count"):
            _require_array(
                getattr(state, field_name),
                name=f"{name}.{field_name}",
                shape=(),
                dtype=jnp.int32,
            )
        for field_name in ("route_words", "generation_words"):
            _require_array(
                getattr(state, field_name),
                name=f"{name}.{field_name}",
                shape=(2,),
                dtype=jnp.uint32,
            )

    def _validate_state_static_contract(self, state: PrototypeRoutedLinearWorldState) -> None:
        if type(state) is not PrototypeRoutedLinearWorldState:
            raise TypeError("state must be an exact PrototypeRoutedLinearWorldState")
        if type(state.model_state) is not PrototypeFixedPhysicalWorldCoreState:
            raise TypeError("state.model_state has the wrong exact type")
        core = state.model_state
        if not _tree_static_contract_matches(core.learner_state, self._learner_template):
            raise ValueError("state learner static contract differs")
        _require_array(
            core.observation_min,
            name="state.model_state.observation_min",
            shape=(self._base_dim,),
            dtype=jnp.float32,
        )
        _require_array(
            core.observation_max,
            name="state.model_state.observation_max",
            shape=(self._base_dim,),
            dtype=jnp.float32,
        )
        for field_name in ("reward_min", "reward_max", "model_error_ema"):
            _require_array(
                getattr(core, field_name),
                name=f"state.model_state.{field_name}",
                shape=(),
                dtype=jnp.float32,
            )
        _require_array(
            core.step_count,
            name="state.model_state.step_count",
            shape=(),
            dtype=jnp.int32,
        )
        _require_array(
            core.step_words,
            name="state.model_state.step_words",
            shape=(2,),
            dtype=jnp.uint32,
        )
        if type(state.buffer_state) is not RecentObservationBufferState:
            raise TypeError("state.buffer_state has the wrong exact type")
        _require_array(
            state.buffer_state.observations,
            name="state.buffer_state.observations",
            shape=(self._config.anchor_capacity, self._base_dim),
            dtype=jnp.float32,
        )
        for field_name in ("size", "index"):
            _require_array(
                getattr(state.buffer_state, field_name),
                name=f"state.buffer_state.{field_name}",
                shape=(),
                dtype=jnp.int32,
            )
        self._validate_binding_static_contract(
            state.consumer_binding,
            name="state.consumer_binding",
        )
        for field_name in ("generation_update_count", "planned_backup_count"):
            _require_array(
                getattr(state, field_name),
                name=f"state.{field_name}",
                shape=(),
                dtype=jnp.int32,
            )
        for field_name in (
            "generation_update_words",
            "generation_birth_model_step_words",
            "planned_backup_words",
        ):
            _require_array(
                getattr(state, field_name),
                name=f"state.{field_name}",
                shape=(2,),
                dtype=jnp.uint32,
            )
        _require_array(
            state.generation_error_ema,
            name="state.generation_error_ema",
            shape=(),
            dtype=jnp.float32,
        )
        _require_array(
            state.generation_error_valid,
            name="state.generation_error_valid",
            shape=(),
            dtype=jnp.bool_,
        )
        _require_array(
            state.schema_digest,
            name="state.schema_digest",
            shape=(_SCHEMA_DIGEST_NBYTES,),
            dtype=jnp.uint8,
        )

    def _validate_oak_static_contract(self, state: OaKState, *, name: str) -> None:
        if type(state) is not OaKState:
            raise TypeError(f"{name} must be an exact OaKState")
        if not _tree_static_contract_matches(state, self._oak_template):
            raise ValueError(f"{name} static contract differs")

    def _validate_transition_static_contract(
        self,
        event: PrototypeRoutedLinearWorldTransition,
    ) -> None:
        if type(event) is not PrototypeRoutedLinearWorldTransition:
            raise TypeError("event must be an exact PrototypeRoutedLinearWorldTransition")
        self._validate_prepared_transition_static_contract(event.prepared)
        for field_name in ("reward", "discount"):
            _require_array(
                getattr(event, field_name),
                name=f"event.{field_name}",
                shape=(),
                dtype=jnp.float32,
            )
        _require_array(
            event.next_base_observation,
            name="event.next_base_observation",
            shape=(self._base_dim,),
            dtype=jnp.float32,
        )
        self._validate_router_static_contract(
            event.destination_router_state,
            name="event.destination_router_state",
        )
        self._validate_binding_static_contract(
            event.destination_binding,
            name="event.destination_binding",
        )

    def _validate_prepared_transition_static_contract(
        self,
        prepared: PrototypeRoutedLinearWorldPreparedTransition,
    ) -> None:
        if type(prepared) is not PrototypeRoutedLinearWorldPreparedTransition:
            raise TypeError(
                "prepared must be an exact PrototypeRoutedLinearWorldPreparedTransition"
            )
        self._validate_state_static_contract(prepared.source_state)
        self._validate_router_static_contract(
            prepared.source_router_state,
            name="prepared.source_router_state",
        )
        _require_array(
            prepared.base_observation,
            name="prepared.base_observation",
            shape=(self._base_dim,),
            dtype=jnp.float32,
        )
        _require_array(
            prepared.cached_augmented_observation,
            name="prepared.cached_augmented_observation",
            shape=(self._total_dim,),
            dtype=jnp.float32,
        )
        _require_array(
            prepared.input_features,
            name="prepared.input_features",
            shape=(self._input_dim,),
            dtype=jnp.float32,
        )
        _require_array(
            prepared.primitive_action,
            name="prepared.primitive_action",
            shape=(),
            dtype=jnp.int32,
        )
        self._validate_prediction_static_contract(
            prepared.prediction,
            name="prepared.prediction",
        )
        _require_array(
            prepared.prepared,
            name="prepared.prepared",
            shape=(),
            dtype=jnp.bool_,
        )

    def _validate_prediction_static_contract(
        self,
        prediction: PrototypeFixedPhysicalWorldPrediction,
        *,
        name: str,
    ) -> None:
        if type(prediction) is not PrototypeFixedPhysicalWorldPrediction:
            raise TypeError(f"{name} has the wrong exact type")
        _require_array(
            prediction.next_base_observation,
            name=f"{name}.next_base_observation",
            shape=(self._base_dim,),
            dtype=jnp.float32,
        )
        for field_name in ("reward", "discount"):
            _require_array(
                getattr(prediction, field_name),
                name=f"{name}.{field_name}",
                shape=(),
                dtype=jnp.float32,
            )
        _require_array(
            prediction.raw_predictions,
            name=f"{name}.raw_predictions",
            shape=(self._n_heads,),
            dtype=jnp.float32,
        )

    def _validate_plan_static_contract(
        self,
        request: PrototypeRoutedLinearWorldPlanRequest,
    ) -> None:
        if type(request) is not PrototypeRoutedLinearWorldPlanRequest:
            raise TypeError("request must be an exact PrototypeRoutedLinearWorldPlanRequest")
        for field_name in ("anchor_index", "primitive_action"):
            _require_array(
                getattr(request, field_name),
                name=f"request.{field_name}",
                shape=(),
                dtype=jnp.int32,
            )
        self._validate_binding_static_contract(
            request.consumer_binding,
            name="request.consumer_binding",
        )
        self._validate_router_static_contract(
            request.router_state,
            name="request.router_state",
        )
        for field_name in (
            "expected_model_step_words",
            "expected_planned_backup_words",
            "expected_oak_step_words",
        ):
            _require_array(
                getattr(request, field_name),
                name=f"request.{field_name}",
                shape=(2,),
                dtype=jnp.uint32,
            )

    def _validate_prepared_plan_static_contract(
        self,
        prepared: PrototypeRoutedLinearWorldPreparedPlan,
    ) -> None:
        if type(prepared) is not PrototypeRoutedLinearWorldPreparedPlan:
            raise TypeError("prepared must be an exact PrototypeRoutedLinearWorldPreparedPlan")
        self._validate_state_static_contract(prepared.source_state)
        self._validate_oak_static_contract(
            prepared.source_oak_state,
            name="prepared.source_oak_state",
        )
        self._validate_router_static_contract(
            prepared.source_router_state,
            name="prepared.source_router_state",
        )
        self._validate_plan_static_contract(prepared.request)
        for field_name, shape in (
            ("anchor_base_observation", (self._base_dim,)),
            ("anchor_augmented_observation", (self._total_dim,)),
            ("predicted_next_augmented_observation", (self._total_dim,)),
        ):
            _require_array(
                getattr(prepared, field_name),
                name=f"prepared.{field_name}",
                shape=shape,
                dtype=jnp.float32,
            )
        self._validate_prediction_static_contract(
            prepared.prediction,
            name="prepared.prediction",
        )
        _require_array(
            prepared.prepared,
            name="prepared.prepared",
            shape=(),
            dtype=jnp.bool_,
        )

    def _binding_valid(self, binding: PrototypeFeatureConsumerBinding) -> Bool[Array, ""]:
        validation = self._router.validate_descriptors(binding.descriptors)
        return cast(
            Bool[Array, ""],
            (binding.semantic_generation >= jnp.int32(0))
            & _lifetime_counter_valid(
                binding.semantic_generation_words,
                binding.semantic_generation,
            )
            & validation.valid
            & jnp.all(validation.live_mask),
        )

    @staticmethod
    def _bindings_equal(
        left: PrototypeFeatureConsumerBinding,
        right: PrototypeFeatureConsumerBinding,
    ) -> Bool[Array, ""]:
        return (
            (left.semantic_generation == right.semantic_generation)
            & jnp.all(left.semantic_generation_words == right.semantic_generation_words)
            & jnp.all(left.descriptors == right.descriptors)
        )

    def _router_valid(self, state: FeatureBankRouterState) -> Bool[Array, ""]:
        validation = self._router.validate_descriptors(state.descriptors)
        return (
            validation.valid
            & jnp.all(validation.live_mask)
            & _lifetime_counter_valid(state.route_words, state.route_count)
            & _lifetime_counter_valid(
                state.generation_words,
                state.generation_count,
            )
            & _words_less_equal(state.generation_words, state.route_words)
        )

    def _router_matches_binding(
        self,
        router_state: FeatureBankRouterState,
        binding: PrototypeFeatureConsumerBinding,
    ) -> Bool[Array, ""]:
        return (
            (router_state.generation_count == binding.semantic_generation)
            & jnp.all(router_state.generation_words == binding.semantic_generation_words)
            & jnp.all(router_state.descriptors == binding.descriptors)
        )

    def _learner_optimizer_matches_config(
        self,
        state: MultiHeadMLPState,
    ) -> Bool[Array, ""]:
        static_valid = (
            type(state.trunk_params.weights) is tuple
            and type(state.trunk_params.biases) is tuple
            and type(state.trunk_optimizer_states) is tuple
            and type(state.trunk_traces) is tuple
            and type(state.hidden_unit_utilities) is tuple
            and type(state.head_params.weights) is tuple
            and type(state.head_params.biases) is tuple
            and type(state.head_optimizer_states) is tuple
            and type(state.head_traces) is tuple
            and state.normalizer_state is None
            and len(state.trunk_params.weights) == 0
            and len(state.trunk_params.biases) == 0
            and len(state.trunk_optimizer_states) == 0
            and len(state.trunk_traces) == 0
            and len(state.hidden_unit_utilities) == 0
            and len(state.head_params.weights) == self._n_heads
            and len(state.head_params.biases) == self._n_heads
            and len(state.head_optimizer_states) == self._n_heads
            and len(state.head_traces) == self._n_heads
        )
        if not static_valid:
            return jnp.asarray(False, dtype=jnp.bool_)
        step_sizes: list[Array] = []
        for optimizer_pair in state.head_optimizer_states:
            if (
                type(optimizer_pair) is not tuple
                or len(optimizer_pair) != 2
                or type(optimizer_pair[0]) is not LMSState
                or type(optimizer_pair[1]) is not LMSState
            ):
                return jnp.asarray(False, dtype=jnp.bool_)
            step_sizes.extend((optimizer_pair[0].step_size, optimizer_pair[1].step_size))
        expected = jnp.asarray(self._world.step_size, dtype=jnp.float32)
        values = jnp.stack(step_sizes)
        return jnp.all(
            jax.lax.bitcast_convert_type(values, jnp.uint32)
            == jax.lax.bitcast_convert_type(expected, jnp.uint32)
        )

    def _buffer_valid(
        self,
        state: RecentObservationBufferState,
        model_step_words: Array,
    ) -> Bool[Array, ""]:
        capacity = self._config.anchor_capacity
        reached_capacity = (model_step_words[0] != jnp.uint32(0)) | (
            model_step_words[1] >= jnp.asarray(capacity, dtype=jnp.uint32)
        )
        expected_size = jnp.where(
            reached_capacity,
            jnp.int32(capacity),
            model_step_words[1].astype(jnp.int32),
        )
        expected_index = _words_mod_small(model_step_words, capacity)
        row_indices = jnp.arange(capacity, dtype=jnp.int32)
        unused = row_indices >= state.size
        bits = jax.lax.bitcast_convert_type(state.observations, jnp.uint32)
        unused_positive_zero = jnp.all((~unused[:, None]) | (bits == jnp.uint32(0)))
        return (
            jnp.all(jnp.isfinite(state.observations))
            & (state.size == expected_size)
            & (state.index == expected_index)
            & unused_positive_zero
        )

    def _state_is_valid(self, state: PrototypeRoutedLinearWorldState) -> Bool[Array, ""]:
        core = state.model_state
        step_valid = _lifetime_counter_valid(core.step_words, core.step_count)
        learner_counter_valid = (
            _lifetime_counter_valid(
                core.learner_state.step_words,
                core.learner_state.step_count,
            )
            & (core.step_count == core.learner_state.step_count)
            & jnp.all(core.step_words == core.learner_state.step_words)
        )
        pristine_bounds = (
            jnp.all(jnp.isposinf(core.observation_min))
            & jnp.all(jnp.isneginf(core.observation_max))
            & jnp.isposinf(core.reward_min)
            & jnp.isneginf(core.reward_max)
        )
        finite_bounds = (
            jnp.all(jnp.isfinite(core.observation_min))
            & jnp.all(jnp.isfinite(core.observation_max))
            & jnp.all(core.observation_min <= core.observation_max)
            & jnp.isfinite(core.reward_min)
            & jnp.isfinite(core.reward_max)
            & (core.reward_min <= core.reward_max)
        )
        bounds_valid = jnp.where(core.step_count == 0, pristine_bounds, finite_bounds)
        generation_counter_valid = _lifetime_counter_valid(
            state.generation_update_words,
            state.generation_update_count,
        )
        reconstructed_step, generation_sum_capacity = _checked_words_add(
            state.generation_birth_model_step_words,
            state.generation_update_words,
        )
        generation_clock_valid = (
            generation_counter_valid
            & generation_sum_capacity
            & jnp.all(reconstructed_step == core.step_words)
            & _words_less_equal(
                state.generation_birth_model_step_words,
                core.step_words,
            )
        )
        generation_has_updates = jnp.any(state.generation_update_words != jnp.uint32(0))
        generation_error_valid = (
            jnp.isfinite(state.generation_error_ema)
            & (state.generation_error_ema >= 0.0)
            & (state.generation_error_valid == generation_has_updates)
            & jnp.where(
                state.generation_error_valid,
                jnp.asarray(True, dtype=jnp.bool_),
                jax.lax.bitcast_convert_type(
                    state.generation_error_ema,
                    jnp.uint32,
                )
                == jnp.uint32(0),
            )
        )
        planned_counter_valid = (
            _lifetime_counter_valid(
                state.planned_backup_words,
                state.planned_backup_count,
            )
            & (state.planned_backup_words[0] == jnp.uint32(0))
            & (
                state.planned_backup_words[1]
                <= jnp.asarray(self._config.max_planned_backups, dtype=jnp.uint32)
            )
        )
        return cast(
            Bool[Array, ""],
            _floating_tree_is_finite(core.learner_state)
            & self._learner_optimizer_matches_config(core.learner_state)
            & step_valid
            & learner_counter_valid
            & bounds_valid
            & jnp.isfinite(core.model_error_ema)
            & (core.model_error_ema >= 0.0)
            & self._binding_valid(state.consumer_binding)
            & generation_clock_valid
            & generation_error_valid
            & planned_counter_valid
            & self._buffer_valid(state.buffer_state, core.step_words)
            & jnp.all(state.schema_digest == self._schema_digest),
        )

    def _oak_state_is_valid(self, state: OaKState) -> Bool[Array, ""]:
        """Validate the complete caller-owned OaK source/candidate."""

        return (
            _oak_outer_state_validity(state, self._config.oak)[-1]
            & jnp.all(state.step_words == state.stomp_state.step_words)
            & self._oak.stomp_agent.state_valid(state.stomp_state)
        )

    def state_valid(self, state: PrototypeRoutedLinearWorldState) -> Bool[Array, ""]:
        """Validate structure, optimizer, bank, cache, bounds, clocks, and digest."""

        self._validate_state_static_contract(state)
        return cast(Bool[Array, ""], self._state_valid_jit(state))

    @functools.partial(jax.jit, static_argnums=(0,))
    def _state_valid_jit(self, state: PrototypeRoutedLinearWorldState) -> Array:
        return self._state_is_valid(state)

    def init(
        self,
        key: Array,
        binding: PrototypeFeatureConsumerBinding,
        router_state: FeatureBankRouterState,
    ) -> PrototypeRoutedLinearWorldState:
        """Initialize and authenticate one canonical live feature bank."""

        if not (
            hasattr(key, "shape")
            and hasattr(key, "dtype")
            and key.shape == ()
            and jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key)
        ):
            raise TypeError("key must be a scalar typed JAX PRNG key")
        self._validate_binding_static_contract(binding, name="binding")
        self._validate_router_static_contract(router_state, name="router_state")
        learner = self._learner.init(self._input_dim, key)
        learner = cast(
            MultiHeadMLPState,
            learner.replace(
                birth_timestamp=jnp.asarray(0.0, dtype=jnp.float32),
                uptime_s=jnp.asarray(0.0, dtype=jnp.float32),
            ),
        )
        zero = jnp.asarray(0, dtype=jnp.int32)
        zero_words = jnp.zeros((2,), dtype=jnp.uint32)
        state = PrototypeRoutedLinearWorldState(
            model_state=PrototypeFixedPhysicalWorldCoreState(
                learner_state=learner,
                observation_min=jnp.full(
                    (self._base_dim,),
                    jnp.inf,
                    dtype=jnp.float32,
                ),
                observation_max=jnp.full(
                    (self._base_dim,),
                    -jnp.inf,
                    dtype=jnp.float32,
                ),
                reward_min=jnp.asarray(jnp.inf, dtype=jnp.float32),
                reward_max=jnp.asarray(-jnp.inf, dtype=jnp.float32),
                model_error_ema=jnp.asarray(0.0, dtype=jnp.float32),
                step_count=zero,
                step_words=zero_words,
            ),
            buffer_state=self._buffer.init(),
            consumer_binding=binding,
            generation_update_count=zero,
            generation_update_words=zero_words,
            generation_birth_model_step_words=zero_words,
            generation_error_ema=jnp.asarray(0.0, dtype=jnp.float32),
            generation_error_valid=jnp.asarray(False, dtype=jnp.bool_),
            planned_backup_count=zero,
            planned_backup_words=zero_words,
            schema_digest=self._schema_digest,
        )
        valid = (
            self._router_valid(router_state)
            & self._router_matches_binding(router_state, binding)
            & self.state_valid(state)
        )
        if not bool(jax.device_get(valid)):
            raise ValueError("initial routed linear world composition is invalid")
        return state

    def _augment(
        self,
        binding: PrototypeFeatureConsumerBinding,
        base_observation: Array,
    ) -> Array:
        safe_left = jnp.clip(binding.descriptors[:, 0], 0, self._base_dim - 1)
        safe_right = jnp.clip(binding.descriptors[:, 1], 0, self._base_dim - 1)
        products = base_observation[safe_left] * base_observation[safe_right]
        return jnp.concatenate((base_observation, products), axis=0)

    @functools.partial(jax.jit, static_argnums=(0,))
    def augment(
        self,
        binding: PrototypeFeatureConsumerBinding,
        base_observation: Array,
    ) -> Float[Array, " total_feature_dim"]:
        """Return exact stable base plus live descriptor pair products."""

        base = jnp.asarray(base_observation, dtype=jnp.float32).reshape((self._base_dim,))
        return self._augment(binding, base)

    def _input_features(self, augmented_observation: Array, action: Array) -> Array:
        one_hot = jax.nn.one_hot(
            action.astype(jnp.int32),
            self._n_actions,
            dtype=jnp.float32,
        )
        return jnp.concatenate((augmented_observation, one_hot), axis=0)

    @functools.partial(jax.jit, static_argnums=(0,))
    def input_features(self, augmented_observation: Array, action: Array) -> Array:
        """Return ``[phi_g(x), one_hot(action)]``."""

        augmented = jnp.asarray(augmented_observation, dtype=jnp.float32).reshape(
            (self._total_dim,)
        )
        return self._input_features(augmented, action)

    def _targets(
        self,
        base_observation: Array,
        reward: Array,
        discount: Array,
        next_base_observation: Array,
    ) -> Array:
        scale = jnp.asarray(
            (
                (1.0,) * self._base_dim
                if self._world.observation_scale is None
                else self._world.observation_scale
            ),
            dtype=jnp.float32,
        )
        safe_scale = jnp.maximum(scale, jnp.asarray(1.0e-6, dtype=jnp.float32))
        delta = (next_base_observation - base_observation) / safe_scale
        return jnp.concatenate(
            (
                delta,
                jnp.reshape(reward / self._world.reward_scale, (1,)),
                jnp.reshape(discount, (1,)),
            ),
            axis=0,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def targets(
        self,
        base_observation: Array,
        reward: Array,
        discount: Array,
        next_base_observation: Array,
    ) -> Float[Array, " physical_heads"]:
        """Build only ``[delta stable-base, reward, discount]`` targets."""

        base = jnp.asarray(base_observation, dtype=jnp.float32).reshape((self._base_dim,))
        next_base = jnp.asarray(next_base_observation, dtype=jnp.float32).reshape((self._base_dim,))
        return self._targets(base, reward, discount, next_base)

    def _decode_prediction(
        self,
        state: PrototypeFixedPhysicalWorldCoreState,
        base_observation: Array,
        raw_predictions: Array,
    ) -> PrototypeFixedPhysicalWorldPrediction:
        scale = jnp.asarray(
            (
                (1.0,) * self._base_dim
                if self._world.observation_scale is None
                else self._world.observation_scale
            ),
            dtype=jnp.float32,
        )
        normalized_delta = jnp.clip(
            raw_predictions[: self._base_dim],
            -self._world.max_delta_scale,
            self._world.max_delta_scale,
        )
        next_base = base_observation + normalized_delta * scale
        low = state.observation_min - self._world.observation_clip_margin
        high = state.observation_max + self._world.observation_clip_margin
        next_base = jnp.where(
            state.step_count > 0,
            jnp.clip(next_base, low, high),
            next_base,
        )
        reward = raw_predictions[self._base_dim] * self._world.reward_scale
        reward = jnp.where(
            state.step_count > 0,
            jnp.clip(
                reward,
                state.reward_min - self._world.observation_clip_margin,
                state.reward_max + self._world.observation_clip_margin,
            ),
            reward,
        )
        discount = jnp.clip(
            raw_predictions[self._base_dim + 1],
            0.0,
            self._world.gamma,
        )
        return PrototypeFixedPhysicalWorldPrediction(
            next_base_observation=next_base,
            reward=reward,
            discount=discount,
            raw_predictions=raw_predictions,
        )

    def _predict(
        self,
        state: PrototypeFixedPhysicalWorldCoreState,
        base_observation: Array,
        augmented_observation: Array,
        action: Array,
    ) -> PrototypeFixedPhysicalWorldPrediction:
        raw = self._learner.predict(
            state.learner_state,
            self._input_features(augmented_observation, action),
        )
        return self._decode_prediction(state, base_observation, raw)

    def _route_learner_inputs(
        self,
        source_router_state: FeatureBankRouterState,
        destination_descriptors: Array,
        learner_state: MultiHeadMLPState,
    ) -> tuple[MultiHeadMLPState, FeatureBankRouterState, FeatureBankRouteDiagnostics]:
        weights = jnp.concatenate(learner_state.head_params.weights, axis=0)
        weight_traces = jnp.concatenate(
            tuple(trace_pair[0] for trace_pair in learner_state.head_traces),
            axis=0,
        )
        observation_weights = weights[:, : self._total_dim]
        action_weights = weights[:, self._total_dim :]
        observation_traces = weight_traces[:, : self._total_dim]
        action_traces = weight_traces[:, self._total_dim :]
        route = self._router.route(
            source_router_state,
            (observation_weights, observation_traces),
            destination_descriptors,
            carry_survivors=self._config.carry_survivors,
        )
        routed_observation_weights, routed_observation_traces = route.consumers
        routed_weights = jnp.concatenate(
            (routed_observation_weights, action_weights),
            axis=1,
        )
        routed_weight_traces = jnp.concatenate(
            (routed_observation_traces, action_traces),
            axis=1,
        )
        head_weights = tuple(routed_weights[index : index + 1] for index in range(self._n_heads))
        head_traces = tuple(
            (
                routed_weight_traces[index : index + 1],
                learner_state.head_traces[index][1],
            )
            for index in range(self._n_heads)
        )
        routed_state = cast(
            MultiHeadMLPState,
            learner_state.replace(
                head_params=MLPParams(
                    weights=head_weights,
                    biases=learner_state.head_params.biases,
                ),
                head_traces=head_traces,
            ),
        )
        return routed_state, route.state, route.diagnostics

    def prepare_transition(
        self,
        state: PrototypeRoutedLinearWorldState,
        router_state: FeatureBankRouterState,
        base_observation: Array,
        primitive_action: Array,
    ) -> PrototypeRoutedLinearWorldPrepareResult:
        """Mint a source-owned prediction cache before reward/successor input."""

        self._validate_state_static_contract(state)
        self._validate_router_static_contract(router_state, name="router_state")
        _require_array(
            base_observation,
            name="base_observation",
            shape=(self._base_dim,),
            dtype=jnp.float32,
        )
        _require_array(
            primitive_action,
            name="primitive_action",
            shape=(),
            dtype=jnp.int32,
        )
        return cast(
            PrototypeRoutedLinearWorldPrepareResult,
            self._prepare_transition_jit(
                state,
                router_state,
                base_observation,
                primitive_action,
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _prepare_transition_jit(
        self,
        state: PrototypeRoutedLinearWorldState,
        router_state: FeatureBankRouterState,
        base_observation: Array,
        primitive_action: Array,
    ) -> PrototypeRoutedLinearWorldPrepareResult:
        source_state_valid = self._state_is_valid(state)
        source_router_valid = self._router_valid(router_state)
        source_router_matches = self._router_matches_binding(
            router_state,
            state.consumer_binding,
        )
        base_valid = jnp.all(jnp.isfinite(base_observation))
        action_valid = (primitive_action >= 0) & (primitive_action < self._n_actions)
        augmented = self._augment(state.consumer_binding, base_observation)
        inputs = self._input_features(augmented, primitive_action)
        prediction = self._predict(
            state.model_state,
            base_observation,
            augmented,
            primitive_action,
        )
        prediction_finite = (
            jnp.all(jnp.isfinite(augmented))
            & jnp.all(jnp.isfinite(inputs))
            & jnp.all(jnp.isfinite(prediction.next_base_observation))
            & jnp.isfinite(prediction.reward)
            & jnp.isfinite(prediction.discount)
            & jnp.all(jnp.isfinite(prediction.raw_predictions))
        )
        prepared_valid = (
            source_state_valid
            & source_router_valid
            & source_router_matches
            & base_valid
            & action_valid
            & prediction_finite
        )
        prepared = PrototypeRoutedLinearWorldPreparedTransition(
            source_state=state,
            source_router_state=router_state,
            base_observation=base_observation,
            cached_augmented_observation=augmented,
            input_features=inputs,
            primitive_action=primitive_action,
            prediction=prediction,
            prepared=prepared_valid,
        )
        diagnostics = PrototypeRoutedLinearWorldPrepareDiagnostics(
            source_state_valid=source_state_valid,
            source_router_valid=source_router_valid,
            source_router_matches_binding=source_router_matches,
            base_observation_valid=base_valid,
            primitive_action_valid=action_valid,
            prediction_values_finite=prediction_finite,
            prepared=prepared_valid,
        )
        return PrototypeRoutedLinearWorldPrepareResult(
            prepared=prepared,
            diagnostics=diagnostics,
        )

    def observe_and_route(
        self,
        state: PrototypeRoutedLinearWorldState,
        source_router_state: FeatureBankRouterState,
        event: PrototypeRoutedLinearWorldTransition,
    ) -> PrototypeRoutedLinearWorldResult:
        """Predict/update under the source bank, then atomically route inputs."""

        prepared = self.prepare_observe_and_route(
            state,
            source_router_state,
            event,
        )
        return prepared.destination_result

    def prepare_observe_and_route(
        self,
        state: PrototypeRoutedLinearWorldState,
        source_router_state: FeatureBankRouterState,
        event: PrototypeRoutedLinearWorldTransition,
    ) -> PrototypeRoutedLinearWorldPreparedAdoption:
        """Consume one prepared outcome into ordinary and routed successors."""

        self._validate_state_static_contract(state)
        self._validate_router_static_contract(
            source_router_state,
            name="source_router_state",
        )
        self._validate_transition_static_contract(event)
        return cast(
            PrototypeRoutedLinearWorldPreparedAdoption,
            self._prepare_observe_and_route_jit(state, source_router_state, event),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _prepare_observe_and_route_jit(
        self,
        state: PrototypeRoutedLinearWorldState,
        source_router_state: FeatureBankRouterState,
        event: PrototypeRoutedLinearWorldTransition,
    ) -> PrototypeRoutedLinearWorldPreparedAdoption:
        prepared = event.prepared
        source_state_valid = self._state_is_valid(state)
        prepared_source_state_matches = _tree_exactly_equal(
            state,
            prepared.source_state,
        )
        prepared_source_router_matches = _tree_exactly_equal(
            source_router_state,
            prepared.source_router_state,
        )
        source_router_valid = self._router_valid(source_router_state)
        source_router_matches = self._router_matches_binding(
            source_router_state,
            state.consumer_binding,
        )
        destination_binding_valid = self._binding_valid(event.destination_binding)
        destination_router_valid = self._router_valid(event.destination_router_state)
        destination_router_matches = self._router_matches_binding(
            event.destination_router_state,
            event.destination_binding,
        )
        descriptors_changed = jnp.any(
            state.consumer_binding.descriptors != event.destination_binding.descriptors
        )
        generation_changed = jnp.any(
            state.consumer_binding.semantic_generation_words
            != event.destination_binding.semantic_generation_words
        )
        generation_is_successor = _words_successor(
            state.consumer_binding.semantic_generation_words,
            event.destination_binding.semantic_generation_words,
        )
        no_bank_change = (~descriptors_changed) & (~generation_changed)
        changed_consistently = descriptors_changed & generation_changed & generation_is_successor
        bank_transition_consistent = no_bank_change | changed_consistently

        expected_cache = self._augment(
            state.consumer_binding,
            prepared.base_observation,
        )
        expected_inputs = self._input_features(
            expected_cache,
            prepared.primitive_action,
        )
        expected_prediction = self._predict(
            state.model_state,
            prepared.base_observation,
            expected_cache,
            prepared.primitive_action,
        )
        source_cache_matches = _tree_exactly_equal(
            expected_cache,
            prepared.cached_augmented_observation,
        ) & _tree_exactly_equal(expected_inputs, prepared.input_features)
        source_prediction_matches = _tree_exactly_equal(
            expected_prediction,
            prepared.prediction,
        )
        event_values_valid = (
            jnp.all(jnp.isfinite(prepared.base_observation))
            & jnp.all(jnp.isfinite(prepared.cached_augmented_observation))
            & jnp.all(jnp.isfinite(prepared.input_features))
            & jnp.all(jnp.isfinite(prepared.prediction.next_base_observation))
            & jnp.isfinite(prepared.prediction.reward)
            & jnp.isfinite(prepared.prediction.discount)
            & jnp.all(jnp.isfinite(prepared.prediction.raw_predictions))
            & jnp.isfinite(event.reward)
            & jnp.isfinite(event.discount)
            & jnp.all(jnp.isfinite(event.next_base_observation))
            & (prepared.primitive_action >= 0)
            & (prepared.primitive_action < self._n_actions)
            & (event.discount >= 0.0)
            & (event.discount <= self._world.gamma)
        )
        source_clock_receipt_valid = jnp.all(
            prepared.source_state.model_state.step_words == state.model_state.step_words
        ) & jnp.all(prepared.source_state.planned_backup_words == state.planned_backup_words)

        targets = self._targets(
            prepared.base_observation,
            event.reward,
            event.discount,
            event.next_base_observation,
        )
        learner_result = self._learner.update(
            state.model_state.learner_state,
            prepared.input_features,
            targets,
        )
        update_prediction = self._decode_prediction(
            state.model_state,
            prepared.base_observation,
            learner_result.predictions,
        )
        source_prediction_matches = source_prediction_matches & _tree_exactly_equal(
            update_prediction,
            prepared.prediction,
        )
        prediction = prepared.prediction
        base_error = jnp.mean((prediction.next_base_observation - event.next_base_observation) ** 2)
        reward_error = prediction.reward - event.reward
        discount_error = prediction.discount - event.discount
        prediction_error = base_error + reward_error**2 + discount_error**2
        error_decay = jnp.asarray(self._world.error_decay, dtype=jnp.float32)
        next_global_error = jnp.where(
            state.model_state.step_count == 0,
            prediction_error,
            error_decay * state.model_state.model_error_ema
            + (1.0 - error_decay) * prediction_error,
        )
        next_generation_error = jnp.where(
            state.generation_error_valid,
            error_decay * state.generation_error_ema + (1.0 - error_decay) * prediction_error,
            prediction_error,
        )
        next_generation_words, generation_capacity = _checked_lifetime_words_increment(
            state.generation_update_words
        )
        next_generation_count = _saturating_int32_counter_increment(state.generation_update_count)
        updated_core = PrototypeFixedPhysicalWorldCoreState(
            learner_state=learner_result.state,
            observation_min=jnp.minimum(
                state.model_state.observation_min,
                jnp.minimum(
                    prepared.base_observation,
                    event.next_base_observation,
                ),
            ),
            observation_max=jnp.maximum(
                state.model_state.observation_max,
                jnp.maximum(
                    prepared.base_observation,
                    event.next_base_observation,
                ),
            ),
            reward_min=jnp.minimum(state.model_state.reward_min, event.reward),
            reward_max=jnp.maximum(state.model_state.reward_max, event.reward),
            model_error_ema=next_global_error,
            step_count=learner_result.state.step_count,
            step_words=learner_result.state.step_words,
        )
        updated_buffer = self._buffer.add(
            state.buffer_state,
            event.next_base_observation,
        )
        ordinary_candidate = PrototypeRoutedLinearWorldState(
            model_state=updated_core,
            buffer_state=updated_buffer,
            consumer_binding=state.consumer_binding,
            generation_update_count=next_generation_count,
            generation_update_words=next_generation_words,
            generation_birth_model_step_words=(
                state.generation_birth_model_step_words
            ),
            generation_error_ema=next_generation_error,
            generation_error_valid=jnp.asarray(True, dtype=jnp.bool_),
            planned_backup_count=state.planned_backup_count,
            planned_backup_words=state.planned_backup_words,
            schema_digest=state.schema_digest,
        )
        ordinary_candidate_values_finite = (
            _floating_tree_is_finite(learner_result.state)
            & jnp.all(jnp.isfinite(updated_buffer.observations))
            & jnp.isfinite(prediction_error)
        )
        ordinary_candidate_state_valid = self._state_is_valid(ordinary_candidate)
        (
            routed_learner,
            route_candidate_state,
            route_diagnostics,
        ) = self._route_learner_inputs(
            source_router_state,
            event.destination_binding.descriptors,
            learner_result.state,
        )
        selected_learner = jax.lax.cond(
            descriptors_changed,
            lambda _: routed_learner,
            lambda _: learner_result.state,
            operand=None,
        )
        selected_core = cast(
            PrototypeFixedPhysicalWorldCoreState,
            updated_core.replace(learner_state=selected_learner),
        )
        route_state_matches = jnp.where(
            descriptors_changed,
            _tree_exactly_equal(
                route_candidate_state,
                event.destination_router_state,
            ),
            _tree_exactly_equal(
                source_router_state,
                event.destination_router_state,
            ),
        )
        selected_generation_count = jnp.where(
            descriptors_changed,
            jnp.int32(0),
            next_generation_count,
        )
        selected_generation_words = jnp.where(
            descriptors_changed,
            jnp.zeros((2,), dtype=jnp.uint32),
            next_generation_words,
        )
        selected_birth_words = jnp.where(
            descriptors_changed,
            updated_core.step_words,
            state.generation_birth_model_step_words,
        )
        selected_generation_error = jnp.where(
            descriptors_changed,
            jnp.asarray(0.0, dtype=jnp.float32),
            next_generation_error,
        )
        selected_generation_error_valid = ~descriptors_changed
        candidate = PrototypeRoutedLinearWorldState(
            model_state=selected_core,
            buffer_state=updated_buffer,
            consumer_binding=event.destination_binding,
            generation_update_count=selected_generation_count,
            generation_update_words=selected_generation_words,
            generation_birth_model_step_words=selected_birth_words,
            generation_error_ema=selected_generation_error,
            generation_error_valid=selected_generation_error_valid,
            planned_backup_count=state.planned_backup_count,
            planned_backup_words=state.planned_backup_words,
            schema_digest=state.schema_digest,
        )
        candidate_values_finite = (
            _floating_tree_is_finite(selected_learner)
            & jnp.all(jnp.isfinite(updated_buffer.observations))
            & jnp.isfinite(prediction_error)
        )
        candidate_state_valid = self._state_is_valid(candidate)
        source_update_prerequisites = (
            prepared.prepared
            & source_state_valid
            & prepared_source_state_matches
            & prepared_source_router_matches
            & source_router_valid
            & source_router_matches
            & source_cache_matches
            & source_prediction_matches
            & event_values_valid
            & source_clock_receipt_valid
            & learner_result.update_applied
            & generation_capacity
        )
        ordinary_prerequisites = (
            source_update_prerequisites
            & ordinary_candidate_values_finite
            & ordinary_candidate_state_valid
        )
        prerequisites = (
            source_update_prerequisites
            & destination_binding_valid
            & destination_router_valid
            & destination_router_matches
            & bank_transition_consistent
            & route_diagnostics.valid
            & route_state_matches
            & candidate_values_finite
            & candidate_state_valid
        )
        selected = jax.lax.cond(
            prerequisites,
            lambda _: candidate,
            lambda _: state,
            operand=None,
        )
        ordinary_selected = jax.lax.cond(
            ordinary_prerequisites,
            lambda _: ordinary_candidate,
            lambda _: state,
            operand=None,
        )
        authenticated_prediction = (
            prepared.prepared
            & prepared_source_state_matches
            & prepared_source_router_matches
            & source_cache_matches
            & source_prediction_matches
        )
        safe_prediction = PrototypeFixedPhysicalWorldPrediction(
            next_base_observation=jnp.where(
                authenticated_prediction,
                prediction.next_base_observation,
                jnp.zeros((self._base_dim,), dtype=jnp.float32),
            ),
            reward=jnp.where(
                authenticated_prediction,
                prediction.reward,
                jnp.float32(0.0),
            ),
            discount=jnp.where(
                authenticated_prediction,
                prediction.discount,
                jnp.float32(0.0),
            ),
            raw_predictions=jnp.where(
                authenticated_prediction,
                prediction.raw_predictions,
                jnp.zeros((self._n_heads,), dtype=jnp.float32),
            ),
        )
        diagnostics = PrototypeRoutedLinearWorldDiagnostics(
            source_state_valid=source_state_valid,
            source_router_valid=source_router_valid,
            source_router_matches_binding=source_router_matches,
            destination_binding_valid=destination_binding_valid,
            destination_router_valid=destination_router_valid,
            destination_router_matches_binding=destination_router_matches,
            descriptors_changed=descriptors_changed,
            generation_changed=generation_changed,
            generation_is_successor=generation_is_successor,
            bank_transition_consistent=bank_transition_consistent,
            prepared_source_state_matches=prepared_source_state_matches,
            prepared_source_router_matches=prepared_source_router_matches,
            source_cache_matches=source_cache_matches,
            source_prediction_matches=source_prediction_matches,
            event_values_valid=event_values_valid,
            source_clock_receipt_valid=source_clock_receipt_valid,
            learner_update_applied=learner_result.update_applied,
            generation_counter_capacity_available=generation_capacity,
            route_attempted=descriptors_changed,
            route_candidate_valid=route_diagnostics.valid,
            route_state_matches_destination=route_state_matches,
            candidate_values_finite=candidate_values_finite,
            candidate_state_valid=candidate_state_valid,
            transaction_applied=prerequisites,
            physical_output_head_count=jnp.int32(self._n_heads),
            generated_output_head_count=jnp.int32(0),
            cache_entries_after=selected.buffer_state.size,
            model_step_words_before=state.model_state.step_words,
            model_step_words_after=selected.model_state.step_words,
            generation_update_words_after=selected.generation_update_words,
        )
        destination_result = PrototypeRoutedLinearWorldResult(
            state=selected,
            prediction=safe_prediction,
            targets=jnp.where(
                event_values_valid,
                targets,
                jnp.zeros((self._n_heads,), dtype=jnp.float32),
            ),
            prediction_error=jnp.where(
                event_values_valid & jnp.isfinite(prediction_error),
                prediction_error,
                jnp.float32(0.0),
            ),
            route_diagnostics=route_diagnostics,
            diagnostics=diagnostics,
        )
        ordinary_diagnostics = diagnostics.replace(
            candidate_values_finite=ordinary_candidate_values_finite,
            candidate_state_valid=ordinary_candidate_state_valid,
            transaction_applied=ordinary_prerequisites,
            cache_entries_after=ordinary_selected.buffer_state.size,
            model_step_words_after=ordinary_selected.model_state.step_words,
            generation_update_words_after=ordinary_selected.generation_update_words,
        )
        ordinary_result = PrototypeRoutedLinearWorldResult(
            state=ordinary_selected,
            prediction=safe_prediction,
            targets=jnp.where(
                event_values_valid,
                targets,
                jnp.zeros((self._n_heads,), dtype=jnp.float32),
            ),
            prediction_error=jnp.where(
                event_values_valid & jnp.isfinite(prediction_error),
                prediction_error,
                jnp.float32(0.0),
            ),
            route_diagnostics=route_diagnostics,
            diagnostics=ordinary_diagnostics,
        )
        return PrototypeRoutedLinearWorldPreparedAdoption(
            source_state=state,
            source_router_state=source_router_state,
            transition=event,
            ordinary_result=ordinary_result,
            destination_result=destination_result,
            ordinary_valid=ordinary_prerequisites,
            destination_valid=prerequisites,
            preparation_learner_update_evaluations=jnp.asarray(1, dtype=jnp.int32),
            preparation_router_evaluations=jnp.asarray(1, dtype=jnp.int32),
        )

    def _validate_prepared_adoption_static_contract(
        self,
        prepared: PrototypeRoutedLinearWorldPreparedAdoption,
    ) -> None:
        if type(prepared) is not PrototypeRoutedLinearWorldPreparedAdoption:
            raise TypeError(
                "prepared must be an exact PrototypeRoutedLinearWorldPreparedAdoption"
            )
        self._validate_state_static_contract(prepared.source_state)
        self._validate_router_static_contract(
            prepared.source_router_state,
            name="prepared.source_router_state",
        )
        self._validate_transition_static_contract(prepared.transition)
        if type(prepared.ordinary_result) is not PrototypeRoutedLinearWorldResult:
            raise TypeError("prepared ordinary result has the wrong exact type")
        if type(prepared.destination_result) is not PrototypeRoutedLinearWorldResult:
            raise TypeError("prepared destination result has the wrong exact type")
        self._validate_state_static_contract(prepared.ordinary_result.state)
        self._validate_state_static_contract(prepared.destination_result.state)
        for field_name in ("ordinary_valid", "destination_valid"):
            _require_array(
                getattr(prepared, field_name),
                name=f"prepared.{field_name}",
                shape=(),
                dtype=jnp.bool_,
            )
        for field_name in (
            "preparation_learner_update_evaluations",
            "preparation_router_evaluations",
        ):
            _require_array(
                getattr(prepared, field_name),
                name=f"prepared.{field_name}",
                shape=(),
                dtype=jnp.int32,
            )

    def external_readiness_receipt(
        self,
        prepared: PrototypeRoutedLinearWorldPreparedAdoption,
        all_consumers_ready: Array,
    ) -> PrototypeRoutedLinearWorldExternalReadinessReceipt:
        """Bind one external all-consumer verdict to every preparation leaf."""

        self._validate_prepared_adoption_static_contract(prepared)
        ready = _require_array(
            all_consumers_ready,
            name="all_consumers_ready",
            shape=(),
            dtype=jnp.bool_,
        )
        return PrototypeRoutedLinearWorldExternalReadinessReceipt(
            prepared_adoption=prepared,
            all_consumers_ready=ready,
        )

    def external_transaction_resource_budget(
        self,
        prepared: PrototypeRoutedLinearWorldPreparedAdoption,
        receipt: PrototypeRoutedLinearWorldExternalReadinessReceipt,
    ) -> PrototypeRoutedLinearWorldExternalTransactionResourceBudget:
        """Measure logical serialized leaves and fixed zero-recompute adoption."""

        self._validate_prepared_adoption_static_contract(prepared)
        if type(receipt) is not PrototypeRoutedLinearWorldExternalReadinessReceipt:
            raise TypeError(
                "receipt must be an exact "
                "PrototypeRoutedLinearWorldExternalReadinessReceipt"
            )
        self._validate_prepared_adoption_static_contract(receipt.prepared_adoption)
        prepared_nbytes = _tree_nbytes(prepared)
        receipt_nbytes = _tree_nbytes(receipt)
        persistent_nbytes = _tree_nbytes(prepared.source_state)
        return PrototypeRoutedLinearWorldExternalTransactionResourceBudget(
            persistent_state_nbytes_before=persistent_nbytes,
            persistent_state_nbytes_after=persistent_nbytes,
            source_router_state_nbytes=_tree_nbytes(prepared.source_router_state),
            prepared_adoption_logical_nbytes=prepared_nbytes,
            readiness_receipt_logical_nbytes=receipt_nbytes,
            simultaneous_logical_transient_nbytes=prepared_nbytes + receipt_nbytes,
            learner_update_evaluations_per_prepare=1,
            learner_update_evaluations_per_adopt=0,
            learner_update_evaluations_per_transaction=1,
            router_evaluations_per_prepare=1,
            router_evaluations_per_adopt=0,
            router_evaluations_per_transaction=1,
            persistent_capacity_growth=0,
        )

    def adopt_prepared_route(
        self,
        state: PrototypeRoutedLinearWorldState,
        source_router_state: FeatureBankRouterState,
        prepared: PrototypeRoutedLinearWorldPreparedAdoption,
        receipt: PrototypeRoutedLinearWorldExternalReadinessReceipt,
    ) -> PrototypeRoutedLinearWorldAdoptionResult:
        """Adopt the routed destination or retain the computed source update.

        Adoption exact-matches source and receipt content without calling the
        world learner, predictor, buffer, or router.  This is unkeyed integrity,
        not caller authentication: coordinated forgery of both preparation and
        receipt is outside this primitive's trust boundary.
        """

        self._validate_state_static_contract(state)
        self._validate_router_static_contract(
            source_router_state,
            name="source_router_state",
        )
        self._validate_prepared_adoption_static_contract(prepared)
        if type(receipt) is not PrototypeRoutedLinearWorldExternalReadinessReceipt:
            raise TypeError(
                "receipt must be an exact "
                "PrototypeRoutedLinearWorldExternalReadinessReceipt"
            )
        self._validate_prepared_adoption_static_contract(receipt.prepared_adoption)
        _require_array(
            receipt.all_consumers_ready,
            name="receipt.all_consumers_ready",
            shape=(),
            dtype=jnp.bool_,
        )

        source_state_matches = _tree_exactly_equal(state, prepared.source_state)
        source_router_matches = _tree_exactly_equal(
            source_router_state,
            prepared.source_router_state,
        )
        receipt_matches = _tree_exactly_equal(
            prepared,
            receipt.prepared_adoption,
        )
        exact_work = (
            prepared.preparation_learner_update_evaluations == jnp.int32(1)
        ) & (prepared.preparation_router_evaluations == jnp.int32(1))
        selected_branch_valid = jnp.where(
            receipt.all_consumers_ready,
            prepared.destination_valid,
            prepared.ordinary_valid,
        )
        preparation_valid = selected_branch_valid & exact_work
        authenticated = (
            source_state_matches
            & source_router_matches
            & receipt_matches
            & exact_work
        )
        applied = authenticated & preparation_valid
        selected = jax.lax.cond(
            receipt.all_consumers_ready,
            lambda _: prepared.destination_result,
            lambda _: prepared.ordinary_result,
            operand=None,
        )
        nested_applied = applied & selected.diagnostics.transaction_applied
        selected_state = jax.lax.cond(
            nested_applied,
            lambda _: selected.state,
            lambda _: state,
            operand=None,
        )
        result = selected.replace(
            state=selected_state,
            prediction=PrototypeFixedPhysicalWorldPrediction(
                next_base_observation=jnp.where(
                    nested_applied,
                    selected.prediction.next_base_observation,
                    jnp.zeros_like(selected.prediction.next_base_observation),
                ),
                reward=jnp.where(
                    nested_applied,
                    selected.prediction.reward,
                    jnp.float32(0.0),
                ),
                discount=jnp.where(
                    nested_applied,
                    selected.prediction.discount,
                    jnp.float32(0.0),
                ),
                raw_predictions=jnp.where(
                    nested_applied,
                    selected.prediction.raw_predictions,
                    jnp.zeros_like(selected.prediction.raw_predictions),
                ),
            ),
            targets=jnp.where(
                nested_applied,
                selected.targets,
                jnp.zeros_like(selected.targets),
            ),
            prediction_error=jnp.where(
                nested_applied,
                selected.prediction_error,
                jnp.float32(0.0),
            ),
            diagnostics=selected.diagnostics.replace(
                transaction_applied=nested_applied,
                cache_entries_after=selected_state.buffer_state.size,
                model_step_words_after=selected_state.model_state.step_words,
                generation_update_words_after=(
                    selected_state.generation_update_words
                ),
            ),
        )
        external_rollback = (
            nested_applied
            & ~receipt.all_consumers_ready
            & prepared.destination_result.diagnostics.route_attempted
        )
        zero_work = jnp.asarray(0, dtype=jnp.int32)
        return PrototypeRoutedLinearWorldAdoptionResult(
            result=result,
            diagnostics=PrototypeRoutedLinearWorldAdoptionDiagnostics(
                source_state_matches=source_state_matches,
                source_router_state_matches=source_router_matches,
                receipt_matches_preparation=receipt_matches,
                ordinary_successor_valid=prepared.ordinary_valid,
                destination_candidate_valid=prepared.destination_valid,
                preparation_internally_valid=preparation_valid,
                all_consumers_ready=receipt.all_consumers_ready,
                destination_adopted=nested_applied & receipt.all_consumers_ready,
                ordinary_update_retained=nested_applied & ~receipt.all_consumers_ready,
                external_route_rolled_back=external_rollback,
                transaction_applied=nested_applied,
                rejected=~nested_applied,
                preparation_learner_update_evaluations=(
                    prepared.preparation_learner_update_evaluations
                ),
                adoption_learner_update_evaluations=zero_work,
                total_learner_update_evaluations=(
                    prepared.preparation_learner_update_evaluations
                ),
                preparation_router_evaluations=(
                    prepared.preparation_router_evaluations
                ),
                adoption_router_evaluations=zero_work,
                total_router_evaluations=prepared.preparation_router_evaluations,
            ),
        )

    def prepare_plan(
        self,
        state: PrototypeRoutedLinearWorldState,
        oak_state: OaKState,
        source_router_state: FeatureBankRouterState,
        request: PrototypeRoutedLinearWorldPlanRequest,
    ) -> PrototypeRoutedLinearWorldPlanPrepareResult:
        """Mint a source-owned world/OaK snapshot and deterministic plan cache."""

        self._validate_state_static_contract(state)
        self._validate_oak_static_contract(oak_state, name="oak_state")
        self._validate_router_static_contract(
            source_router_state,
            name="source_router_state",
        )
        self._validate_plan_static_contract(request)
        return cast(
            PrototypeRoutedLinearWorldPlanPrepareResult,
            self._prepare_plan_jit(
                state,
                oak_state,
                source_router_state,
                request,
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _prepare_plan_jit(
        self,
        state: PrototypeRoutedLinearWorldState,
        oak_state: OaKState,
        source_router_state: FeatureBankRouterState,
        request: PrototypeRoutedLinearWorldPlanRequest,
    ) -> PrototypeRoutedLinearWorldPlanPrepareResult:
        source_state_valid = self._state_is_valid(state)
        source_oak_state_valid = self._oak_state_is_valid(oak_state)
        router_valid = self._router_valid(source_router_state)
        router_matches_binding = self._router_matches_binding(
            source_router_state,
            state.consumer_binding,
        )
        request_router_matches_source = _tree_exactly_equal(
            request.router_state,
            source_router_state,
        )
        consumer_binding_valid = self._binding_valid(request.consumer_binding)
        consumer_binding_matches = self._bindings_equal(
            state.consumer_binding,
            request.consumer_binding,
        )
        clock_receipts_valid = (
            jnp.all(request.expected_model_step_words == state.model_state.step_words)
            & jnp.all(request.expected_planned_backup_words == state.planned_backup_words)
            & jnp.all(request.expected_oak_step_words == oak_state.step_words)
        )
        anchor_available = (request.anchor_index >= 0) & (
            request.anchor_index < state.buffer_state.size
        )
        safe_anchor_index = jnp.clip(
            request.anchor_index,
            0,
            self._config.anchor_capacity - 1,
        )
        anchor_base = state.buffer_state.observations[safe_anchor_index]
        anchor_augmented = self._augment(state.consumer_binding, anchor_base)
        action_valid = (request.primitive_action >= 0) & (
            request.primitive_action < self._n_actions
        )
        prediction = self._predict(
            state.model_state,
            anchor_base,
            anchor_augmented,
            request.primitive_action,
        )
        predicted_next_augmented = self._augment(
            state.consumer_binding,
            prediction.next_base_observation,
        )
        prediction_values_finite = (
            jnp.all(jnp.isfinite(anchor_base))
            & jnp.all(jnp.isfinite(anchor_augmented))
            & jnp.all(jnp.isfinite(prediction.next_base_observation))
            & jnp.isfinite(prediction.reward)
            & jnp.isfinite(prediction.discount)
            & jnp.all(jnp.isfinite(prediction.raw_predictions))
            & jnp.all(jnp.isfinite(predicted_next_augmented))
        )
        prepared_valid = (
            source_state_valid
            & source_oak_state_valid
            & router_valid
            & router_matches_binding
            & request_router_matches_source
            & consumer_binding_valid
            & consumer_binding_matches
            & clock_receipts_valid
            & anchor_available
            & action_valid
            & prediction_values_finite
        )
        prepared = PrototypeRoutedLinearWorldPreparedPlan(
            source_state=state,
            source_oak_state=oak_state,
            source_router_state=source_router_state,
            request=request,
            anchor_base_observation=anchor_base,
            anchor_augmented_observation=anchor_augmented,
            predicted_next_augmented_observation=predicted_next_augmented,
            prediction=prediction,
            prepared=prepared_valid,
        )
        diagnostics = PrototypeRoutedLinearWorldPlanPrepareDiagnostics(
            source_state_valid=source_state_valid,
            source_oak_state_valid=source_oak_state_valid,
            router_valid=router_valid,
            router_matches_binding=router_matches_binding,
            request_router_matches_source=request_router_matches_source,
            consumer_binding_valid=consumer_binding_valid,
            consumer_binding_matches=consumer_binding_matches,
            clock_receipts_valid=clock_receipts_valid,
            anchor_available=anchor_available,
            action_valid=action_valid,
            prediction_values_finite=prediction_values_finite,
            prepared=prepared_valid,
        )
        return PrototypeRoutedLinearWorldPlanPrepareResult(
            prepared=prepared,
            diagnostics=diagnostics,
        )

    def plan_one(
        self,
        state: PrototypeRoutedLinearWorldState,
        oak_state: OaKState,
        source_router_state: FeatureBankRouterState,
        prepared: PrototypeRoutedLinearWorldPreparedPlan,
    ) -> PrototypeRoutedLinearWorldPlanResult:
        """Consume one authenticated plan snapshot into one OaK base backup."""

        self._validate_state_static_contract(state)
        self._validate_oak_static_contract(oak_state, name="oak_state")
        self._validate_router_static_contract(
            source_router_state,
            name="source_router_state",
        )
        self._validate_prepared_plan_static_contract(prepared)
        return cast(
            PrototypeRoutedLinearWorldPlanResult,
            self._plan_one_jit(
                state,
                oak_state,
                source_router_state,
                prepared,
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _plan_one_jit(
        self,
        state: PrototypeRoutedLinearWorldState,
        oak_state: OaKState,
        source_router_state: FeatureBankRouterState,
        prepared: PrototypeRoutedLinearWorldPreparedPlan,
    ) -> PrototypeRoutedLinearWorldPlanResult:
        request = prepared.request
        source_state_valid = self._state_is_valid(state)
        source_oak_state_valid = self._oak_state_is_valid(oak_state)
        prepared_source_state_matches = _tree_exactly_equal(
            state,
            prepared.source_state,
        )
        prepared_source_oak_state_matches = _tree_exactly_equal(
            oak_state,
            prepared.source_oak_state,
        )
        prepared_source_router_matches = _tree_exactly_equal(
            source_router_state,
            prepared.source_router_state,
        )
        router_valid = self._router_valid(source_router_state)
        router_matches_binding = self._router_matches_binding(
            source_router_state,
            state.consumer_binding,
        )
        consumer_binding_valid = self._binding_valid(request.consumer_binding)
        consumer_binding_matches = self._bindings_equal(
            state.consumer_binding,
            request.consumer_binding,
        )
        clock_receipts_valid = (
            jnp.all(request.expected_model_step_words == state.model_state.step_words)
            & jnp.all(request.expected_planned_backup_words == state.planned_backup_words)
            & jnp.all(request.expected_oak_step_words == oak_state.step_words)
        )
        anchor_available = (request.anchor_index >= 0) & (
            request.anchor_index < state.buffer_state.size
        )
        safe_anchor_index = jnp.clip(
            request.anchor_index,
            0,
            self._config.anchor_capacity - 1,
        )
        anchor_base = state.buffer_state.observations[safe_anchor_index]
        anchor_augmented = self._augment(state.consumer_binding, anchor_base)
        action_valid = (request.primitive_action >= 0) & (
            request.primitive_action < self._n_actions
        )
        prediction = self._predict(
            state.model_state,
            anchor_base,
            anchor_augmented,
            request.primitive_action,
        )
        predicted_next_augmented = self._augment(
            state.consumer_binding,
            prediction.next_base_observation,
        )
        prepared_cache_matches = (
            _tree_exactly_equal(request.router_state, source_router_state)
            & _tree_exactly_equal(anchor_base, prepared.anchor_base_observation)
            & _tree_exactly_equal(
                anchor_augmented,
                prepared.anchor_augmented_observation,
            )
            & _tree_exactly_equal(prediction, prepared.prediction)
            & _tree_exactly_equal(
                predicted_next_augmented,
                prepared.predicted_next_augmented_observation,
            )
        )
        generation_warm = _words_at_least_small(
            state.generation_update_words,
            self._config.planning_warmup_steps,
        )
        generation_error_ready = state.generation_error_valid
        generation_error_below_limit = state.generation_error_ema <= jnp.asarray(
            self._config.max_generation_model_error,
            dtype=jnp.float32,
        )
        next_planned_words, planned_word_capacity = _checked_lifetime_words_increment(
            state.planned_backup_words
        )
        next_planned_count = _saturating_int32_counter_increment(state.planned_backup_count)
        planning_capacity = (
            planned_word_capacity
            & (state.planned_backup_words[0] == jnp.uint32(0))
            & (
                state.planned_backup_words[1]
                < jnp.asarray(self._config.max_planned_backups, dtype=jnp.uint32)
            )
        )
        prediction_values_finite = (
            jnp.all(jnp.isfinite(anchor_base))
            & jnp.all(jnp.isfinite(anchor_augmented))
            & jnp.all(jnp.isfinite(prediction.next_base_observation))
            & jnp.isfinite(prediction.reward)
            & jnp.isfinite(prediction.discount)
            & jnp.all(jnp.isfinite(prediction.raw_predictions))
            & jnp.all(jnp.isfinite(predicted_next_augmented))
        )

        temp_stomp = oak_state.stomp_state.replace(
            base_last_obs=anchor_augmented,
            base_last_action=request.primitive_action,
            last_primitive_action=request.primitive_action,
            executing_option=jnp.asarray(-1, dtype=jnp.int32),
        )
        temp_oak = cast(OaKState, oak_state.replace(stomp_state=temp_stomp))
        dream_result = self._oak.update(
            temp_oak,
            prediction.reward,
            predicted_next_augmented,
            prediction.discount,
            enable_option_planning=False,
        )
        candidate_base_learner = dream_result.state.stomp_state.base_learner_state
        candidate_base_finite = _floating_tree_is_finite(candidate_base_learner)
        candidate_oak = cast(
            OaKState,
            oak_state.replace(
                stomp_state=oak_state.stomp_state.replace(
                    base_learner_state=candidate_base_learner,
                )
            ),
        )
        candidate_oak_state_valid = self._oak_state_is_valid(candidate_oak)
        candidate_world = cast(
            PrototypeRoutedLinearWorldState,
            state.replace(
                planned_backup_count=next_planned_count,
                planned_backup_words=next_planned_words,
            ),
        )
        candidate_world_valid = self._state_is_valid(candidate_world)
        base_learner_changed = ~_tree_exactly_equal(
            candidate_base_learner,
            oak_state.stomp_state.base_learner_state,
        )
        transaction_applied = (
            prepared.prepared
            & source_state_valid
            & source_oak_state_valid
            & prepared_source_state_matches
            & prepared_source_oak_state_matches
            & prepared_source_router_matches
            & prepared_cache_matches
            & router_valid
            & router_matches_binding
            & consumer_binding_valid
            & consumer_binding_matches
            & clock_receipts_valid
            & anchor_available
            & action_valid
            & jnp.asarray(self._config.planning_enabled, dtype=jnp.bool_)
            & generation_warm
            & generation_error_ready
            & generation_error_below_limit
            & planning_capacity
            & prediction_values_finite
            & dream_result.update_applied
            & candidate_base_finite
            & candidate_oak_state_valid
            & candidate_world_valid
        )
        selected_world = jax.lax.cond(
            transaction_applied,
            lambda _: candidate_world,
            lambda _: state,
            operand=None,
        )
        selected_oak = jax.lax.cond(
            transaction_applied,
            lambda _: candidate_oak,
            lambda _: oak_state,
            operand=None,
        )
        diagnostics = PrototypeRoutedLinearWorldPlanDiagnostics(
            source_state_valid=source_state_valid,
            source_oak_state_valid=source_oak_state_valid,
            prepared_source_state_matches=prepared_source_state_matches,
            prepared_source_oak_state_matches=prepared_source_oak_state_matches,
            prepared_source_router_matches=prepared_source_router_matches,
            prepared_cache_matches=prepared_cache_matches,
            router_valid=router_valid,
            router_matches_binding=router_matches_binding,
            consumer_binding_valid=consumer_binding_valid,
            consumer_binding_matches=consumer_binding_matches,
            clock_receipts_valid=clock_receipts_valid,
            anchor_available=anchor_available,
            action_valid=action_valid,
            planning_enabled=jnp.asarray(
                self._config.planning_enabled,
                dtype=jnp.bool_,
            ),
            generation_warm=generation_warm,
            generation_error_ready=generation_error_ready,
            generation_error_below_limit=generation_error_below_limit,
            planning_capacity_available=planning_capacity,
            prediction_values_finite=prediction_values_finite,
            oak_update_applied=dream_result.update_applied,
            candidate_base_learner_finite=candidate_base_finite,
            candidate_oak_state_valid=candidate_oak_state_valid,
            candidate_state_valid=candidate_world_valid,
            base_learner_changed=base_learner_changed,
            transaction_applied=transaction_applied,
            pair_products_evaluated=jnp.int32(2 * self._active_slots),
            planned_backup_words_before=state.planned_backup_words,
            planned_backup_words_after=selected_world.planned_backup_words,
        )
        return PrototypeRoutedLinearWorldPlanResult(
            state=selected_world,
            oak_state=selected_oak,
            anchor_base_observation=anchor_base,
            anchor_augmented_observation=anchor_augmented,
            predicted_next_augmented_observation=predicted_next_augmented,
            prediction=prediction,
            td_error=jnp.where(
                transaction_applied,
                dream_result.td_error,
                jnp.asarray(0.0, dtype=jnp.float32),
            ),
            diagnostics=diagnostics,
        )

    def resource_budget(self) -> PrototypeRoutedLinearWorldResourceBudget:
        """Return exact state/cache bytes and hard two-phase work maxima."""

        lifecycle = PrototypeFeatureLifecycle(self._feature)
        lifecycle_state, binding = lifecycle.init_bound(jr.key(0))
        template = self.init(jr.key(1), binding, lifecycle_state.router_state)
        learner_nbytes = _tree_nbytes(template.model_state.learner_state)
        model_nbytes = _tree_nbytes(template.model_state)
        buffer_nbytes = _tree_nbytes(template.buffer_state)
        binding_nbytes = _tree_nbytes(template.consumer_binding)
        digest_nbytes = _tree_nbytes(template.schema_digest)
        total_nbytes = _tree_nbytes(template)
        metadata_nbytes = total_nbytes - model_nbytes - buffer_nbytes
        zero_prediction = PrototypeFixedPhysicalWorldPrediction(
            next_base_observation=jnp.zeros(
                (self._base_dim,),
                dtype=jnp.float32,
            ),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            discount=jnp.asarray(0.0, dtype=jnp.float32),
            raw_predictions=jnp.zeros((self._n_heads,), dtype=jnp.float32),
        )
        prepared_transition_template = PrototypeRoutedLinearWorldPreparedTransition(
            source_state=template,
            source_router_state=lifecycle_state.router_state,
            base_observation=jnp.zeros((self._base_dim,), dtype=jnp.float32),
            cached_augmented_observation=jnp.zeros(
                (self._total_dim,),
                dtype=jnp.float32,
            ),
            input_features=jnp.zeros((self._input_dim,), dtype=jnp.float32),
            primitive_action=jnp.asarray(0, dtype=jnp.int32),
            prediction=zero_prediction,
            prepared=jnp.asarray(False, dtype=jnp.bool_),
        )
        plan_request_template = PrototypeRoutedLinearWorldPlanRequest(
            anchor_index=jnp.asarray(0, dtype=jnp.int32),
            primitive_action=jnp.asarray(0, dtype=jnp.int32),
            consumer_binding=binding,
            router_state=lifecycle_state.router_state,
            expected_model_step_words=jnp.zeros((2,), dtype=jnp.uint32),
            expected_planned_backup_words=jnp.zeros((2,), dtype=jnp.uint32),
            expected_oak_step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )
        prepared_plan_template = PrototypeRoutedLinearWorldPreparedPlan(
            source_state=template,
            source_oak_state=self._oak_template,
            source_router_state=lifecycle_state.router_state,
            request=plan_request_template,
            anchor_base_observation=jnp.zeros(
                (self._base_dim,),
                dtype=jnp.float32,
            ),
            anchor_augmented_observation=jnp.zeros(
                (self._total_dim,),
                dtype=jnp.float32,
            ),
            predicted_next_augmented_observation=jnp.zeros(
                (self._total_dim,),
                dtype=jnp.float32,
            ),
            prediction=zero_prediction,
            prepared=jnp.asarray(False, dtype=jnp.bool_),
        )
        oak_nbytes = _tree_nbytes(self._oak_template)
        prepared_transition_nbytes = _tree_nbytes(prepared_transition_template)
        prepared_plan_nbytes = _tree_nbytes(prepared_plan_template)
        expected_buffer_nbytes = 4 * self._config.anchor_capacity * self._base_dim + 8
        expected_binding_nbytes = 8 * self._active_slots + 12
        if buffer_nbytes != expected_buffer_nbytes:
            raise RuntimeError("routed linear world buffer byte formula drifted")
        if binding_nbytes != expected_binding_nbytes:
            raise RuntimeError("routed linear world binding byte formula drifted")
        if digest_nbytes != _SCHEMA_DIGEST_NBYTES:
            raise RuntimeError("routed linear world digest byte formula drifted")
        return PrototypeRoutedLinearWorldResourceBudget(
            mechanism_status=PROTOTYPE_ROUTED_LINEAR_WORLD_MECHANISM_STATUS,
            scientific_promotion_allowed=(
                PROTOTYPE_ROUTED_LINEAR_WORLD_SCIENTIFIC_PROMOTION_ALLOWED
            ),
            evidence_level=PROTOTYPE_ROUTED_LINEAR_WORLD_EVIDENCE_LEVEL,
            base_feature_dim=self._base_dim,
            active_pair_slots=self._active_slots,
            total_feature_dim=self._total_dim,
            n_primitive_actions=self._n_actions,
            physical_output_heads=self._n_heads,
            generated_output_heads=0,
            model_input_dim=self._input_dim,
            anchor_capacity=self._config.anchor_capacity,
            planning_enabled=self._config.planning_enabled,
            learner_state_nbytes=learner_nbytes,
            model_core_state_nbytes=model_nbytes,
            buffer_state_nbytes=buffer_nbytes,
            consumer_binding_nbytes=binding_nbytes,
            schema_digest_nbytes=digest_nbytes,
            wrapper_metadata_nbytes=metadata_nbytes,
            persistent_state_nbytes=total_nbytes,
            source_oak_state_nbytes=oak_nbytes,
            prepared_transition_cache_nbytes=prepared_transition_nbytes,
            prepared_plan_cache_nbytes=prepared_plan_nbytes,
            # One float32 weight plus one float32 eligibility trace for every
            # physical head and dynamic input slot.
            incremental_dynamic_input_nbytes=(8 * self._n_heads * self._active_slots),
            routed_input_feature_groups=2 * self._n_heads,
            routed_input_scalars=2 * self._n_heads * self._total_dim,
            routed_dynamic_input_scalars=(2 * self._n_heads * self._active_slots),
            preserved_action_input_scalars=2 * self._n_heads * self._n_actions,
            max_router_calls_per_real_transition=1,
            max_pair_products_per_transition_prepare=self._active_slots,
            max_pair_products_per_transition_consume=self._active_slots,
            max_pair_products_per_real_transition=2 * self._active_slots,
            max_pair_products_per_plan_prepare=2 * self._active_slots,
            max_pair_products_per_plan_consume=2 * self._active_slots,
            max_pair_products_per_planning_call=4 * self._active_slots,
            max_pair_products_per_planning_transaction=4 * self._active_slots,
            max_world_forwards_per_transition_prepare=1,
            # Consume recomputes one authenticated prediction, then the
            # learner update performs its own predict-before-update forward.
            max_world_forwards_per_transition_consume=2,
            max_world_forwards_per_real_transition=3,
            max_world_backwards_per_real_transition=1,
            max_world_forwards_per_plan_prepare=1,
            max_world_forwards_per_plan_consume=1,
            max_world_forwards_per_planning_call=2,
            max_oak_updates_per_planning_call=1,
            max_oak_base_backups_per_planning_call=1,
            persistent_capacity_growth=0,
        )


__all__ = [
    "PROTOTYPE_ROUTED_LINEAR_WORLD_CONFIG_SCHEMA",
    "PROTOTYPE_ROUTED_LINEAR_WORLD_EVIDENCE_LEVEL",
    "PROTOTYPE_ROUTED_LINEAR_WORLD_MECHANISM_STATUS",
    "PROTOTYPE_ROUTED_LINEAR_WORLD_SCIENTIFIC_PROMOTION_ALLOWED",
    "PROTOTYPE_ROUTED_LINEAR_WORLD_STATE_SCHEMA",
    "PrototypeFixedPhysicalWorldCoreState",
    "PrototypeFixedPhysicalWorldPrediction",
    "PrototypeRoutedLinearWorldAdoptionDiagnostics",
    "PrototypeRoutedLinearWorldAdoptionResult",
    "PrototypeRoutedLinearWorldConfig",
    "PrototypeRoutedLinearWorldDiagnostics",
    "PrototypeRoutedLinearWorldExternalReadinessReceipt",
    "PrototypeRoutedLinearWorldExternalTransactionResourceBudget",
    "PrototypeRoutedLinearWorldModel",
    "PrototypeRoutedLinearWorldPlanDiagnostics",
    "PrototypeRoutedLinearWorldPlanPrepareDiagnostics",
    "PrototypeRoutedLinearWorldPlanPrepareResult",
    "PrototypeRoutedLinearWorldPlanRequest",
    "PrototypeRoutedLinearWorldPreparedPlan",
    "PrototypeRoutedLinearWorldPreparedAdoption",
    "PrototypeRoutedLinearWorldPreparedTransition",
    "PrototypeRoutedLinearWorldPrepareDiagnostics",
    "PrototypeRoutedLinearWorldPrepareResult",
    "PrototypeRoutedLinearWorldPlanResult",
    "PrototypeRoutedLinearWorldResourceBudget",
    "PrototypeRoutedLinearWorldResult",
    "PrototypeRoutedLinearWorldState",
    "PrototypeRoutedLinearWorldTransition",
]
