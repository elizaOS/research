# mypy: disable-error-code="attr-defined,call-arg,no-any-return"
"""Atomic Prototype-to-routed-linear-ensemble adoption seam.

This L0 adapter composes one existing :class:`PrototypeAgent` with one
external :class:`RoutedLinearWorldModelEnsemble`.  Prototype remains the sole
feature-lifecycle and ``FeatureBankRouterState`` owner.  The ensemble stores
only its consumer binding and consumes the exact source and applied
destination bank identities exposed by the Prototype source/result states.

Preparation caches the ensemble prediction from the old bank and invokes
Prototype exactly once.  Adoption checks an unkeyed, integrity-bound receipt,
updates every ensemble member on that cached old-bank input, evaluates one
stacked ensemble route, and atomically returns either both destinations or the
complete composite source.  It never reruns feature curation.  A complete
event evaluates three bank mappings honestly: the existing Prototype
lifecycle evaluates its input and output mappings (two), and the external
ensemble evaluates one stacked member mapping.

The receipt is source-bound and integrity-bound by exact array content.  It is
not authenticated and grants no keyed authority.  This adapter grants no
planning, dispatch, safety, evidence, or scientific-promotion authority and is
strictly separate from the historical Prototype v18 composition.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array

from alberta_framework.core.checkpoints import (
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)
from alberta_framework.core.prototype_agent import (
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeAgentState,
    PrototypeCandidateUpdateAuditEvidence,
    PrototypeExperientialMemoryInput,
    PrototypeGradientJoyEvidence,
    PrototypePartnerPolicyFusionFeedback,
    PrototypePartnerPolicyFusionInput,
    PrototypeTransition,
    PrototypeUpdateResult,
    measure_prototype_agent_state_resources,
)
from alberta_framework.core.prototype_feature_lifecycle import (
    PrototypeFeatureConsumerBinding,
    PrototypeFeatureLifecycleConfig,
    PrototypeFeatureLifecycleState,
)
from alberta_framework.core.routed_linear_world_model_ensemble import (
    RoutedLinearWorldModelEnsemble,
    RoutedLinearWorldModelEnsembleConfig,
    RoutedLinearWorldModelEnsemblePreparedTransition,
    RoutedLinearWorldModelEnsembleResult,
    RoutedLinearWorldModelEnsembleState,
    RoutedLinearWorldModelEnsembleTransition,
    measure_routed_linear_world_model_ensemble_state_nbytes,
)
from alberta_framework.core.state_builder import IdentityStateBuilderConfig

PROTOTYPE_ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_ADAPTER_CONFIG_SCHEMA = (
    "alberta.prototype-routed-linear-world-model-ensemble-adapter.config.v1"
)
PROTOTYPE_ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_ADAPTER_STATE_SCHEMA = (
    "alberta.prototype-routed-linear-world-model-ensemble-adapter.state.v1"
)
PROTOTYPE_ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_ADAPTER_RECEIPT_SCHEMA = (
    "alberta.prototype-routed-linear-world-model-ensemble-adapter.receipt.v1"
)
PROTOTYPE_ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_ADAPTER_CHECKPOINT_SCHEMA = (
    "alberta.prototype-routed-linear-world-model-ensemble-adapter.checkpoint.v1"
)
PROTOTYPE_ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_ADAPTER_EVIDENCE_LEVEL = "L0"
PROTOTYPE_ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_ADAPTER_OUTCOME_STATUS = (
    "not_assessed"
)
PROTOTYPE_ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_ADAPTER_SCIENTIFIC_PROMOTION_ALLOWED = (
    False
)

_SCHEMA_DIGEST_NBYTES = 32
_AUTHORITY_SEMANTICS = (
    "one-prototype-lifecycle-router-owner;external-ensemble-binding-only;"
    "source-predict;prototype-update-once;no-curation-recompute;"
    "atomic-composite-adoption"
)
_MAPPING_SEMANTICS = (
    "three-evaluations-per-complete-event:prototype-input-route=1;"
    "prototype-output-route=1;ensemble-stacked-route=1"
)


def _config_digest(config: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tree_equal(left: object, right: object) -> Array:
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    if cast(Any, left_tree) != right_tree or len(left_leaves) != len(right_leaves):
        return jnp.asarray(False, dtype=jnp.bool_)
    equal = jnp.asarray(True, dtype=jnp.bool_)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = jnp.asarray(left_leaf)
        right_array = jnp.asarray(right_leaf)
        if jax.dtypes.issubdtype(left_array.dtype, jax.dtypes.prng_key):
            equal = equal & jnp.array_equal(
                jr.key_data(left_array),
                jr.key_data(right_array),
            )
        elif left_array.dtype == jnp.dtype(jnp.float32):
            equal = equal & jnp.array_equal(
                jax.lax.bitcast_convert_type(left_array, jnp.uint32),
                jax.lax.bitcast_convert_type(right_array, jnp.uint32),
            )
        else:
            equal = equal & jnp.array_equal(left_array, right_array)
    return equal


def _words_successor(source: Array, destination: Array) -> Array:
    uint32_max = jnp.asarray(0xFFFFFFFF, dtype=jnp.uint32)
    low = source[1] + jnp.asarray(1, dtype=jnp.uint32)
    carry = (low == 0).astype(jnp.uint32)
    high = source[0] + carry
    capacity = ~(
        (source[0] == uint32_max)
        & (source[1] == uint32_max)
    )
    return capacity & jnp.array_equal(destination, jnp.stack((high, low)))


def _tree_nbytes(tree: object) -> int:
    total = 0
    for leaf in jax.tree.leaves(tree):
        value = jnp.asarray(leaf)
        if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
            value = jr.key_data(value)
        total += int(value.nbytes)
    return total


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeRoutedLinearWorldModelEnsembleAdapterConfig:
    """Exact non-v18 composition of one Prototype and one routed ensemble."""

    prototype: PrototypeAgentConfig
    ensemble: RoutedLinearWorldModelEnsembleConfig

    def __post_init__(self) -> None:
        if type(self.prototype) is not PrototypeAgentConfig:
            raise TypeError("prototype must be an exact PrototypeAgentConfig")
        if type(self.ensemble) is not RoutedLinearWorldModelEnsembleConfig:
            raise TypeError(
                "ensemble must be an exact RoutedLinearWorldModelEnsembleConfig"
            )
        feature = self.prototype.prototype_feature_lifecycle
        if type(feature) is not PrototypeFeatureLifecycleConfig:
            raise ValueError(
                "adapter requires one exact Prototype feature lifecycle"
            )
        if feature.managed_horde_demons <= 0:
            raise ValueError("adapter requires the Prototype-managed linear Horde")
        if type(self.prototype.state_builder) is not IdentityStateBuilderConfig:
            raise ValueError("adapter requires an exact Identity state_builder")
        if self.prototype.prototype_atomic_feature_world_memory is not None:
            raise ValueError(
                "adapter is separate from Prototype v18; disable the v18 composition"
            )
        if any(
            model is not None
            for model in (
                self.prototype.world_model,
                self.prototype.world_model_ensemble,
                self.prototype.model_replay_rehearsal,
                self.prototype.recurrent_latent_world_model_ensemble,
            )
        ):
            raise ValueError(
                "adapter requires Prototype world-model lanes to remain disabled"
            )
        if self.ensemble.router.base_dim != feature.base_feature_dim:
            raise ValueError("ensemble router base_dim must match the feature lifecycle")
        if self.ensemble.router.active_slots != feature.active_pair_slots:
            raise ValueError(
                "ensemble router active_slots must match the feature lifecycle"
            )
        if self.ensemble.world_model.n_actions != (
            feature.n_primitive_actions
        ):
            raise ValueError(
                "ensemble primitive actions must match the feature lifecycle"
            )
        if self.ensemble.carry_survivors != feature.carry_survivors:
            raise ValueError(
                "ensemble survivor routing must match the feature lifecycle"
            )

    def to_config(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "schema": (
                PROTOTYPE_ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_ADAPTER_CONFIG_SCHEMA
            ),
            "state_schema": (
                PROTOTYPE_ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_ADAPTER_STATE_SCHEMA
            ),
            "receipt_schema": (
                PROTOTYPE_ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_ADAPTER_RECEIPT_SCHEMA
            ),
            "evidence_level": (
                PROTOTYPE_ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_ADAPTER_EVIDENCE_LEVEL
            ),
            "outcome_status": (
                PROTOTYPE_ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_ADAPTER_OUTCOME_STATUS
            ),
            "scientific_promotion_allowed": False,
            "authority_semantics": _AUTHORITY_SEMANTICS,
            "mapping_semantics": _MAPPING_SEMANTICS,
            "prototype": self.prototype.to_config(),
            "ensemble": self.ensemble.to_config(),
            "prototype_v18_required": False,
            "prototype_v18_allowed": False,
            "curation_recomputed": False,
            "feature_lifecycle_authority_count": 1,
            "feature_router_authority_count": 1,
            "external_ensemble_router_state_owned": False,
            "planning_authority": False,
            "dispatch_authority": False,
            "safety_authority": False,
            "evidence_authority": False,
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> PrototypeRoutedLinearWorldModelEnsembleAdapterConfig:
        expected = {
            "type",
            "schema",
            "state_schema",
            "receipt_schema",
            "evidence_level",
            "outcome_status",
            "scientific_promotion_allowed",
            "authority_semantics",
            "mapping_semantics",
            "prototype",
            "ensemble",
            "prototype_v18_required",
            "prototype_v18_allowed",
            "curation_recomputed",
            "feature_lifecycle_authority_count",
            "feature_router_authority_count",
            "external_ensemble_router_state_owned",
            "planning_authority",
            "dispatch_authority",
            "safety_authority",
            "evidence_authority",
        }
        if type(config) is not dict or set(config) != expected:
            raise ValueError("adapter config fields are not exact")
        fixed = {
            "type": cls.__name__,
            "schema": (
                PROTOTYPE_ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_ADAPTER_CONFIG_SCHEMA
            ),
            "state_schema": (
                PROTOTYPE_ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_ADAPTER_STATE_SCHEMA
            ),
            "receipt_schema": (
                PROTOTYPE_ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_ADAPTER_RECEIPT_SCHEMA
            ),
            "evidence_level": (
                PROTOTYPE_ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_ADAPTER_EVIDENCE_LEVEL
            ),
            "outcome_status": (
                PROTOTYPE_ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_ADAPTER_OUTCOME_STATUS
            ),
            "scientific_promotion_allowed": False,
            "authority_semantics": _AUTHORITY_SEMANTICS,
            "mapping_semantics": _MAPPING_SEMANTICS,
            "prototype_v18_required": False,
            "prototype_v18_allowed": False,
            "curation_recomputed": False,
            "feature_lifecycle_authority_count": 1,
            "feature_router_authority_count": 1,
            "external_ensemble_router_state_owned": False,
            "planning_authority": False,
            "dispatch_authority": False,
            "safety_authority": False,
            "evidence_authority": False,
        }
        if any(config.get(name) != value for name, value in fixed.items()):
            raise ValueError("adapter fixed semantics differ")
        if type(config["prototype"]) is not dict or type(config["ensemble"]) is not dict:
            raise ValueError("adapter nested configs must be exact dicts")
        restored = cls(
            prototype=PrototypeAgentConfig.from_config(
                cast(dict[str, Any], config["prototype"])
            ),
            ensemble=RoutedLinearWorldModelEnsembleConfig.from_config(
                cast(dict[str, object], config["ensemble"])
            ),
        )
        if restored.to_config() != dict(config):
            raise ValueError("adapter config is not canonical")
        return restored


@chex.dataclass(frozen=True)
class PrototypeRoutedLinearWorldModelEnsembleAdapterState:
    """One Prototype owner plus a binding-only external ensemble."""

    prototype_state: PrototypeAgentState
    ensemble_state: RoutedLinearWorldModelEnsembleState
    schema_digest: Array


@chex.dataclass(frozen=True)
class PrototypeRoutedLinearWorldModelEnsembleAdapterPreparedTransition:
    """One old-bank prediction and one already-evaluated Prototype result."""

    source_state: PrototypeRoutedLinearWorldModelEnsembleAdapterState
    transition: PrototypeTransition
    prototype_result: PrototypeUpdateResult
    ensemble_prepared: RoutedLinearWorldModelEnsemblePreparedTransition
    preparation_valid: Array
    prototype_update_evaluations: Array
    prototype_lifecycle_router_evaluations: Array
    ensemble_member_prediction_evaluations: Array
    curation_recomputations: Array


@chex.dataclass(frozen=True)
class PrototypeRoutedLinearWorldModelEnsembleAdapterIntegrityReceipt:
    """Unkeyed exact-content binding; integrity-bound, not authenticated."""

    prepared: PrototypeRoutedLinearWorldModelEnsembleAdapterPreparedTransition
    integrity_bound: Array


@chex.dataclass(frozen=True)
class PrototypeRoutedLinearWorldModelEnsembleAdapterDiagnostics:
    """Ownership, identity, exact-work, and all-or-nothing adoption facts."""

    source_state_matches: Array
    receipt_matches_preparation: Array
    receipt_integrity_bound: Array
    source_binding_integrity_valid: Array
    destination_binding_integrity_valid: Array
    prototype_preparation_valid: Array
    prototype_transition_applied: Array
    ensemble_transition_applied: Array
    candidate_state_valid: Array
    descriptors_changed: Array
    destination_adopted: Array
    complete_source_returned: Array
    transaction_applied: Array
    rejected: Array
    prototype_update_evaluations: Array
    prototype_lifecycle_router_evaluations: Array
    ensemble_member_prediction_evaluations: Array
    ensemble_member_update_evaluations: Array
    ensemble_router_evaluations: Array
    total_bank_mapping_evaluations: Array
    curation_recomputations: Array
    feature_lifecycle_authority_count: Array
    feature_router_authority_count: Array
    external_ensemble_router_state_owned: Array
    planning_authority: Array
    dispatch_authority: Array
    safety_authority: Array
    evidence_authority: Array


@chex.dataclass(frozen=True)
class PrototypeRoutedLinearWorldModelEnsembleAdapterResult:
    """Selected composite state plus unchanged attempted component sidecars."""

    state: PrototypeRoutedLinearWorldModelEnsembleAdapterState
    prototype_result: PrototypeUpdateResult
    ensemble_result: RoutedLinearWorldModelEnsembleResult
    receipt: PrototypeRoutedLinearWorldModelEnsembleAdapterIntegrityReceipt
    diagnostics: PrototypeRoutedLinearWorldModelEnsembleAdapterDiagnostics


@chex.dataclass(frozen=True)
class PrototypeRoutedLinearWorldModelEnsembleAdapterArrayResult:
    """Narrow scan surface; full per-event audits remain on ``step``."""

    state: PrototypeRoutedLinearWorldModelEnsembleAdapterState
    actions: Array
    transaction_applied: Array
    descriptors_changed: Array
    observed_loss: Array


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeRoutedLinearWorldModelEnsembleAdapterResourceBudget:
    """Exact persistent bytes and fixed complete-event evaluation counts."""

    persistent_state_bytes: int
    prototype_state_bytes: int
    ensemble_state_bytes: int
    adapter_schema_digest_bytes: int
    persistent_capacity_growth: int
    feature_lifecycle_authority_count: int
    feature_router_authority_count: int
    external_ensemble_router_state_owned: int
    managed_linear_horde_count: int
    feature_bound_memory_count: int
    prototype_update_evaluations_per_event: int
    prototype_lifecycle_router_evaluations_per_event: int
    ensemble_member_prediction_evaluations_per_event: int
    ensemble_member_update_evaluations_per_event: int
    ensemble_router_evaluations_per_event: int
    total_bank_mapping_evaluations_per_event: int
    curation_recomputations_per_event: int
    memory_rebind_evaluations_per_event: int
    planning_authority: int
    dispatch_authority: int
    safety_authority: int
    evidence_authority: int
    scientific_promotion_allowed: bool

    def to_config(self) -> dict[str, int | bool]:
        return dataclasses.asdict(self)


class PrototypeRoutedLinearWorldModelEnsembleAdapter:
    """Compose one Prototype update with one source-bound ensemble update."""

    def __init__(
        self,
        config: PrototypeRoutedLinearWorldModelEnsembleAdapterConfig,
    ) -> None:
        if type(config) is not PrototypeRoutedLinearWorldModelEnsembleAdapterConfig:
            raise TypeError(
                "config must be an exact "
                "PrototypeRoutedLinearWorldModelEnsembleAdapterConfig"
            )
        self._config = config
        self._prototype = PrototypeAgent(config.prototype)
        self._ensemble = RoutedLinearWorldModelEnsemble(config.ensemble)
        digest = hashlib.sha256(
            json.dumps(
                config.to_config(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).digest()
        self._schema_digest = jnp.asarray(tuple(digest), dtype=jnp.uint8)

    @property
    def config(self) -> PrototypeRoutedLinearWorldModelEnsembleAdapterConfig:
        return self._config

    @property
    def prototype(self) -> PrototypeAgent:
        return self._prototype

    @property
    def ensemble(self) -> RoutedLinearWorldModelEnsemble:
        return self._ensemble

    def to_config(self) -> dict[str, object]:
        return self._config.to_config()

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> PrototypeRoutedLinearWorldModelEnsembleAdapter:
        return cls(
            PrototypeRoutedLinearWorldModelEnsembleAdapterConfig.from_config(
                config
            )
        )

    def _bank(
        self,
        state: PrototypeAgentState,
    ) -> tuple[PrototypeFeatureLifecycleState, PrototypeFeatureConsumerBinding]:
        lifecycle = self._prototype._feature_lifecycle_component_state(
            state.state_builder_state
        )
        binding = self._prototype._feature_consumer_binding(state.oak_state)
        return lifecycle, binding

    def _prototype_state_valid(self, state: PrototypeAgentState) -> Array:
        return self._prototype._checkpoint_state_valid(state)

    def _bank_integrity_valid(self, state: PrototypeAgentState) -> Array:
        lifecycle, binding = self._bank(state)
        return self._ensemble.router_state_matches_binding(
            lifecycle.router_state,
            binding,
        )

    def state_valid(
        self,
        state: PrototypeRoutedLinearWorldModelEnsembleAdapterState,
    ) -> Array:
        if type(state) is not PrototypeRoutedLinearWorldModelEnsembleAdapterState:
            raise TypeError(
                "state must be an exact "
                "PrototypeRoutedLinearWorldModelEnsembleAdapterState"
            )
        if (
            not hasattr(state.schema_digest, "shape")
            or not hasattr(state.schema_digest, "dtype")
            or state.schema_digest.shape != (_SCHEMA_DIGEST_NBYTES,)
            or state.schema_digest.dtype != jnp.uint8
        ):
            raise ValueError("adapter schema_digest has the wrong contract")
        lifecycle, binding = self._bank(state.prototype_state)
        return (
            jnp.array_equal(state.schema_digest, self._schema_digest)
            & self._prototype_state_valid(state.prototype_state)
            & self._ensemble.state_valid(state.ensemble_state)
            & self._ensemble.router_state_matches_binding(
                lifecycle.router_state,
                binding,
            )
            & self._ensemble.router_state_matches_binding(
                lifecycle.router_state,
                state.ensemble_state.consumer_binding,
            )
            & _tree_equal(binding, state.ensemble_state.consumer_binding)
        )

    def init(
        self,
        key: Array,
        *,
        lifecycle_id: Array | None = None,
    ) -> PrototypeRoutedLinearWorldModelEnsembleAdapterState:
        if not (
            hasattr(key, "shape")
            and hasattr(key, "dtype")
            and key.shape == ()
            and jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key)
        ):
            raise TypeError("key must be a scalar typed JAX PRNG key")
        prototype_key, ensemble_key = jr.split(key)
        prototype_state = self._prototype.init(
            prototype_key,
            lifecycle_id=lifecycle_id,
        )
        lifecycle, binding = self._bank(prototype_state)
        ensemble_state = self._ensemble.init(
            ensemble_key,
            binding,
            lifecycle.router_state,
        )
        state = PrototypeRoutedLinearWorldModelEnsembleAdapterState(
            prototype_state=prototype_state,
            ensemble_state=ensemble_state,
            schema_digest=self._schema_digest,
        )
        if not bool(jax.device_get(self.state_valid(state))):
            raise ValueError("initial adapter composition is invalid")
        return state

    def start(
        self,
        state: PrototypeRoutedLinearWorldModelEnsembleAdapterState,
        initial_observation: Array,
        *,
        extended_action_mask: Array | None = None,
    ) -> PrototypeRoutedLinearWorldModelEnsembleAdapterState:
        self.state_valid(state)
        started = self._prototype.start(
            state.prototype_state,
            initial_observation,
            extended_action_mask=extended_action_mask,
        )
        candidate = PrototypeRoutedLinearWorldModelEnsembleAdapterState(
            prototype_state=started,
            ensemble_state=state.ensemble_state,
            schema_digest=state.schema_digest,
        )
        valid = self.state_valid(candidate)
        if not any(
            isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(valid)
        ) and not bool(jax.device_get(valid)):
            raise ValueError("starting Prototype changed the bound bank identity")
        return cast(
            PrototypeRoutedLinearWorldModelEnsembleAdapterState,
            jax.lax.cond(valid, lambda: candidate, lambda: state),
        )

    def _prototype_result_integrity(
        self,
        source: PrototypeAgentState,
        result: PrototypeUpdateResult,
    ) -> Array:
        integration = result.prototype_feature_lifecycle_diagnostics
        if integration is None:
            raise RuntimeError(
                "configured Prototype lifecycle did not return its diagnostics"
            )
        source_lifecycle, source_binding = self._bank(source)
        destination_lifecycle, destination_binding = self._bank(result.state)
        source_router = source_lifecycle.router_state
        destination_router = destination_lifecycle.router_state
        descriptors_changed = jnp.any(
            source_router.descriptors != destination_router.descriptors
        )
        generation_changed = jnp.any(
            source_router.generation_words != destination_router.generation_words
        )
        route_changed = jnp.any(
            source_router.route_words != destination_router.route_words
        )
        changed_transition = (
            descriptors_changed
            & generation_changed
            & route_changed
            & _words_successor(
                source_router.generation_words,
                destination_router.generation_words,
            )
            & _words_successor(
                source_router.route_words,
                destination_router.route_words,
            )
        )
        unchanged_transition = (
            ~descriptors_changed
            & ~generation_changed
            & ~route_changed
            & _tree_equal(source_router, destination_router)
        )
        lifecycle = integration.lifecycle
        return (
            integration.available
            & integration.outer_transaction_committed
            & lifecycle.transaction_applied
            & result.transition_diagnostics.valid
            & self._prototype_state_valid(source)
            & self._prototype_state_valid(result.state)
            & self._ensemble.router_state_matches_binding(
                source_router,
                source_binding,
            )
            & self._ensemble.router_state_matches_binding(
                destination_router,
                destination_binding,
            )
            & (changed_transition | unchanged_transition)
            & (lifecycle.semantic_generation_before == source_router.generation_count)
            & (lifecycle.semantic_generation_after == destination_router.generation_count)
            & jnp.array_equal(
                lifecycle.semantic_generation_words_before,
                source_router.generation_words,
            )
            & jnp.array_equal(
                lifecycle.semantic_generation_words_after,
                destination_router.generation_words,
            )
        )

    def prepare_transition(
        self,
        state: PrototypeRoutedLinearWorldModelEnsembleAdapterState,
        transition: PrototypeTransition,
        candidate_update_audit_evidence: (
            PrototypeCandidateUpdateAuditEvidence | None
        ) = None,
        *,
        gradient_joy_evidence: PrototypeGradientJoyEvidence | None = None,
        experiential_memory_input: PrototypeExperientialMemoryInput | None = None,
        partner_policy_fusion_input: PrototypePartnerPolicyFusionInput | None = None,
        partner_policy_fusion_feedback: (
            PrototypePartnerPolicyFusionFeedback | None
        ) = None,
        extended_action_mask: Array | None = None,
    ) -> PrototypeRoutedLinearWorldModelEnsembleAdapterPreparedTransition:
        """Predict on the source bank, then evaluate Prototype exactly once."""

        if type(state) is not PrototypeRoutedLinearWorldModelEnsembleAdapterState:
            raise TypeError("state must be an exact adapter state")
        if type(transition) is not PrototypeTransition:
            raise TypeError("transition must be an exact PrototypeTransition")
        source_lifecycle, _ = self._bank(state.prototype_state)
        ensemble_prepare = self._ensemble.prepare_transition(
            state.ensemble_state,
            source_lifecycle.router_state,
            state.prototype_state.current_raw_observation,
            transition.action,
        )
        prototype_result = self._prototype.update_transition(
            state.prototype_state,
            transition,
            candidate_update_audit_evidence,
            gradient_joy_evidence=gradient_joy_evidence,
            experiential_memory_input=experiential_memory_input,
            partner_policy_fusion_input=partner_policy_fusion_input,
            partner_policy_fusion_feedback=partner_policy_fusion_feedback,
            extended_action_mask=extended_action_mask,
        )
        integration = prototype_result.prototype_feature_lifecycle_diagnostics
        if integration is None:
            raise RuntimeError(
                "configured Prototype lifecycle did not return diagnostics"
            )
        prototype_integrity = self._prototype_result_integrity(
            state.prototype_state,
            prototype_result,
        )
        state_is_valid = self.state_valid(state)
        preparation_valid = (
            state_is_valid
            & ensemble_prepare.diagnostics.prepared
            & prototype_integrity
        )
        return PrototypeRoutedLinearWorldModelEnsembleAdapterPreparedTransition(
            source_state=state,
            transition=transition,
            prototype_result=prototype_result,
            ensemble_prepared=ensemble_prepare.prepared,
            preparation_valid=preparation_valid,
            prototype_update_evaluations=jnp.asarray(1, dtype=jnp.int32),
            prototype_lifecycle_router_evaluations=jnp.where(
                integration.available,
                jnp.asarray(2, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
            ),
            ensemble_member_prediction_evaluations=jnp.asarray(
                self._config.ensemble.ensemble_size,
                dtype=jnp.int32,
            ),
            curation_recomputations=jnp.asarray(0, dtype=jnp.int32),
        )

    def integrity_receipt(
        self,
        prepared: PrototypeRoutedLinearWorldModelEnsembleAdapterPreparedTransition,
    ) -> PrototypeRoutedLinearWorldModelEnsembleAdapterIntegrityReceipt:
        """Bind exact preparation content without claiming authentication."""

        if type(prepared) is not (
            PrototypeRoutedLinearWorldModelEnsembleAdapterPreparedTransition
        ):
            raise TypeError("prepared must be an exact adapter preparation")
        return PrototypeRoutedLinearWorldModelEnsembleAdapterIntegrityReceipt(
            prepared=prepared,
            integrity_bound=jnp.asarray(True, dtype=jnp.bool_),
        )

    def adopt_prepared_transition(
        self,
        state: PrototypeRoutedLinearWorldModelEnsembleAdapterState,
        prepared: PrototypeRoutedLinearWorldModelEnsembleAdapterPreparedTransition,
        receipt: PrototypeRoutedLinearWorldModelEnsembleAdapterIntegrityReceipt,
    ) -> PrototypeRoutedLinearWorldModelEnsembleAdapterResult:
        """Adopt both destinations or return the complete composite source."""

        if type(state) is not PrototypeRoutedLinearWorldModelEnsembleAdapterState:
            raise TypeError("state must be an exact adapter state")
        if type(prepared) is not (
            PrototypeRoutedLinearWorldModelEnsembleAdapterPreparedTransition
        ):
            raise TypeError("prepared must be an exact adapter preparation")
        if type(receipt) is not (
            PrototypeRoutedLinearWorldModelEnsembleAdapterIntegrityReceipt
        ):
            raise TypeError("receipt must be an exact adapter integrity receipt")
        source_lifecycle, source_binding = self._bank(state.prototype_state)
        destination_lifecycle, destination_binding = self._bank(
            prepared.prototype_result.state
        )
        source_state_matches = _tree_equal(state, prepared.source_state)
        receipt_matches = _tree_equal(receipt.prepared, prepared)
        source_binding_valid = (
            self._ensemble.router_state_matches_binding(
                source_lifecycle.router_state,
                source_binding,
            )
            & _tree_equal(source_binding, state.ensemble_state.consumer_binding)
        )
        destination_binding_valid = self._ensemble.router_state_matches_binding(
            destination_lifecycle.router_state,
            destination_binding,
        )
        prototype_integrity = self._prototype_result_integrity(
            state.prototype_state,
            prepared.prototype_result,
        )
        event = RoutedLinearWorldModelEnsembleTransition(
            prepared=prepared.ensemble_prepared,
            reward=prepared.transition.reward,
            discount=prepared.transition.discount,
            next_base_observation=prepared.transition.next_observation,
            destination_router_state=destination_lifecycle.router_state,
            destination_binding=destination_binding,
        )
        ensemble_result = self._ensemble.observe_and_route(
            state.ensemble_state,
            source_lifecycle.router_state,
            event,
        )
        candidate = PrototypeRoutedLinearWorldModelEnsembleAdapterState(
            prototype_state=prepared.prototype_result.state,
            ensemble_state=ensemble_result.state,
            schema_digest=state.schema_digest,
        )
        candidate_valid = self.state_valid(candidate)
        commit = (
            self.state_valid(state)
            & source_state_matches
            & receipt_matches
            & receipt.integrity_bound
            & prepared.preparation_valid
            & prototype_integrity
            & source_binding_valid
            & destination_binding_valid
            & ensemble_result.diagnostics.transaction_applied
            & candidate_valid
        )
        selected_state = cast(
            PrototypeRoutedLinearWorldModelEnsembleAdapterState,
            jax.lax.cond(commit, lambda: candidate, lambda: state),
        )
        selected_prototype_state = cast(
            PrototypeAgentState,
            jax.lax.cond(
                commit,
                lambda: prepared.prototype_result.state,
                lambda: state.prototype_state,
            ),
        )
        selected_ensemble_state = cast(
            RoutedLinearWorldModelEnsembleState,
            jax.lax.cond(
                commit,
                lambda: ensemble_result.state,
                lambda: state.ensemble_state,
            ),
        )
        selected_prototype_result = cast(
            PrototypeUpdateResult,
            prepared.prototype_result.replace(state=selected_prototype_state),
        )
        selected_ensemble_result = cast(
            RoutedLinearWorldModelEnsembleResult,
            ensemble_result.replace(state=selected_ensemble_state),
        )
        descriptors_changed = jnp.any(
            source_binding.descriptors != destination_binding.descriptors
        )
        total_mappings = (
            prepared.prototype_lifecycle_router_evaluations
            + ensemble_result.diagnostics.router_evaluations
        )
        diagnostics = PrototypeRoutedLinearWorldModelEnsembleAdapterDiagnostics(
            source_state_matches=source_state_matches,
            receipt_matches_preparation=receipt_matches,
            receipt_integrity_bound=receipt.integrity_bound,
            source_binding_integrity_valid=source_binding_valid,
            destination_binding_integrity_valid=destination_binding_valid,
            prototype_preparation_valid=prepared.preparation_valid,
            prototype_transition_applied=prototype_integrity,
            ensemble_transition_applied=(
                ensemble_result.diagnostics.transaction_applied
            ),
            candidate_state_valid=candidate_valid,
            descriptors_changed=descriptors_changed,
            destination_adopted=commit,
            complete_source_returned=~commit,
            transaction_applied=commit,
            rejected=~commit,
            prototype_update_evaluations=prepared.prototype_update_evaluations,
            prototype_lifecycle_router_evaluations=(
                prepared.prototype_lifecycle_router_evaluations
            ),
            ensemble_member_prediction_evaluations=(
                prepared.ensemble_member_prediction_evaluations
            ),
            ensemble_member_update_evaluations=(
                ensemble_result.diagnostics.member_update_evaluations
            ),
            ensemble_router_evaluations=(
                ensemble_result.diagnostics.router_evaluations
            ),
            total_bank_mapping_evaluations=total_mappings,
            curation_recomputations=prepared.curation_recomputations,
            feature_lifecycle_authority_count=jnp.asarray(1, dtype=jnp.int32),
            feature_router_authority_count=jnp.asarray(1, dtype=jnp.int32),
            external_ensemble_router_state_owned=jnp.asarray(
                0, dtype=jnp.int32
            ),
            planning_authority=jnp.asarray(False, dtype=jnp.bool_),
            dispatch_authority=jnp.asarray(False, dtype=jnp.bool_),
            safety_authority=jnp.asarray(False, dtype=jnp.bool_),
            evidence_authority=jnp.asarray(False, dtype=jnp.bool_),
        )
        return PrototypeRoutedLinearWorldModelEnsembleAdapterResult(
            state=selected_state,
            prototype_result=selected_prototype_result,
            ensemble_result=selected_ensemble_result,
            receipt=receipt,
            diagnostics=diagnostics,
        )

    def step(
        self,
        state: PrototypeRoutedLinearWorldModelEnsembleAdapterState,
        transition: PrototypeTransition,
        candidate_update_audit_evidence: (
            PrototypeCandidateUpdateAuditEvidence | None
        ) = None,
        *,
        gradient_joy_evidence: PrototypeGradientJoyEvidence | None = None,
        experiential_memory_input: PrototypeExperientialMemoryInput | None = None,
        partner_policy_fusion_input: PrototypePartnerPolicyFusionInput | None = None,
        partner_policy_fusion_feedback: (
            PrototypePartnerPolicyFusionFeedback | None
        ) = None,
        extended_action_mask: Array | None = None,
    ) -> PrototypeRoutedLinearWorldModelEnsembleAdapterResult:
        """Prepare, integrity-bind, and adopt one complete transition."""

        prepared = self.prepare_transition(
            state,
            transition,
            candidate_update_audit_evidence,
            gradient_joy_evidence=gradient_joy_evidence,
            experiential_memory_input=experiential_memory_input,
            partner_policy_fusion_input=partner_policy_fusion_input,
            partner_policy_fusion_feedback=partner_policy_fusion_feedback,
            extended_action_mask=extended_action_mask,
        )
        receipt = self.integrity_receipt(prepared)
        return self.adopt_prepared_transition(state, prepared, receipt)

    def scan_transitions(
        self,
        state: PrototypeRoutedLinearWorldModelEnsembleAdapterState,
        transitions: PrototypeTransition,
    ) -> PrototypeRoutedLinearWorldModelEnsembleAdapterArrayResult:
        """Scan the fixed no-sidecar boundary over explicit transitions."""

        def scan_step(
            carry: PrototypeRoutedLinearWorldModelEnsembleAdapterState,
            transition: PrototypeTransition,
        ) -> tuple[
            PrototypeRoutedLinearWorldModelEnsembleAdapterState,
            tuple[Array, Array, Array, Array],
        ]:
            result = self.step(carry, transition)
            return result.state, (
                result.prototype_result.action,
                result.diagnostics.transaction_applied,
                result.diagnostics.descriptors_changed,
                result.ensemble_result.observed_loss,
            )

        final_state, outputs = jax.lax.scan(scan_step, state, transitions)
        actions, applied, changed, loss = outputs
        return PrototypeRoutedLinearWorldModelEnsembleAdapterArrayResult(
            state=final_state,
            actions=actions,
            transaction_applied=applied,
            descriptors_changed=changed,
            observed_loss=loss,
        )

    @property
    def resource_budget(
        self,
    ) -> PrototypeRoutedLinearWorldModelEnsembleAdapterResourceBudget:
        """Declare one owner and the honest two-plus-one mapping work."""

        state = self.init(jr.key(0))
        prototype_bytes = measure_prototype_agent_state_resources(
            state.prototype_state
        ).total_nbytes
        ensemble_bytes = measure_routed_linear_world_model_ensemble_state_nbytes(
            state.ensemble_state
        )
        total_bytes = _tree_nbytes(state)
        memory_enabled = self._config.prototype.experiential_memory is not None
        return PrototypeRoutedLinearWorldModelEnsembleAdapterResourceBudget(
            persistent_state_bytes=total_bytes,
            prototype_state_bytes=prototype_bytes,
            ensemble_state_bytes=ensemble_bytes,
            adapter_schema_digest_bytes=_SCHEMA_DIGEST_NBYTES,
            persistent_capacity_growth=0,
            feature_lifecycle_authority_count=1,
            feature_router_authority_count=1,
            external_ensemble_router_state_owned=0,
            managed_linear_horde_count=1,
            feature_bound_memory_count=int(memory_enabled),
            prototype_update_evaluations_per_event=1,
            prototype_lifecycle_router_evaluations_per_event=2,
            ensemble_member_prediction_evaluations_per_event=(
                self._config.ensemble.ensemble_size
            ),
            ensemble_member_update_evaluations_per_event=(
                self._config.ensemble.ensemble_size
            ),
            ensemble_router_evaluations_per_event=1,
            total_bank_mapping_evaluations_per_event=3,
            curation_recomputations_per_event=0,
            memory_rebind_evaluations_per_event=int(memory_enabled),
            planning_authority=0,
            dispatch_authority=0,
            safety_authority=0,
            evidence_authority=0,
            scientific_promotion_allowed=False,
        )


def measure_prototype_routed_linear_world_model_ensemble_adapter_state_nbytes(
    state: PrototypeRoutedLinearWorldModelEnsembleAdapterState,
) -> int:
    """Measure every persistent array leaf in one adapter state."""

    if type(state) is not PrototypeRoutedLinearWorldModelEnsembleAdapterState:
        raise TypeError("state must be an exact adapter state")
    return _tree_nbytes(state)


def save_prototype_routed_linear_world_model_ensemble_adapter_checkpoint(
    owner: PrototypeRoutedLinearWorldModelEnsembleAdapter,
    state: PrototypeRoutedLinearWorldModelEnsembleAdapterState,
    path: str | Path,
) -> None:
    """Persist the one Prototype owner and the binding-only ensemble state."""

    if type(owner) is not PrototypeRoutedLinearWorldModelEnsembleAdapter:
        raise TypeError("owner must be an exact adapter")
    if not bool(jax.device_get(owner.state_valid(state))):
        raise ValueError("refusing to save an invalid adapter state")
    config = owner.to_config()
    save_checkpoint(
        state,
        path,
        metadata={
            "schema": (
                PROTOTYPE_ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_ADAPTER_CHECKPOINT_SCHEMA
            ),
            "owner_config": config,
            "config_sha256": _config_digest(config),
            "resource_budget": owner.resource_budget.to_config(),
            "evidence_level": (
                PROTOTYPE_ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_ADAPTER_EVIDENCE_LEVEL
            ),
            "outcome_status": (
                PROTOTYPE_ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_ADAPTER_OUTCOME_STATUS
            ),
            "scientific_promotion_allowed": False,
            "feature_lifecycle_authority_count": 1,
            "feature_router_authority_count": 1,
            "external_ensemble_router_state_included": False,
            "transient_receipt_included": False,
            "planning_authority": False,
            "dispatch_authority": False,
            "safety_authority": False,
            "evidence_authority": False,
        },
    )


def load_prototype_routed_linear_world_model_ensemble_adapter_checkpoint(
    path: str | Path,
) -> tuple[
    PrototypeRoutedLinearWorldModelEnsembleAdapter,
    PrototypeRoutedLinearWorldModelEnsembleAdapterState,
]:
    """Strictly restore the sole current adapter v1 schema."""

    metadata = load_checkpoint_metadata(path)
    expected = {
        "schema",
        "owner_config",
        "config_sha256",
        "resource_budget",
        "evidence_level",
        "outcome_status",
        "scientific_promotion_allowed",
        "feature_lifecycle_authority_count",
        "feature_router_authority_count",
        "external_ensemble_router_state_included",
        "transient_receipt_included",
        "planning_authority",
        "dispatch_authority",
        "safety_authority",
        "evidence_authority",
    }
    if set(metadata) != expected:
        raise ValueError("adapter checkpoint metadata fields are not exact")
    if metadata.get("schema") != (
        PROTOTYPE_ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_ADAPTER_CHECKPOINT_SCHEMA
    ):
        raise ValueError("checkpoint is not an adapter v1 checkpoint")
    config = metadata.get("owner_config")
    if type(config) is not dict:
        raise ValueError("adapter checkpoint lacks exact owner_config")
    if metadata.get("config_sha256") != _config_digest(config):
        raise ValueError("adapter checkpoint config digest does not match")
    owner = PrototypeRoutedLinearWorldModelEnsembleAdapter.from_config(config)
    if metadata.get("resource_budget") != owner.resource_budget.to_config():
        raise ValueError("adapter checkpoint resource budget does not match")
    fixed = {
        "evidence_level": (
            PROTOTYPE_ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_ADAPTER_EVIDENCE_LEVEL
        ),
        "outcome_status": (
            PROTOTYPE_ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_ADAPTER_OUTCOME_STATUS
        ),
        "scientific_promotion_allowed": False,
        "feature_lifecycle_authority_count": 1,
        "feature_router_authority_count": 1,
        "external_ensemble_router_state_included": False,
        "transient_receipt_included": False,
        "planning_authority": False,
        "dispatch_authority": False,
        "safety_authority": False,
        "evidence_authority": False,
    }
    if any(metadata.get(name) != value for name, value in fixed.items()):
        raise ValueError("adapter checkpoint fixed semantics differ")
    template = owner.init(jr.key(0))
    restored, second_metadata = load_checkpoint(template, path)
    if second_metadata != metadata:
        raise ValueError("adapter checkpoint metadata changed between reads")
    state = cast(
        PrototypeRoutedLinearWorldModelEnsembleAdapterState,
        restored,
    )
    if not bool(jax.device_get(owner.state_valid(state))):
        raise ValueError("adapter checkpoint restored an invalid state")
    if measure_prototype_routed_linear_world_model_ensemble_adapter_state_nbytes(
        state
    ) != owner.resource_budget.persistent_state_bytes:
        raise ValueError("adapter checkpoint restored a wrong-size state")
    return owner, state


__all__ = [
    "PROTOTYPE_ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_ADAPTER_CHECKPOINT_SCHEMA",
    "PROTOTYPE_ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_ADAPTER_CONFIG_SCHEMA",
    "PROTOTYPE_ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_ADAPTER_EVIDENCE_LEVEL",
    "PROTOTYPE_ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_ADAPTER_OUTCOME_STATUS",
    "PROTOTYPE_ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_ADAPTER_RECEIPT_SCHEMA",
    "PROTOTYPE_ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_ADAPTER_SCIENTIFIC_PROMOTION_ALLOWED",
    "PROTOTYPE_ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_ADAPTER_STATE_SCHEMA",
    "PrototypeRoutedLinearWorldModelEnsembleAdapter",
    "PrototypeRoutedLinearWorldModelEnsembleAdapterArrayResult",
    "PrototypeRoutedLinearWorldModelEnsembleAdapterConfig",
    "PrototypeRoutedLinearWorldModelEnsembleAdapterDiagnostics",
    "PrototypeRoutedLinearWorldModelEnsembleAdapterIntegrityReceipt",
    "PrototypeRoutedLinearWorldModelEnsembleAdapterPreparedTransition",
    "PrototypeRoutedLinearWorldModelEnsembleAdapterResourceBudget",
    "PrototypeRoutedLinearWorldModelEnsembleAdapterResult",
    "PrototypeRoutedLinearWorldModelEnsembleAdapterState",
    "load_prototype_routed_linear_world_model_ensemble_adapter_checkpoint",
    "measure_prototype_routed_linear_world_model_ensemble_adapter_state_nbytes",
    "save_prototype_routed_linear_world_model_ensemble_adapter_checkpoint",
]
