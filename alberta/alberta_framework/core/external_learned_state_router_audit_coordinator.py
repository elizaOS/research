# mypy: disable-error-code="attr-defined,call-arg,no-any-return"
"""Single-owner external learned-state/router/audit coordination.

This L0 coordinator puts one exact full-GRU state builder outside one
exact-Identity Prototype routed-ensemble adapter.  The external builder alone
maps raw observations to the stable base consumed by the inner Prototype.  The
inner Prototype remains the sole STOMP, OaK, managed-Horde, feature-lifecycle,
feature-router, optional feature-bound-memory, and routed-model owner.  One
separate :class:`LearningValueRouter` and one candidate-update audit may update
only the external builder's parameters for the next event.

The coordinator never accepts a caller-supplied learning target or
representation gradient.  It forms stopped physical targets from the real
transition through the routed ensemble, then analytically pulls the cached
pre-update member residuals through the source linear weights and exact pair
descriptors.  This adds no model forward evaluation.  External probe gradients
remain caller evidence under the existing explicit independence attestation;
they audit the internally formed candidate and are not target authority.

Version 1 is deliberately continuing-only.  A terminal, truncation, autoreset,
or distinct next-decision observation fails closed instead of evaluating the
full-GRU owner twice.  Preparation evaluates each state owner once, audit
evaluation routes one learning-value event and assesses one candidate, and an
unkeyed exact-content receipt atomically selects the complete candidate or the
complete source.  The receipt is source-bound and integrity-bound, not
authenticated.  A direct ``step`` has a bounded JIT contract; batched scan is
an explicit host loop because monolithic compilation of the whole Prototype
composition is not resource-bounded.  No curation, planning, dispatch, safety,
evidence, or promotion authority is created, and Prototype v18 remains
forbidden.
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
from alberta_framework.core.delight import (
    CandidateUpdateAuditAssessment,
    CandidateUpdateAuditConfig,
    CandidateUpdateAuditEvidence,
    LearningValue,
    LearningValueAvailability,
    assess_candidate_update,
)
from alberta_framework.core.learning_value_router import (
    LearningValueRouter,
    LearningValueRouterConfig,
    LearningValueRouterResult,
    LearningValueRouterState,
)
from alberta_framework.core.normalizers import (
    _checked_lifetime_words_increment,
    _lifetime_counter_valid,
    _saturating_int32_counter_increment,
)
from alberta_framework.core.prototype_agent import (
    PrototypeExperientialMemoryInput,
    PrototypePartnerPolicyFusionFeedback,
    PrototypePartnerPolicyFusionInput,
    PrototypeTransition,
)
from alberta_framework.core.prototype_routed_linear_world_model_ensemble_adapter import (
    PrototypeRoutedLinearWorldModelEnsembleAdapter,
    PrototypeRoutedLinearWorldModelEnsembleAdapterConfig,
    PrototypeRoutedLinearWorldModelEnsembleAdapterResult,
    PrototypeRoutedLinearWorldModelEnsembleAdapterState,
    measure_prototype_routed_linear_world_model_ensemble_adapter_state_nbytes,
)
from alberta_framework.core.state_builder import (
    LearnableGRUStateBuilder,
    LearnableGRUStateBuilderConfig,
    LearnableGRUStateBuilderState,
    OnlineGatedStateBuilderTransitionResult,
    StateBuilderLearningDiagnostics,
    StateBuilderLearningProposal,
    replace_state_builder_learning_proposal_update,
)

EXTERNAL_LEARNED_STATE_ROUTER_AUDIT_COORDINATOR_CONFIG_SCHEMA = (
    "alberta.external-learned-state-router-audit-coordinator.config.v1"
)
EXTERNAL_LEARNED_STATE_ROUTER_AUDIT_COORDINATOR_STATE_SCHEMA = (
    "alberta.external-learned-state-router-audit-coordinator.state.v1"
)
EXTERNAL_LEARNED_STATE_ROUTER_AUDIT_COORDINATOR_RECEIPT_SCHEMA = (
    "alberta.external-learned-state-router-audit-coordinator.receipt.v1"
)
EXTERNAL_LEARNED_STATE_ROUTER_AUDIT_COORDINATOR_CHECKPOINT_SCHEMA = (
    "alberta.external-learned-state-router-audit-coordinator.checkpoint.v1"
)
EXTERNAL_LEARNED_STATE_ROUTER_AUDIT_COORDINATOR_EVIDENCE_LEVEL = "L0"
EXTERNAL_LEARNED_STATE_ROUTER_AUDIT_COORDINATOR_OUTCOME_STATUS = "not_assessed"
EXTERNAL_LEARNED_STATE_ROUTER_AUDIT_COORDINATOR_SCIENTIFIC_PROMOTION_ALLOWED = (
    False
)

_INT32_MAX = 2_147_483_647
_SCHEMA_DIGEST_NBYTES = 32
_AUTHORITY_SEMANTICS = (
    "one-external-full-gru-owner;one-inner-identity-prototype;"
    "one-feature-lifecycle-router-owner;one-learning-value-router;"
    "candidate-audit-trains-external-builder-only"
)
_EVENT_SEMANTICS = (
    "continuing-only;one-external-recurrence;one-inner-prototype-update;"
    "one-routed-ensemble-transaction;internal-stopped-target;"
    "analytic-source-linear-pullback;one-router;one-candidate-audit;"
    "atomic-outer-adoption"
)
_SCAN_EXECUTION = "host-loop-only"


def _config_digest(config: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _float32_equal(left: Array, right: Array) -> Array:
    return jnp.array_equal(
        jax.lax.bitcast_convert_type(left, jnp.uint32),
        jax.lax.bitcast_convert_type(right, jnp.uint32),
    )


def _tree_nbytes(tree: object) -> int:
    total = 0
    for leaf in jax.tree.leaves(tree):
        value = jnp.asarray(leaf)
        if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
            value = jr.key_data(value)
        total += int(value.nbytes)
    return total


@dataclasses.dataclass(frozen=True, slots=True)
class ExternalLearnedStateRouterAuditCoordinatorConfig:
    """Exact full-GRU outer owner and exact non-v18 inner composition."""

    builder: LearnableGRUStateBuilderConfig
    inner: PrototypeRoutedLinearWorldModelEnsembleAdapterConfig
    learning_value_router: LearningValueRouterConfig
    candidate_audit: CandidateUpdateAuditConfig
    max_events: int = _INT32_MAX

    def __post_init__(self) -> None:
        if type(self.builder) is not LearnableGRUStateBuilderConfig:
            raise TypeError("builder must be an exact LearnableGRUStateBuilderConfig")
        if type(self.inner) is not (
            PrototypeRoutedLinearWorldModelEnsembleAdapterConfig
        ):
            raise TypeError("inner must be an exact routed-ensemble adapter config")
        if type(self.learning_value_router) is not LearningValueRouterConfig:
            raise TypeError(
                "learning_value_router must be an exact LearningValueRouterConfig"
            )
        if type(self.candidate_audit) is not CandidateUpdateAuditConfig:
            raise TypeError("candidate_audit must be an exact CandidateUpdateAuditConfig")
        if not self.builder.include_raw_observation:
            raise ValueError(
                "coordinator v1 requires full-GRU include_raw_observation=True"
            )
        feature = self.inner.prototype.prototype_feature_lifecycle
        if feature is None:
            raise ValueError("inner Prototype feature lifecycle is required")
        if self.builder.feature_dim() != feature.base_feature_dim:
            raise ValueError(
                "full-GRU feature_dim must equal the inner stable base width"
            )
        if self.builder.n_actions != feature.n_primitive_actions:
            raise ValueError(
                "full-GRU n_actions must equal inner primitive-action count"
            )
        if self.inner.prototype.prototype_atomic_feature_world_memory is not None:
            raise ValueError("coordinator is incompatible with Prototype v18")
        if self.inner.prototype.learning_value_router is not None:
            raise ValueError(
                "inner Prototype cannot own a second LearningValueRouter"
            )
        if self.inner.prototype.gradient_joy is not None:
            raise ValueError("inner Prototype cannot own a second candidate audit")
        if self.candidate_audit.candidate_semantics != "update":
            raise ValueError("external builder candidate audit must use update semantics")
        if type(self.max_events) is not int or not 1 <= self.max_events <= _INT32_MAX:
            raise ValueError("max_events must be an exact integer in [1, INT32_MAX]")
        if self.max_events > self.learning_value_router.max_steps:
            raise ValueError("max_events must not exceed LearningValueRouter capacity")

    def to_config(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "schema": EXTERNAL_LEARNED_STATE_ROUTER_AUDIT_COORDINATOR_CONFIG_SCHEMA,
            "state_schema": (
                EXTERNAL_LEARNED_STATE_ROUTER_AUDIT_COORDINATOR_STATE_SCHEMA
            ),
            "receipt_schema": (
                EXTERNAL_LEARNED_STATE_ROUTER_AUDIT_COORDINATOR_RECEIPT_SCHEMA
            ),
            "evidence_level": (
                EXTERNAL_LEARNED_STATE_ROUTER_AUDIT_COORDINATOR_EVIDENCE_LEVEL
            ),
            "outcome_status": (
                EXTERNAL_LEARNED_STATE_ROUTER_AUDIT_COORDINATOR_OUTCOME_STATUS
            ),
            "scientific_promotion_allowed": False,
            "authority_semantics": _AUTHORITY_SEMANTICS,
            "event_semantics": _EVENT_SEMANTICS,
            "builder": self.builder.to_config(),
            "inner": self.inner.to_config(),
            "learning_value_router": self.learning_value_router.to_config(),
            "candidate_audit": self.candidate_audit.to_config(),
            "max_events": self.max_events,
            "caller_target_authority": False,
            "prototype_v18_allowed": False,
            "terminal_boundary_supported": False,
            "direct_step_jit_supported": True,
            "monolithic_scan_jit_supported": False,
            "scan_execution": _SCAN_EXECUTION,
            "feature_lifecycle_authority_count": 1,
            "feature_router_authority_count": 1,
            "learning_value_router_count": 1,
            "planning_authority": False,
            "dispatch_authority": False,
            "safety_authority": False,
            "evidence_authority": False,
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> ExternalLearnedStateRouterAuditCoordinatorConfig:
        expected = {
            "type",
            "schema",
            "state_schema",
            "receipt_schema",
            "evidence_level",
            "outcome_status",
            "scientific_promotion_allowed",
            "authority_semantics",
            "event_semantics",
            "builder",
            "inner",
            "learning_value_router",
            "candidate_audit",
            "max_events",
            "caller_target_authority",
            "prototype_v18_allowed",
            "terminal_boundary_supported",
            "direct_step_jit_supported",
            "monolithic_scan_jit_supported",
            "scan_execution",
            "feature_lifecycle_authority_count",
            "feature_router_authority_count",
            "learning_value_router_count",
            "planning_authority",
            "dispatch_authority",
            "safety_authority",
            "evidence_authority",
        }
        if type(config) is not dict or set(config) != expected:
            raise ValueError("coordinator config fields are not exact")
        fixed = {
            "type": cls.__name__,
            "schema": EXTERNAL_LEARNED_STATE_ROUTER_AUDIT_COORDINATOR_CONFIG_SCHEMA,
            "state_schema": (
                EXTERNAL_LEARNED_STATE_ROUTER_AUDIT_COORDINATOR_STATE_SCHEMA
            ),
            "receipt_schema": (
                EXTERNAL_LEARNED_STATE_ROUTER_AUDIT_COORDINATOR_RECEIPT_SCHEMA
            ),
            "evidence_level": (
                EXTERNAL_LEARNED_STATE_ROUTER_AUDIT_COORDINATOR_EVIDENCE_LEVEL
            ),
            "outcome_status": (
                EXTERNAL_LEARNED_STATE_ROUTER_AUDIT_COORDINATOR_OUTCOME_STATUS
            ),
            "scientific_promotion_allowed": False,
            "authority_semantics": _AUTHORITY_SEMANTICS,
            "event_semantics": _EVENT_SEMANTICS,
            "caller_target_authority": False,
            "prototype_v18_allowed": False,
            "terminal_boundary_supported": False,
            "direct_step_jit_supported": True,
            "monolithic_scan_jit_supported": False,
            "scan_execution": _SCAN_EXECUTION,
            "feature_lifecycle_authority_count": 1,
            "feature_router_authority_count": 1,
            "learning_value_router_count": 1,
            "planning_authority": False,
            "dispatch_authority": False,
            "safety_authority": False,
            "evidence_authority": False,
        }
        if any(config.get(name) != value for name, value in fixed.items()):
            raise ValueError("coordinator fixed semantics differ")
        for name in (
            "builder",
            "inner",
            "learning_value_router",
            "candidate_audit",
        ):
            if type(config[name]) is not dict:
                raise ValueError(f"coordinator {name} config must be an exact dict")
        restored = cls(
            builder=LearnableGRUStateBuilderConfig.from_config(
                cast(dict[str, Any], config["builder"])
            ),
            inner=PrototypeRoutedLinearWorldModelEnsembleAdapterConfig.from_config(
                cast(dict[str, object], config["inner"])
            ),
            learning_value_router=LearningValueRouterConfig.from_config(
                cast(dict[str, object], config["learning_value_router"])
            ),
            candidate_audit=CandidateUpdateAuditConfig.from_config(
                cast(dict[str, Any], config["candidate_audit"])
            ),
            max_events=cast(int, config["max_events"]),
        )
        if restored.to_config() != dict(config):
            raise ValueError("coordinator config is not canonical")
        return restored


@chex.dataclass(frozen=True)
class ExternalLearnedStateTransition:
    """One exact continuing raw transition bound to every source revision."""

    source_event_words: Array
    source_builder_step_words: Array
    source_prototype_step_words: Array
    source_feature_generation_words: Array
    observation: Array
    representation: Array
    action: Array
    decision_id: Array
    reward: Array
    discount: Array
    terminated: Array
    truncated: Array
    next_observation: Array
    next_decision_observation: Array
    horde_cumulants: Any = None
    horde_discounts: Any = None


@chex.dataclass(frozen=True)
class ExternalBuilderCandidateAuditEvidence:
    """Source-bound external probes and actor/safety learning-value channels."""

    source_event_words: Array
    source_builder_step_words: Array
    source_prototype_step_words: Array
    source_feature_generation_words: Array
    decision_id: Array
    objective_probe_gradient: Array
    retention_probe_gradient: Array
    safety_cost_gradient: Array
    objective_probe_available: Array
    retention_probe_available: Array
    safety_probe_available: Array
    probe_independence_attested: Array
    advantage: Array
    action_surprisal: Array
    safety_cost: Array
    advantage_available: Array
    action_surprisal_available: Array
    safety_cost_available: Array


@chex.dataclass(frozen=True)
class ExternalLearnedStateRouterAuditCoordinatorState:
    """One full-GRU, one inner composition, one LVR, and exact event caches."""

    builder_state: LearnableGRUStateBuilderState
    inner_state: PrototypeRoutedLinearWorldModelEnsembleAdapterState
    learning_value_router_state: LearningValueRouterState
    current_raw_observation: Array
    current_representation: Array
    current_action: Array
    current_decision_id: Array
    cached_builder_step_words: Array
    cached_prototype_step_words: Array
    cached_feature_generation_words: Array
    event_count: Array
    event_words: Array
    started: Array
    schema_digest: Array


@chex.dataclass(frozen=True)
class ExternalBuilderCausalTargetReceipt:
    """Internally formed stopped target and analytic source-representation pullback."""

    source_event_words: Array
    source_builder_step_words: Array
    source_prototype_step_words: Array
    source_feature_generation_words: Array
    decision_id: Array
    targets: Array
    member_raw_predictions: Array
    representation_objective: Array
    representation_gradient: Array
    target_values_valid: Array
    source_prediction_valid: Array
    gradient_valid: Array
    caller_target_supplied: Array
    additional_model_forward_evaluations: Array


@chex.dataclass(frozen=True)
class ExternalLearnedStateRouterAuditCoordinatorPreparedTransition:
    """All state-owner evaluations and one internally formed learning proposal."""

    source_state: ExternalLearnedStateRouterAuditCoordinatorState
    transition: ExternalLearnedStateTransition
    builder_transition: OnlineGatedStateBuilderTransitionResult
    inner_result: PrototypeRoutedLinearWorldModelEnsembleAdapterResult
    causal_target: ExternalBuilderCausalTargetReceipt
    learning_proposal: StateBuilderLearningProposal
    next_event_count: Array
    next_event_words: Array
    source_matches: Array
    preparation_valid: Array
    external_builder_transition_evaluations: Array
    inner_prototype_update_evaluations: Array
    inner_identity_transition_evaluations: Array
    ensemble_prepare_prediction_evaluations: Array
    ensemble_integrity_prediction_evaluations: Array
    ensemble_member_update_evaluations: Array
    analytic_pullback_evaluations: Array
    additional_model_forward_evaluations: Array


@chex.dataclass(frozen=True)
class ExternalLearnedStateRouterAuditCoordinatorEvaluatedTransition:
    """One routed/audited builder candidate ready for atomic outer adoption."""

    prepared: ExternalLearnedStateRouterAuditCoordinatorPreparedTransition
    candidate_evidence: ExternalBuilderCandidateAuditEvidence
    candidate_evidence_supplied: Array
    candidate_evidence_identity_valid: Array
    learning_value_router_state: LearningValueRouterState
    learning_value_router_result: LearningValueRouterResult
    candidate_audit_evidence: CandidateUpdateAuditEvidence
    candidate_audit_assessment: CandidateUpdateAuditAssessment
    filtered_learning_proposal: StateBuilderLearningProposal
    builder_learning_diagnostics: StateBuilderLearningDiagnostics
    candidate_state: ExternalLearnedStateRouterAuditCoordinatorState
    internally_valid: Array
    learning_value_router_evaluations: Array
    candidate_audit_evaluations: Array
    builder_learning_proposal_evaluations: Array
    builder_learning_commit_evaluations: Array


@chex.dataclass(frozen=True)
class ExternalLearnedStateRouterAuditCoordinatorIntegrityReceipt:
    """Unkeyed exact-content binding; integrity-bound, not authenticated."""

    evaluated: ExternalLearnedStateRouterAuditCoordinatorEvaluatedTransition
    integrity_bound: Array


@chex.dataclass(frozen=True)
class ExternalLearnedStateRouterAuditCoordinatorDiagnostics:
    """Single-owner, exact-work, learning, and outer-adoption facts."""

    source_state_matches: Array
    receipt_matches_evaluation: Array
    receipt_integrity_bound: Array
    source_transition_matches: Array
    continuing_boundary_valid: Array
    external_builder_transition_applied: Array
    inner_transaction_applied: Array
    causal_target_valid: Array
    candidate_evidence_supplied: Array
    candidate_evidence_identity_valid: Array
    candidate_audit_accepted: Array
    builder_learning_applied: Array
    builder_learning_vetoed: Array
    candidate_state_valid: Array
    transaction_applied: Array
    complete_source_returned: Array
    rejected: Array
    external_builder_transition_evaluations: Array
    external_builder_representation_materializations: Array
    inner_prototype_update_evaluations: Array
    inner_identity_transition_evaluations: Array
    ensemble_prepare_prediction_evaluations: Array
    ensemble_integrity_prediction_evaluations: Array
    ensemble_member_update_evaluations: Array
    ensemble_total_member_forward_evaluations: Array
    analytic_pullback_evaluations: Array
    additional_model_forward_evaluations: Array
    learning_value_router_evaluations: Array
    candidate_audit_evaluations: Array
    builder_learning_proposal_evaluations: Array
    builder_learning_commit_evaluations: Array
    feature_bank_mapping_evaluations: Array
    curation_recomputations: Array
    external_builder_owner_count: Array
    inner_identity_builder_count: Array
    feature_lifecycle_authority_count: Array
    feature_router_authority_count: Array
    learning_value_router_count: Array
    caller_target_authority: Array
    planning_authority: Array
    dispatch_authority: Array
    safety_authority: Array
    evidence_authority: Array


@chex.dataclass(frozen=True)
class ExternalLearnedStateRouterAuditCoordinatorResult:
    """Selected outer state plus complete attempted evaluation audit."""

    state: ExternalLearnedStateRouterAuditCoordinatorState
    evaluated: ExternalLearnedStateRouterAuditCoordinatorEvaluatedTransition
    receipt: ExternalLearnedStateRouterAuditCoordinatorIntegrityReceipt
    diagnostics: ExternalLearnedStateRouterAuditCoordinatorDiagnostics


@chex.dataclass(frozen=True)
class ExternalLearnedStateRouterAuditCoordinatorArrayResult:
    """Narrow no-sidecar scan result."""

    state: ExternalLearnedStateRouterAuditCoordinatorState
    actions: Array
    transaction_applied: Array
    builder_learning_applied: Array
    representation_objective: Array


@dataclasses.dataclass(frozen=True, slots=True)
class ExternalLearnedStateRouterAuditCoordinatorResourceBudget:
    """Exact persistent bytes and honest fixed complete-event call counts."""

    persistent_state_bytes: int
    external_builder_state_bytes: int
    inner_state_bytes: int
    learning_value_router_state_bytes: int
    coordinator_cache_and_schema_bytes: int
    persistent_capacity_growth: int
    external_builder_owner_count: int
    inner_identity_builder_count: int
    feature_lifecycle_authority_count: int
    feature_router_authority_count: int
    learning_value_router_count: int
    managed_linear_horde_count: int
    feature_bound_memory_count: int
    routed_ensemble_count: int
    external_builder_transition_evaluations_per_event: int
    external_builder_representation_materializations_per_event: int
    inner_prototype_update_evaluations_per_event: int
    inner_identity_transition_evaluations_per_event: int
    ensemble_prepare_prediction_evaluations_per_event: int
    ensemble_integrity_prediction_evaluations_per_event: int
    ensemble_member_update_evaluations_per_event: int
    ensemble_total_member_forward_evaluations_per_event: int
    analytic_pullback_evaluations_per_event: int
    additional_model_forward_evaluations_for_pullback: int
    learning_value_router_evaluations_per_event: int
    candidate_audit_evaluations_per_event: int
    builder_learning_proposal_evaluations_per_event: int
    builder_learning_commit_evaluations_per_event: int
    feature_bank_mapping_evaluations_per_event: int
    curation_recomputations_per_event: int
    memory_rebind_evaluations_per_event: int
    caller_target_authority: int
    terminal_boundary_supported: int
    direct_step_jit_supported: bool
    monolithic_scan_jit_supported: bool
    scan_execution_host_loop_only: bool
    planning_authority: int
    dispatch_authority: int
    safety_authority: int
    evidence_authority: int
    scientific_promotion_allowed: bool

    def to_config(self) -> dict[str, int | bool]:
        return dataclasses.asdict(self)


class ExternalLearnedStateRouterAuditCoordinator:
    """One full-GRU owner around one exact-Identity inner Prototype."""

    def __init__(
        self,
        config: ExternalLearnedStateRouterAuditCoordinatorConfig,
    ) -> None:
        if type(config) is not ExternalLearnedStateRouterAuditCoordinatorConfig:
            raise TypeError(
                "config must be an exact "
                "ExternalLearnedStateRouterAuditCoordinatorConfig"
            )
        self._config = config
        self._builder = LearnableGRUStateBuilder(config.builder)
        self._inner = PrototypeRoutedLinearWorldModelEnsembleAdapter(config.inner)
        self._learning_value_router = LearningValueRouter(
            config.learning_value_router
        )
        self._raw_dim = config.builder.observation_dim
        self._base_dim = config.builder.feature_dim()
        self._hidden_dim = config.builder.hidden_dim
        self._parameter_count = config.builder.parameter_count()
        self._ensemble_size = config.inner.ensemble.ensemble_size
        self._target_dim = config.inner.ensemble.target_dim
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
    def config(self) -> ExternalLearnedStateRouterAuditCoordinatorConfig:
        return self._config

    @property
    def builder(self) -> LearnableGRUStateBuilder:
        return self._builder

    @property
    def inner(self) -> PrototypeRoutedLinearWorldModelEnsembleAdapter:
        return self._inner

    @property
    def learning_value_router(self) -> LearningValueRouter:
        return self._learning_value_router

    def to_config(self) -> dict[str, object]:
        return self._config.to_config()

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> ExternalLearnedStateRouterAuditCoordinator:
        return cls(ExternalLearnedStateRouterAuditCoordinatorConfig.from_config(config))

    def _feature_generation_words(
        self,
        state: PrototypeRoutedLinearWorldModelEnsembleAdapterState,
    ) -> Array:
        lifecycle, _ = self._inner._bank(state.prototype_state)
        return lifecycle.router_state.generation_words

    def _expected_representation(
        self,
        builder_state: LearnableGRUStateBuilderState,
        raw_observation: Array,
    ) -> Array:
        return jnp.concatenate((raw_observation, builder_state.hidden), axis=0)

    def state_valid(
        self,
        state: ExternalLearnedStateRouterAuditCoordinatorState,
    ) -> Array:
        if type(state) is not ExternalLearnedStateRouterAuditCoordinatorState:
            raise TypeError("state must be an exact coordinator state")
        checks = (
            (state.current_raw_observation, (self._raw_dim,), jnp.float32),
            (state.current_representation, (self._base_dim,), jnp.float32),
            (state.current_action, (), jnp.int32),
            (state.current_decision_id, (4,), jnp.uint32),
            (state.cached_builder_step_words, (2,), jnp.uint32),
            (state.cached_prototype_step_words, (2,), jnp.uint32),
            (state.cached_feature_generation_words, (2,), jnp.uint32),
            (state.event_count, (), jnp.int32),
            (state.event_words, (2,), jnp.uint32),
            (state.started, (), jnp.bool_),
            (state.schema_digest, (_SCHEMA_DIGEST_NBYTES,), jnp.uint8),
        )
        for value, shape, dtype in checks:
            _require_array(value, name="coordinator state field", shape=shape, dtype=dtype)
        prototype = state.inner_state.prototype_state
        feature_state, _ = self._inner._bank(prototype)
        expected_observation_words, observation_capacity = (
            _checked_lifetime_words_increment(state.event_words)
        )
        started_relations = (
            prototype.started
            & jnp.array_equal(
                state.builder_state.step_words,
                expected_observation_words,
            )
            & jnp.array_equal(
                prototype.observation_event_words,
                expected_observation_words,
            )
            & jnp.array_equal(prototype.step_words, state.event_words)
            & jnp.array_equal(feature_state.observe_words, state.event_words)
            & observation_capacity
        )
        unstarted_relations = (
            ~prototype.started
            & jnp.array_equal(state.builder_state.step_words, state.event_words)
            & jnp.array_equal(prototype.observation_event_words, state.event_words)
            & jnp.array_equal(prototype.step_words, state.event_words)
            & jnp.array_equal(feature_state.observe_words, state.event_words)
            & (state.event_count == 0)
        )
        return (
            jnp.array_equal(state.schema_digest, self._schema_digest)
            & self._builder.state_valid(state.builder_state)
            & self._inner.state_valid(state.inner_state)
            & self._learning_value_router.state_valid(
                state.learning_value_router_state
            )
            & _lifetime_counter_valid(state.event_words, state.event_count)
            & (state.event_count <= self._config.max_events)
            & (state.learning_value_router_state.step_count == state.event_count)
            & jnp.array_equal(
                state.cached_builder_step_words,
                state.builder_state.step_words,
            )
            & jnp.array_equal(
                state.cached_prototype_step_words,
                prototype.step_words,
            )
            & jnp.array_equal(
                state.cached_feature_generation_words,
                feature_state.router_state.generation_words,
            )
            & _float32_equal(
                state.current_representation,
                self._expected_representation(
                    state.builder_state,
                    state.current_raw_observation,
                ),
            )
            & _float32_equal(
                state.current_representation,
                prototype.current_raw_observation,
            )
            & (state.current_action == prototype.current_action)
            & jnp.array_equal(
                state.current_decision_id,
                prototype.current_decision_id,
            )
            & (state.started == prototype.started)
            & jnp.where(state.started, started_relations, unstarted_relations)
        )

    def init(
        self,
        key: Array,
        *,
        lifecycle_id: Array | None = None,
    ) -> ExternalLearnedStateRouterAuditCoordinatorState:
        if not (
            hasattr(key, "shape")
            and hasattr(key, "dtype")
            and key.shape == ()
            and jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key)
        ):
            raise TypeError("key must be a scalar typed JAX PRNG key")
        builder_key, inner_key = jr.split(key)
        builder_state = self._builder.init(builder_key)
        inner_state = self._inner.init(inner_key, lifecycle_id=lifecycle_id)
        prototype = inner_state.prototype_state
        state = ExternalLearnedStateRouterAuditCoordinatorState(
            builder_state=builder_state,
            inner_state=inner_state,
            learning_value_router_state=self._learning_value_router.init(),
            current_raw_observation=jnp.zeros((self._raw_dim,), dtype=jnp.float32),
            current_representation=jnp.zeros((self._base_dim,), dtype=jnp.float32),
            current_action=prototype.current_action,
            current_decision_id=prototype.current_decision_id,
            cached_builder_step_words=builder_state.step_words,
            cached_prototype_step_words=prototype.step_words,
            cached_feature_generation_words=self._feature_generation_words(inner_state),
            event_count=jnp.asarray(0, dtype=jnp.int32),
            event_words=jnp.zeros((2,), dtype=jnp.uint32),
            started=jnp.asarray(False, dtype=jnp.bool_),
            schema_digest=self._schema_digest,
        )
        if not bool(jax.device_get(self.state_valid(state))):
            raise ValueError("initial coordinator composition is invalid")
        return state

    def start(
        self,
        state: ExternalLearnedStateRouterAuditCoordinatorState,
        initial_observation: Array,
        *,
        extended_action_mask: Array | None = None,
    ) -> ExternalLearnedStateRouterAuditCoordinatorState:
        raw = _require_array(
            initial_observation,
            name="initial_observation",
            shape=(self._raw_dim,),
            dtype=jnp.float32,
        )
        builder_transition = self._builder.update_with_status(
            state.builder_state,
            raw,
            jnp.asarray(-1, dtype=jnp.int32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(1.0, dtype=jnp.float32),
        )
        inner_state = self._inner.start(
            state.inner_state,
            builder_transition.representation,
            extended_action_mask=extended_action_mask,
        )
        prototype = inner_state.prototype_state
        candidate = ExternalLearnedStateRouterAuditCoordinatorState(
            builder_state=builder_transition.state,
            inner_state=inner_state,
            learning_value_router_state=state.learning_value_router_state,
            current_raw_observation=raw,
            current_representation=builder_transition.representation,
            current_action=prototype.current_action,
            current_decision_id=prototype.current_decision_id,
            cached_builder_step_words=builder_transition.state.step_words,
            cached_prototype_step_words=prototype.step_words,
            cached_feature_generation_words=self._feature_generation_words(inner_state),
            event_count=state.event_count,
            event_words=state.event_words,
            started=jnp.asarray(True, dtype=jnp.bool_),
            schema_digest=state.schema_digest,
        )
        valid = (
            self.state_valid(state)
            & ~state.started
            & builder_transition.transition_applied
            & self.state_valid(candidate)
        )
        return cast(
            ExternalLearnedStateRouterAuditCoordinatorState,
            jax.lax.cond(valid, lambda: candidate, lambda: state),
        )

    def _validate_transition_static(
        self,
        transition: ExternalLearnedStateTransition,
    ) -> None:
        if type(transition) is not ExternalLearnedStateTransition:
            raise TypeError("transition must be an exact ExternalLearnedStateTransition")
        checks = (
            (transition.source_event_words, (2,), jnp.uint32),
            (transition.source_builder_step_words, (2,), jnp.uint32),
            (transition.source_prototype_step_words, (2,), jnp.uint32),
            (transition.source_feature_generation_words, (2,), jnp.uint32),
            (transition.observation, (self._raw_dim,), jnp.float32),
            (transition.representation, (self._base_dim,), jnp.float32),
            (transition.action, (), jnp.int32),
            (transition.decision_id, (4,), jnp.uint32),
            (transition.reward, (), jnp.float32),
            (transition.discount, (), jnp.float32),
            (transition.terminated, (), jnp.bool_),
            (transition.truncated, (), jnp.bool_),
            (transition.next_observation, (self._raw_dim,), jnp.float32),
            (transition.next_decision_observation, (self._raw_dim,), jnp.float32),
        )
        for value, shape, dtype in checks:
            _require_array(value, name="transition field", shape=shape, dtype=dtype)

    def _source_transition_matches(
        self,
        state: ExternalLearnedStateRouterAuditCoordinatorState,
        transition: ExternalLearnedStateTransition,
    ) -> Array:
        return (
            state.started
            & jnp.array_equal(transition.source_event_words, state.event_words)
            & jnp.array_equal(
                transition.source_builder_step_words,
                state.cached_builder_step_words,
            )
            & jnp.array_equal(
                transition.source_prototype_step_words,
                state.cached_prototype_step_words,
            )
            & jnp.array_equal(
                transition.source_feature_generation_words,
                state.cached_feature_generation_words,
            )
            & _float32_equal(transition.observation, state.current_raw_observation)
            & _float32_equal(
                transition.representation,
                state.current_representation,
            )
            & (transition.action == state.current_action)
            & jnp.array_equal(transition.decision_id, state.current_decision_id)
        )

    def _continuing_boundary_valid(
        self,
        transition: ExternalLearnedStateTransition,
    ) -> Array:
        return (
            ~transition.terminated
            & ~transition.truncated
            & _float32_equal(
                transition.next_observation,
                transition.next_decision_observation,
            )
            & jnp.all(jnp.isfinite(transition.next_observation))
            & jnp.isfinite(transition.reward)
            & jnp.isfinite(transition.discount)
            & (transition.discount >= 0.0)
            & (transition.discount <= self._config.inner.ensemble.world_model.gamma)
        )

    def _causal_target(
        self,
        state: ExternalLearnedStateRouterAuditCoordinatorState,
        inner_result: PrototypeRoutedLinearWorldModelEnsembleAdapterResult,
    ) -> ExternalBuilderCausalTargetReceipt:
        adapter_prepared = inner_result.receipt.prepared
        ensemble_prepared = adapter_prepared.ensemble_prepared
        prediction = ensemble_prepared.prediction
        targets = jax.lax.stop_gradient(inner_result.ensemble_result.targets)
        residuals = prediction.member_raw_predictions - targets[None, :]
        weights = jnp.stack(
            tuple(
                jnp.concatenate(
                    member.learner_state.head_params.weights,
                    axis=0,
                )
                for member in ensemble_prepared.source_state.member_states
            )
        )
        input_gradient = jnp.einsum(
            "mh,mhi->i",
            residuals,
            weights,
        ) / jnp.asarray(
            self._ensemble_size * self._target_dim,
            dtype=jnp.float32,
        )
        augmented_gradient = input_gradient[
            : self._config.inner.ensemble.router.total_feature_dim
        ]
        base = ensemble_prepared.base_observation
        descriptors = ensemble_prepared.source_state.consumer_binding.descriptors
        pair_gradient = augmented_gradient[self._base_dim :]
        left = descriptors[:, 0]
        right = descriptors[:, 1]
        representation_gradient = augmented_gradient[: self._base_dim]
        representation_gradient = representation_gradient.at[left].add(
            pair_gradient * base[right]
        )
        representation_gradient = representation_gradient.at[right].add(
            pair_gradient * base[left]
        )
        objective = 0.5 * jnp.mean(jnp.square(residuals))
        target_values_valid = jnp.all(jnp.isfinite(targets))
        source_prediction_valid = (
            prediction.valid
            & jnp.all(jnp.isfinite(prediction.member_raw_predictions))
        )
        gradient_valid = (
            inner_result.diagnostics.transaction_applied
            & target_values_valid
            & source_prediction_valid
            & jnp.isfinite(objective)
            & (objective >= 0.0)
            & jnp.all(jnp.isfinite(representation_gradient))
        )
        return ExternalBuilderCausalTargetReceipt(
            source_event_words=state.event_words,
            source_builder_step_words=state.cached_builder_step_words,
            source_prototype_step_words=state.cached_prototype_step_words,
            source_feature_generation_words=(
                state.cached_feature_generation_words
            ),
            decision_id=state.current_decision_id,
            targets=targets,
            member_raw_predictions=prediction.member_raw_predictions,
            representation_objective=jnp.where(
                gradient_valid,
                objective,
                jnp.asarray(0.0, dtype=jnp.float32),
            ),
            representation_gradient=jnp.where(
                gradient_valid,
                representation_gradient,
                jnp.zeros((self._base_dim,), dtype=jnp.float32),
            ),
            target_values_valid=target_values_valid,
            source_prediction_valid=source_prediction_valid,
            gradient_valid=gradient_valid,
            caller_target_supplied=jnp.asarray(False, dtype=jnp.bool_),
            additional_model_forward_evaluations=jnp.asarray(0, dtype=jnp.int32),
        )

    def prepare_transition(
        self,
        state: ExternalLearnedStateRouterAuditCoordinatorState,
        transition: ExternalLearnedStateTransition,
        *,
        experiential_memory_input: PrototypeExperientialMemoryInput | None = None,
        partner_policy_fusion_input: PrototypePartnerPolicyFusionInput | None = None,
        partner_policy_fusion_feedback: (
            PrototypePartnerPolicyFusionFeedback | None
        ) = None,
        extended_action_mask: Array | None = None,
    ) -> ExternalLearnedStateRouterAuditCoordinatorPreparedTransition:
        """Evaluate each state owner once and form the internal causal proposal."""

        self._validate_transition_static(transition)
        state_is_valid = self.state_valid(state)
        source_matches = self._source_transition_matches(state, transition)
        continuing = self._continuing_boundary_valid(transition)
        next_event_words, event_capacity = _checked_lifetime_words_increment(
            state.event_words
        )
        next_builder_step_words, builder_capacity = (
            _checked_lifetime_words_increment(state.builder_state.step_words)
        )
        configured_capacity = state.event_count < self._config.max_events
        next_event_count = _saturating_int32_counter_increment(state.event_count)
        builder_transition = self._builder.update_with_status(
            state.builder_state,
            transition.next_observation,
            transition.action,
            transition.reward,
            transition.discount,
        )
        next_representation = builder_transition.representation
        prototype_transition = PrototypeTransition(
            observation=state.current_representation,
            action=transition.action,
            decision_id=transition.decision_id,
            reward=transition.reward,
            discount=transition.discount,
            terminated=transition.terminated,
            truncated=transition.truncated,
            next_observation=next_representation,
            next_decision_observation=next_representation,
            horde_cumulants=transition.horde_cumulants,
            horde_discounts=transition.horde_discounts,
        )
        inner_result = self._inner.step(
            state.inner_state,
            prototype_transition,
            experiential_memory_input=experiential_memory_input,
            partner_policy_fusion_input=partner_policy_fusion_input,
            partner_policy_fusion_feedback=partner_policy_fusion_feedback,
            extended_action_mask=extended_action_mask,
        )
        causal_target = self._causal_target(state, inner_result)
        learning_proposal = self._builder.propose_learning_update(
            state.builder_state,
            causal_target.representation_gradient,
        )
        preparation_valid = (
            state_is_valid
            & source_matches
            & continuing
            & event_capacity
            & builder_capacity
            & configured_capacity
            & builder_transition.transition_applied
            & jnp.array_equal(
                builder_transition.pre_step_words,
                state.builder_state.step_words,
            )
            & jnp.array_equal(
                builder_transition.post_step_words,
                next_builder_step_words,
            )
            & inner_result.diagnostics.transaction_applied
            & causal_target.gradient_valid
            & learning_proposal.valid
        )
        ensemble = self._config.inner.ensemble.ensemble_size
        return ExternalLearnedStateRouterAuditCoordinatorPreparedTransition(
            source_state=state,
            transition=transition,
            builder_transition=builder_transition,
            inner_result=inner_result,
            causal_target=causal_target,
            learning_proposal=learning_proposal,
            next_event_count=next_event_count,
            next_event_words=next_event_words,
            source_matches=source_matches,
            preparation_valid=preparation_valid,
            external_builder_transition_evaluations=jnp.asarray(
                1, dtype=jnp.int32
            ),
            inner_prototype_update_evaluations=jnp.asarray(1, dtype=jnp.int32),
            inner_identity_transition_evaluations=jnp.asarray(1, dtype=jnp.int32),
            ensemble_prepare_prediction_evaluations=jnp.asarray(
                ensemble, dtype=jnp.int32
            ),
            ensemble_integrity_prediction_evaluations=jnp.asarray(
                ensemble, dtype=jnp.int32
            ),
            ensemble_member_update_evaluations=jnp.asarray(
                ensemble, dtype=jnp.int32
            ),
            analytic_pullback_evaluations=jnp.asarray(1, dtype=jnp.int32),
            additional_model_forward_evaluations=jnp.asarray(
                0, dtype=jnp.int32
            ),
        )

    def _missing_candidate_evidence(
        self,
        prepared: ExternalLearnedStateRouterAuditCoordinatorPreparedTransition,
    ) -> ExternalBuilderCandidateAuditEvidence:
        source = prepared.source_state
        zero_vector = jnp.zeros((self._parameter_count,), dtype=jnp.float32)
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        false = jnp.asarray(False, dtype=jnp.bool_)
        return ExternalBuilderCandidateAuditEvidence(
            source_event_words=source.event_words,
            source_builder_step_words=source.cached_builder_step_words,
            source_prototype_step_words=source.cached_prototype_step_words,
            source_feature_generation_words=source.cached_feature_generation_words,
            decision_id=source.current_decision_id,
            objective_probe_gradient=zero_vector,
            retention_probe_gradient=zero_vector,
            safety_cost_gradient=zero_vector,
            objective_probe_available=false,
            retention_probe_available=false,
            safety_probe_available=false,
            probe_independence_attested=false,
            advantage=zero,
            action_surprisal=zero,
            safety_cost=zero,
            advantage_available=false,
            action_surprisal_available=false,
            safety_cost_available=false,
        )

    def _validate_candidate_evidence_static(
        self,
        evidence: ExternalBuilderCandidateAuditEvidence,
    ) -> None:
        if type(evidence) is not ExternalBuilderCandidateAuditEvidence:
            raise TypeError(
                "candidate evidence must be exact ExternalBuilderCandidateAuditEvidence"
            )
        checks = (
            (evidence.source_event_words, (2,), jnp.uint32),
            (evidence.source_builder_step_words, (2,), jnp.uint32),
            (evidence.source_prototype_step_words, (2,), jnp.uint32),
            (evidence.source_feature_generation_words, (2,), jnp.uint32),
            (evidence.decision_id, (4,), jnp.uint32),
            (evidence.objective_probe_gradient, (self._parameter_count,), jnp.float32),
            (evidence.retention_probe_gradient, (self._parameter_count,), jnp.float32),
            (evidence.safety_cost_gradient, (self._parameter_count,), jnp.float32),
            (evidence.objective_probe_available, (), jnp.bool_),
            (evidence.retention_probe_available, (), jnp.bool_),
            (evidence.safety_probe_available, (), jnp.bool_),
            (evidence.probe_independence_attested, (), jnp.bool_),
            (evidence.advantage, (), jnp.float32),
            (evidence.action_surprisal, (), jnp.float32),
            (evidence.safety_cost, (), jnp.float32),
            (evidence.advantage_available, (), jnp.bool_),
            (evidence.action_surprisal_available, (), jnp.bool_),
            (evidence.safety_cost_available, (), jnp.bool_),
        )
        for value, shape, dtype in checks:
            _require_array(value, name="candidate evidence field", shape=shape, dtype=dtype)

    def _candidate_evidence_identity(
        self,
        prepared: ExternalLearnedStateRouterAuditCoordinatorPreparedTransition,
        evidence: ExternalBuilderCandidateAuditEvidence,
    ) -> Array:
        source = prepared.source_state
        return (
            jnp.array_equal(evidence.source_event_words, source.event_words)
            & jnp.array_equal(
                evidence.source_builder_step_words,
                source.cached_builder_step_words,
            )
            & jnp.array_equal(
                evidence.source_prototype_step_words,
                source.cached_prototype_step_words,
            )
            & jnp.array_equal(
                evidence.source_feature_generation_words,
                source.cached_feature_generation_words,
            )
            & jnp.array_equal(evidence.decision_id, source.current_decision_id)
        )

    def evaluate_candidate(
        self,
        prepared: ExternalLearnedStateRouterAuditCoordinatorPreparedTransition,
        candidate_evidence: ExternalBuilderCandidateAuditEvidence | None = None,
    ) -> ExternalLearnedStateRouterAuditCoordinatorEvaluatedTransition:
        """Route one learning-value event and audit one external-builder update."""

        if type(prepared) is not (
            ExternalLearnedStateRouterAuditCoordinatorPreparedTransition
        ):
            raise TypeError("prepared must be an exact coordinator preparation")
        supplied = candidate_evidence is not None
        evidence = (
            self._missing_candidate_evidence(prepared)
            if candidate_evidence is None
            else candidate_evidence
        )
        self._validate_candidate_evidence_static(evidence)
        identity_valid = self._candidate_evidence_identity(prepared, evidence)
        identity_gate = identity_valid & prepared.causal_target.gradient_valid
        signals = prepared.inner_result.ensemble_result.signals
        signal_input = signals.availability.input_valid
        delight = jnp.asarray(
            evidence.advantage * evidence.action_surprisal,
            dtype=jnp.float32,
        )
        learning_value = LearningValue(
            advantage=evidence.advantage,
            action_surprisal=evidence.action_surprisal,
            delight=delight,
            epistemic_surprise=signals.epistemic_surprise,
            aleatoric_uncertainty=signals.aleatoric_uncertainty,
            learning_progress=signals.learning_progress,
            change_probability=signals.change_probability,
            safety_cost=evidence.safety_cost,
        )
        declared = LearningValueAvailability(
            advantage=(
                evidence.advantage_available & identity_gate & supplied
            ),
            action_surprisal=(
                evidence.action_surprisal_available & identity_gate & supplied
            ),
            delight=(
                evidence.advantage_available
                & evidence.action_surprisal_available
                & identity_gate
                & supplied
            ),
            epistemic_surprise=(
                signal_input
                & signals.availability.epistemic
                & prepared.causal_target.gradient_valid
            ),
            aleatoric_uncertainty=(
                signal_input
                & signals.availability.aleatoric
                & prepared.causal_target.gradient_valid
            ),
            learning_progress=(
                signal_input
                & signals.availability.learning_progress
                & prepared.causal_target.gradient_valid
            ),
            change_probability=(
                signal_input
                & signals.availability.change_probability
                & prepared.causal_target.gradient_valid
            ),
            safety_cost=(
                evidence.safety_cost_available & identity_gate & supplied
            ),
        )
        router_state, router_result = self._learning_value_router.route(
            prepared.source_state.learning_value_router_state,
            learning_value,
            declared,
        )
        candidate_available = prepared.causal_target.gradient_valid & identity_gate
        raw_route = router_result.candidate_update_audit_evidence
        audit_evidence = CandidateUpdateAuditEvidence(
            objective_probe_gradient=evidence.objective_probe_gradient,
            retention_probe_gradient=evidence.retention_probe_gradient,
            safety_cost_gradient=evidence.safety_cost_gradient,
            objective_probe_available=(
                evidence.objective_probe_available
                & candidate_available
                & supplied
            ),
            retention_probe_available=(
                evidence.retention_probe_available
                & candidate_available
                & supplied
            ),
            safety_probe_available=(
                evidence.safety_probe_available
                & candidate_available
                & supplied
            ),
            probe_independence_attested=(
                evidence.probe_independence_attested
                & candidate_available
                & supplied
            ),
            learning_value=raw_route.values,
            learning_value_availability=raw_route.availability,
        )
        assessment = assess_candidate_update(
            prepared.learning_proposal.candidate_parameter_update,
            audit_evidence,
            self._config.candidate_audit,
        )
        filtered = replace_state_builder_learning_proposal_update(
            prepared.learning_proposal,
            cast(Array, assessment.weighted_update),
            assessment.accepted,
        )
        learned_builder, learning_diagnostics = (
            self._builder.commit_learning_update(
                prepared.builder_transition.state,
                filtered,
            )
        )
        learning_consistent = jnp.where(
            assessment.accepted,
            learning_diagnostics.applied,
            learning_diagnostics.rejected
            & learning_diagnostics.source_matches
            & _tree_equal(learned_builder, prepared.builder_transition.state),
        )
        inner_state = prepared.inner_result.state
        prototype = inner_state.prototype_state
        candidate_state = ExternalLearnedStateRouterAuditCoordinatorState(
            builder_state=learned_builder,
            inner_state=inner_state,
            learning_value_router_state=router_state,
            current_raw_observation=prepared.transition.next_observation,
            current_representation=prepared.builder_transition.representation,
            current_action=prototype.current_action,
            current_decision_id=prototype.current_decision_id,
            cached_builder_step_words=learned_builder.step_words,
            cached_prototype_step_words=prototype.step_words,
            cached_feature_generation_words=self._feature_generation_words(inner_state),
            event_count=prepared.next_event_count,
            event_words=prepared.next_event_words,
            started=jnp.asarray(True, dtype=jnp.bool_),
            schema_digest=prepared.source_state.schema_digest,
        )
        evidence_integrity = jnp.where(
            jnp.asarray(supplied, dtype=jnp.bool_),
            identity_valid,
            jnp.asarray(True, dtype=jnp.bool_),
        )
        internally_valid = (
            prepared.preparation_valid
            & evidence_integrity
            & router_result.diagnostics.state_valid
            & router_result.diagnostics.counter_capacity_available
            & self._learning_value_router.state_valid(router_state)
            & learning_consistent
            & self.state_valid(candidate_state)
        )
        return ExternalLearnedStateRouterAuditCoordinatorEvaluatedTransition(
            prepared=prepared,
            candidate_evidence=evidence,
            candidate_evidence_supplied=jnp.asarray(supplied, dtype=jnp.bool_),
            candidate_evidence_identity_valid=identity_valid,
            learning_value_router_state=router_state,
            learning_value_router_result=router_result,
            candidate_audit_evidence=audit_evidence,
            candidate_audit_assessment=assessment,
            filtered_learning_proposal=filtered,
            builder_learning_diagnostics=learning_diagnostics,
            candidate_state=candidate_state,
            internally_valid=internally_valid,
            learning_value_router_evaluations=jnp.asarray(1, dtype=jnp.int32),
            candidate_audit_evaluations=jnp.asarray(1, dtype=jnp.int32),
            builder_learning_proposal_evaluations=jnp.asarray(1, dtype=jnp.int32),
            builder_learning_commit_evaluations=jnp.asarray(1, dtype=jnp.int32),
        )

    def integrity_receipt(
        self,
        evaluated: ExternalLearnedStateRouterAuditCoordinatorEvaluatedTransition,
    ) -> ExternalLearnedStateRouterAuditCoordinatorIntegrityReceipt:
        """Bind exact evaluated content without claiming authentication."""

        if type(evaluated) is not (
            ExternalLearnedStateRouterAuditCoordinatorEvaluatedTransition
        ):
            raise TypeError("evaluated must be an exact coordinator evaluation")
        return ExternalLearnedStateRouterAuditCoordinatorIntegrityReceipt(
            evaluated=evaluated,
            integrity_bound=jnp.asarray(True, dtype=jnp.bool_),
        )

    def adopt_evaluated_transition(
        self,
        state: ExternalLearnedStateRouterAuditCoordinatorState,
        evaluated: ExternalLearnedStateRouterAuditCoordinatorEvaluatedTransition,
        receipt: ExternalLearnedStateRouterAuditCoordinatorIntegrityReceipt,
    ) -> ExternalLearnedStateRouterAuditCoordinatorResult:
        """Atomically select the complete evaluated destination or source."""

        if type(state) is not ExternalLearnedStateRouterAuditCoordinatorState:
            raise TypeError("state must be an exact coordinator state")
        if type(evaluated) is not (
            ExternalLearnedStateRouterAuditCoordinatorEvaluatedTransition
        ):
            raise TypeError("evaluated must be an exact coordinator evaluation")
        if type(receipt) is not (
            ExternalLearnedStateRouterAuditCoordinatorIntegrityReceipt
        ):
            raise TypeError("receipt must be an exact coordinator receipt")
        _require_array(
            receipt.integrity_bound,
            name="receipt.integrity_bound",
            shape=(),
            dtype=jnp.bool_,
        )
        source_matches = _tree_equal(state, evaluated.prepared.source_state)
        receipt_matches = _tree_equal(receipt.evaluated, evaluated)
        candidate_valid = self.state_valid(evaluated.candidate_state)
        commit = (
            self.state_valid(state)
            & source_matches
            & receipt_matches
            & receipt.integrity_bound
            & evaluated.internally_valid
            & candidate_valid
        )
        selected_state = cast(
            ExternalLearnedStateRouterAuditCoordinatorState,
            jax.lax.cond(commit, lambda: evaluated.candidate_state, lambda: state),
        )
        prepared = evaluated.prepared
        ensemble_total = (
            prepared.ensemble_prepare_prediction_evaluations
            + prepared.ensemble_integrity_prediction_evaluations
            + prepared.ensemble_member_update_evaluations
        )
        diagnostics = ExternalLearnedStateRouterAuditCoordinatorDiagnostics(
            source_state_matches=source_matches,
            receipt_matches_evaluation=receipt_matches,
            receipt_integrity_bound=receipt.integrity_bound,
            source_transition_matches=prepared.source_matches,
            continuing_boundary_valid=self._continuing_boundary_valid(
                prepared.transition
            ),
            external_builder_transition_applied=(
                prepared.builder_transition.transition_applied
            ),
            inner_transaction_applied=(
                prepared.inner_result.diagnostics.transaction_applied
            ),
            causal_target_valid=prepared.causal_target.gradient_valid,
            candidate_evidence_supplied=evaluated.candidate_evidence_supplied,
            candidate_evidence_identity_valid=(
                evaluated.candidate_evidence_identity_valid
            ),
            candidate_audit_accepted=(
                evaluated.candidate_audit_assessment.accepted
            ),
            builder_learning_applied=(
                evaluated.builder_learning_diagnostics.applied & commit
            ),
            builder_learning_vetoed=(
                ~evaluated.candidate_audit_assessment.accepted & commit
            ),
            candidate_state_valid=candidate_valid,
            transaction_applied=commit,
            complete_source_returned=~commit,
            rejected=~commit,
            external_builder_transition_evaluations=(
                prepared.external_builder_transition_evaluations
            ),
            external_builder_representation_materializations=jnp.asarray(
                2, dtype=jnp.int32
            ),
            inner_prototype_update_evaluations=(
                prepared.inner_prototype_update_evaluations
            ),
            inner_identity_transition_evaluations=(
                prepared.inner_identity_transition_evaluations
            ),
            ensemble_prepare_prediction_evaluations=(
                prepared.ensemble_prepare_prediction_evaluations
            ),
            ensemble_integrity_prediction_evaluations=(
                prepared.ensemble_integrity_prediction_evaluations
            ),
            ensemble_member_update_evaluations=(
                prepared.ensemble_member_update_evaluations
            ),
            ensemble_total_member_forward_evaluations=ensemble_total,
            analytic_pullback_evaluations=prepared.analytic_pullback_evaluations,
            additional_model_forward_evaluations=(
                prepared.additional_model_forward_evaluations
            ),
            learning_value_router_evaluations=(
                evaluated.learning_value_router_evaluations
            ),
            candidate_audit_evaluations=evaluated.candidate_audit_evaluations,
            builder_learning_proposal_evaluations=(
                evaluated.builder_learning_proposal_evaluations
            ),
            builder_learning_commit_evaluations=(
                evaluated.builder_learning_commit_evaluations
            ),
            feature_bank_mapping_evaluations=jnp.asarray(3, dtype=jnp.int32),
            curation_recomputations=jnp.asarray(0, dtype=jnp.int32),
            external_builder_owner_count=jnp.asarray(1, dtype=jnp.int32),
            inner_identity_builder_count=jnp.asarray(1, dtype=jnp.int32),
            feature_lifecycle_authority_count=jnp.asarray(1, dtype=jnp.int32),
            feature_router_authority_count=jnp.asarray(1, dtype=jnp.int32),
            learning_value_router_count=jnp.asarray(1, dtype=jnp.int32),
            caller_target_authority=jnp.asarray(False, dtype=jnp.bool_),
            planning_authority=jnp.asarray(False, dtype=jnp.bool_),
            dispatch_authority=jnp.asarray(False, dtype=jnp.bool_),
            safety_authority=jnp.asarray(False, dtype=jnp.bool_),
            evidence_authority=jnp.asarray(False, dtype=jnp.bool_),
        )
        return ExternalLearnedStateRouterAuditCoordinatorResult(
            state=selected_state,
            evaluated=evaluated,
            receipt=receipt,
            diagnostics=diagnostics,
        )

    def step(
        self,
        state: ExternalLearnedStateRouterAuditCoordinatorState,
        transition: ExternalLearnedStateTransition,
        candidate_evidence: ExternalBuilderCandidateAuditEvidence | None = None,
        *,
        experiential_memory_input: PrototypeExperientialMemoryInput | None = None,
        partner_policy_fusion_input: PrototypePartnerPolicyFusionInput | None = None,
        partner_policy_fusion_feedback: (
            PrototypePartnerPolicyFusionFeedback | None
        ) = None,
        extended_action_mask: Array | None = None,
    ) -> ExternalLearnedStateRouterAuditCoordinatorResult:
        """Prepare once, audit once, integrity-bind, and atomically adopt."""

        prepared = self.prepare_transition(
            state,
            transition,
            experiential_memory_input=experiential_memory_input,
            partner_policy_fusion_input=partner_policy_fusion_input,
            partner_policy_fusion_feedback=partner_policy_fusion_feedback,
            extended_action_mask=extended_action_mask,
        )
        evaluated = self.evaluate_candidate(prepared, candidate_evidence)
        receipt = self.integrity_receipt(evaluated)
        return self.adopt_evaluated_transition(state, evaluated, receipt)

    def scan_transitions(
        self,
        state: ExternalLearnedStateRouterAuditCoordinatorState,
        transitions: ExternalLearnedStateTransition,
    ) -> ExternalLearnedStateRouterAuditCoordinatorArrayResult:
        """Run the explicit no-sidecar batch as a bounded host loop.

        The full nested Prototype transaction has a supported direct-step JIT
        boundary.  A monolithic ``lax.scan`` compilation is deliberately not
        exposed because its compiler-memory footprint is not bounded by this
        mechanism's persistent resource budget.
        """

        if isinstance(state.event_count, jax.core.Tracer):
            raise RuntimeError(
                "coordinator scan_transitions is host-only; "
                "monolithic scan JIT is unsupported"
            )
        leaves = jax.tree.leaves(transitions)
        if not leaves:
            raise ValueError("transitions must contain at least one array field")
        lengths: set[int] = set()
        for leaf in leaves:
            value = jnp.asarray(leaf)
            if value.ndim < 1:
                raise ValueError("every batched transition field needs a leading axis")
            lengths.add(int(value.shape[0]))
        if len(lengths) != 1:
            raise ValueError("batched transition leading dimensions must match")
        length = lengths.pop()
        if length < 1:
            raise ValueError("batched transition length must be positive")
        if length > self._config.max_events:
            raise ValueError("batched transition length exceeds coordinator capacity")
        current = state
        actions: list[Array] = []
        applied: list[Array] = []
        learned: list[Array] = []
        objective: list[Array] = []
        for index in range(length):
            transition = jax.tree.map(lambda value: value[index], transitions)
            result = self.step(current, transition)
            current = result.state
            actions.append(current.current_action)
            applied.append(result.diagnostics.transaction_applied)
            learned.append(result.diagnostics.builder_learning_applied)
            objective.append(
                result.evaluated.prepared.causal_target.representation_objective
            )
        return ExternalLearnedStateRouterAuditCoordinatorArrayResult(
            state=current,
            actions=jnp.stack(actions),
            transaction_applied=jnp.stack(applied),
            builder_learning_applied=jnp.stack(learned),
            representation_objective=jnp.stack(objective),
        )

    @property
    def resource_budget(
        self,
    ) -> ExternalLearnedStateRouterAuditCoordinatorResourceBudget:
        """Declare exact ownership and every fixed complete-event evaluation."""

        state = self.init(jr.key(0))
        builder_bytes = _tree_nbytes(state.builder_state)
        inner_bytes = (
            measure_prototype_routed_linear_world_model_ensemble_adapter_state_nbytes(
                state.inner_state
            )
        )
        router_bytes = _tree_nbytes(state.learning_value_router_state)
        total_bytes = _tree_nbytes(state)
        cache_bytes = total_bytes - builder_bytes - inner_bytes - router_bytes
        ensemble = self._ensemble_size
        memory_enabled = self._config.inner.prototype.experiential_memory is not None
        return ExternalLearnedStateRouterAuditCoordinatorResourceBudget(
            persistent_state_bytes=total_bytes,
            external_builder_state_bytes=builder_bytes,
            inner_state_bytes=inner_bytes,
            learning_value_router_state_bytes=router_bytes,
            coordinator_cache_and_schema_bytes=cache_bytes,
            persistent_capacity_growth=0,
            external_builder_owner_count=1,
            inner_identity_builder_count=1,
            feature_lifecycle_authority_count=1,
            feature_router_authority_count=1,
            learning_value_router_count=1,
            managed_linear_horde_count=1,
            feature_bound_memory_count=int(memory_enabled),
            routed_ensemble_count=1,
            external_builder_transition_evaluations_per_event=1,
            external_builder_representation_materializations_per_event=2,
            inner_prototype_update_evaluations_per_event=1,
            inner_identity_transition_evaluations_per_event=1,
            ensemble_prepare_prediction_evaluations_per_event=ensemble,
            ensemble_integrity_prediction_evaluations_per_event=ensemble,
            ensemble_member_update_evaluations_per_event=ensemble,
            ensemble_total_member_forward_evaluations_per_event=3 * ensemble,
            analytic_pullback_evaluations_per_event=1,
            additional_model_forward_evaluations_for_pullback=0,
            learning_value_router_evaluations_per_event=1,
            candidate_audit_evaluations_per_event=1,
            builder_learning_proposal_evaluations_per_event=1,
            builder_learning_commit_evaluations_per_event=1,
            feature_bank_mapping_evaluations_per_event=3,
            curation_recomputations_per_event=0,
            memory_rebind_evaluations_per_event=int(memory_enabled),
            caller_target_authority=0,
            terminal_boundary_supported=0,
            direct_step_jit_supported=True,
            monolithic_scan_jit_supported=False,
            scan_execution_host_loop_only=True,
            planning_authority=0,
            dispatch_authority=0,
            safety_authority=0,
            evidence_authority=0,
            scientific_promotion_allowed=False,
        )


def measure_external_learned_state_router_audit_coordinator_state_nbytes(
    state: ExternalLearnedStateRouterAuditCoordinatorState,
) -> int:
    """Measure every persistent array leaf in one coordinator state."""

    if type(state) is not ExternalLearnedStateRouterAuditCoordinatorState:
        raise TypeError("state must be an exact coordinator state")
    return _tree_nbytes(state)


def save_external_learned_state_router_audit_coordinator_checkpoint(
    owner: ExternalLearnedStateRouterAuditCoordinator,
    state: ExternalLearnedStateRouterAuditCoordinatorState,
    path: str | Path,
) -> None:
    """Persist the one-owner coordinator state, excluding transient receipts."""

    if type(owner) is not ExternalLearnedStateRouterAuditCoordinator:
        raise TypeError("owner must be an exact coordinator")
    if not bool(jax.device_get(owner.state_valid(state))):
        raise ValueError("refusing to save an invalid coordinator state")
    config = owner.to_config()
    save_checkpoint(
        state,
        path,
        metadata={
            "schema": EXTERNAL_LEARNED_STATE_ROUTER_AUDIT_COORDINATOR_CHECKPOINT_SCHEMA,
            "owner_config": config,
            "config_sha256": _config_digest(config),
            "resource_budget": owner.resource_budget.to_config(),
            "evidence_level": (
                EXTERNAL_LEARNED_STATE_ROUTER_AUDIT_COORDINATOR_EVIDENCE_LEVEL
            ),
            "outcome_status": (
                EXTERNAL_LEARNED_STATE_ROUTER_AUDIT_COORDINATOR_OUTCOME_STATUS
            ),
            "scientific_promotion_allowed": False,
            "feature_lifecycle_authority_count": 1,
            "feature_router_authority_count": 1,
            "learning_value_router_count": 1,
            "transient_receipt_included": False,
            "caller_target_authority": False,
            "prototype_v18_allowed": False,
            "direct_step_jit_supported": True,
            "monolithic_scan_jit_supported": False,
            "scan_execution": _SCAN_EXECUTION,
            "planning_authority": False,
            "dispatch_authority": False,
            "safety_authority": False,
            "evidence_authority": False,
        },
    )


def load_external_learned_state_router_audit_coordinator_checkpoint(
    path: str | Path,
) -> tuple[
    ExternalLearnedStateRouterAuditCoordinator,
    ExternalLearnedStateRouterAuditCoordinatorState,
]:
    """Strictly restore the sole current coordinator v1 schema."""

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
        "learning_value_router_count",
        "transient_receipt_included",
        "caller_target_authority",
        "prototype_v18_allowed",
        "direct_step_jit_supported",
        "monolithic_scan_jit_supported",
        "scan_execution",
        "planning_authority",
        "dispatch_authority",
        "safety_authority",
        "evidence_authority",
    }
    if set(metadata) != expected:
        raise ValueError("coordinator checkpoint metadata fields are not exact")
    if metadata.get("schema") != (
        EXTERNAL_LEARNED_STATE_ROUTER_AUDIT_COORDINATOR_CHECKPOINT_SCHEMA
    ):
        raise ValueError("checkpoint is not a coordinator v1 checkpoint")
    config = metadata.get("owner_config")
    if type(config) is not dict:
        raise ValueError("coordinator checkpoint lacks exact owner_config")
    if metadata.get("config_sha256") != _config_digest(config):
        raise ValueError("coordinator checkpoint config digest does not match")
    owner = ExternalLearnedStateRouterAuditCoordinator.from_config(config)
    if metadata.get("resource_budget") != owner.resource_budget.to_config():
        raise ValueError("coordinator checkpoint resource budget does not match")
    fixed = {
        "evidence_level": (
            EXTERNAL_LEARNED_STATE_ROUTER_AUDIT_COORDINATOR_EVIDENCE_LEVEL
        ),
        "outcome_status": (
            EXTERNAL_LEARNED_STATE_ROUTER_AUDIT_COORDINATOR_OUTCOME_STATUS
        ),
        "scientific_promotion_allowed": False,
        "feature_lifecycle_authority_count": 1,
        "feature_router_authority_count": 1,
        "learning_value_router_count": 1,
        "transient_receipt_included": False,
        "caller_target_authority": False,
        "prototype_v18_allowed": False,
        "direct_step_jit_supported": True,
        "monolithic_scan_jit_supported": False,
        "scan_execution": _SCAN_EXECUTION,
        "planning_authority": False,
        "dispatch_authority": False,
        "safety_authority": False,
        "evidence_authority": False,
    }
    if any(metadata.get(name) != value for name, value in fixed.items()):
        raise ValueError("coordinator checkpoint fixed semantics differ")
    template = owner.init(jr.key(0))
    restored, second_metadata = load_checkpoint(template, path)
    if second_metadata != metadata:
        raise ValueError("coordinator checkpoint metadata changed between reads")
    state = cast(ExternalLearnedStateRouterAuditCoordinatorState, restored)
    if not bool(jax.device_get(owner.state_valid(state))):
        raise ValueError("coordinator checkpoint restored an invalid state")
    if measure_external_learned_state_router_audit_coordinator_state_nbytes(
        state
    ) != owner.resource_budget.persistent_state_bytes:
        raise ValueError("coordinator checkpoint restored a wrong-size state")
    return owner, state


__all__ = [
    "EXTERNAL_LEARNED_STATE_ROUTER_AUDIT_COORDINATOR_CHECKPOINT_SCHEMA",
    "EXTERNAL_LEARNED_STATE_ROUTER_AUDIT_COORDINATOR_CONFIG_SCHEMA",
    "EXTERNAL_LEARNED_STATE_ROUTER_AUDIT_COORDINATOR_EVIDENCE_LEVEL",
    "EXTERNAL_LEARNED_STATE_ROUTER_AUDIT_COORDINATOR_OUTCOME_STATUS",
    "EXTERNAL_LEARNED_STATE_ROUTER_AUDIT_COORDINATOR_RECEIPT_SCHEMA",
    "EXTERNAL_LEARNED_STATE_ROUTER_AUDIT_COORDINATOR_SCIENTIFIC_PROMOTION_ALLOWED",
    "EXTERNAL_LEARNED_STATE_ROUTER_AUDIT_COORDINATOR_STATE_SCHEMA",
    "ExternalBuilderCandidateAuditEvidence",
    "ExternalBuilderCausalTargetReceipt",
    "ExternalLearnedStateRouterAuditCoordinator",
    "ExternalLearnedStateRouterAuditCoordinatorArrayResult",
    "ExternalLearnedStateRouterAuditCoordinatorConfig",
    "ExternalLearnedStateRouterAuditCoordinatorDiagnostics",
    "ExternalLearnedStateRouterAuditCoordinatorEvaluatedTransition",
    "ExternalLearnedStateRouterAuditCoordinatorIntegrityReceipt",
    "ExternalLearnedStateRouterAuditCoordinatorPreparedTransition",
    "ExternalLearnedStateRouterAuditCoordinatorResourceBudget",
    "ExternalLearnedStateRouterAuditCoordinatorResult",
    "ExternalLearnedStateRouterAuditCoordinatorState",
    "ExternalLearnedStateTransition",
    "load_external_learned_state_router_audit_coordinator_checkpoint",
    "measure_external_learned_state_router_audit_coordinator_state_nbytes",
    "save_external_learned_state_router_audit_coordinator_checkpoint",
]
