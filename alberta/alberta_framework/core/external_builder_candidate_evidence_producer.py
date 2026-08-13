# mypy: disable-error-code="attr-defined,call-arg"
"""Source-bound representation-probe evidence for the external full GRU.

The external coordinator's candidate-update audit consumes gradients in the
full-GRU parameter space.  Real consumers naturally observe independent
objective, retention, and safety probes at the representation boundary.  This
stateless adapter binds those probes to one exact coordinator decision and
pulls them through that decision's cached RTRL sensitivity matrix.

The adapter does not invent probes, infer their independence, own a learner,
or execute an actor backward.  In particular, its result has no
``sparks_joy`` field: paper-defined joy is established only by an actual
``KondoSparseActor`` backward.  Missing, stale, malformed, or non-finite probe
evidence becomes an exact unavailable zero vector and therefore fails closed
at the existing candidate audit.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
from jax import Array

from alberta_framework.core.external_learned_state_router_audit_coordinator import (
    ExternalBuilderCandidateAuditEvidence,
    ExternalLearnedStateRouterAuditCoordinatorState,
)
from alberta_framework.core.state_builder import (
    LearnableGRUStateBuilder,
    LearnableGRUStateBuilderConfig,
    LearnableGRUStateBuilderState,
)

EXTERNAL_BUILDER_CANDIDATE_EVIDENCE_PRODUCER_SCHEMA = (
    "alberta.external-builder-candidate-evidence-producer.v1"
)


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


@chex.dataclass(frozen=True)
class ExternalBuilderRepresentationProbeEvidence:
    """Caller-owned probes bound to one exact emitted representation.

    ``probe_independence_attested`` remains an explicit caller attestation.
    The producer checks identity and numeric contracts but cannot prove that
    the three losses were measured independently of the candidate update.
    """

    source_event_words: Array
    source_builder_step_words: Array
    source_prototype_step_words: Array
    source_feature_generation_words: Array
    decision_id: Array
    objective_representation_gradient: Array
    retention_representation_gradient: Array
    safety_representation_gradient: Array
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
class ExternalBuilderCandidateEvidenceProducerDiagnostics:
    """Exact source, pullback, and non-authority facts."""

    source_state_valid: Array
    source_identity_matches: Array
    objective_probe_gradient_valid: Array
    retention_probe_gradient_valid: Array
    safety_probe_gradient_valid: Array
    probe_pullbacks_valid: Array
    scalar_channels_valid: Array
    evidence_ready: Array
    analytic_probe_pullback_evaluations: Array
    additional_model_forward_evaluations: Array
    actor_backward_evaluations: Array
    delight_or_actor_backward: Array


@chex.dataclass(frozen=True)
class ExternalBuilderCandidateEvidenceProducerResult:
    """One exact candidate-audit sidecar and its source-binding audit."""

    evidence: ExternalBuilderCandidateAuditEvidence
    diagnostics: ExternalBuilderCandidateEvidenceProducerDiagnostics


@dataclasses.dataclass(frozen=True, slots=True)
class ExternalBuilderCandidateEvidenceProducerResourceBudget:
    """Stateless ownership and fixed work declaration."""

    persistent_state_bytes: int
    persistent_capacity_growth: int
    full_gru_owner_count: int
    candidate_audit_owner_count: int
    analytic_probe_pullbacks_per_event: int
    additional_model_forward_evaluations_per_event: int
    actor_backward_evaluations_per_event: int
    caller_probe_authority: bool
    caller_independence_attestation_required: bool
    delight_or_actor_backward: bool


class ExternalBuilderCandidateEvidenceProducer:
    """Pull exact representation probes back to full-GRU parameters."""

    def __init__(self, builder_config: LearnableGRUStateBuilderConfig) -> None:
        if type(builder_config) is not LearnableGRUStateBuilderConfig:
            raise TypeError(
                "builder_config must be an exact LearnableGRUStateBuilderConfig"
            )
        if not builder_config.include_raw_observation:
            raise ValueError(
                "external candidate evidence requires include_raw_observation=True"
            )
        self._config = builder_config
        self._builder = LearnableGRUStateBuilder(builder_config)

    @property
    def config(self) -> LearnableGRUStateBuilderConfig:
        return self._config

    @property
    def resource_budget(self) -> ExternalBuilderCandidateEvidenceProducerResourceBudget:
        return ExternalBuilderCandidateEvidenceProducerResourceBudget(
            persistent_state_bytes=0,
            persistent_capacity_growth=0,
            full_gru_owner_count=0,
            candidate_audit_owner_count=0,
            analytic_probe_pullbacks_per_event=3,
            additional_model_forward_evaluations_per_event=0,
            actor_backward_evaluations_per_event=0,
            caller_probe_authority=True,
            caller_independence_attestation_required=True,
            delight_or_actor_backward=False,
        )

    def to_config(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "schema": EXTERNAL_BUILDER_CANDIDATE_EVIDENCE_PRODUCER_SCHEMA,
            "builder": self._config.to_config(),
            "source_binding": (
                "event+builder+prototype+feature-generation+decision-id"
            ),
            "probe_space": "emitted-representation",
            "candidate_space": "full-gru-parameters",
            "pullback": "cached-source-rtrl-sensitivity",
            "caller_probe_authority": True,
            "caller_independence_attestation_required": True,
            "additional_model_forward_evaluations": 0,
            "actor_backward_evaluations": 0,
            "delight_or_actor_backward": False,
            "scientific_promotion_allowed": False,
        }

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, object],
    ) -> ExternalBuilderCandidateEvidenceProducer:
        """Strictly reconstruct the canonical stateless producer manifest."""

        expected = {
            "type",
            "schema",
            "builder",
            "source_binding",
            "probe_space",
            "candidate_space",
            "pullback",
            "caller_probe_authority",
            "caller_independence_attestation_required",
            "additional_model_forward_evaluations",
            "actor_backward_evaluations",
            "delight_or_actor_backward",
            "scientific_promotion_allowed",
        }
        if type(payload) is not dict or set(payload) != expected:
            raise ValueError("candidate-evidence producer config fields are not exact")
        fixed: dict[str, object] = {
            "type": cls.__name__,
            "schema": EXTERNAL_BUILDER_CANDIDATE_EVIDENCE_PRODUCER_SCHEMA,
            "source_binding": (
                "event+builder+prototype+feature-generation+decision-id"
            ),
            "probe_space": "emitted-representation",
            "candidate_space": "full-gru-parameters",
            "pullback": "cached-source-rtrl-sensitivity",
            "caller_probe_authority": True,
            "caller_independence_attestation_required": True,
            "additional_model_forward_evaluations": 0,
            "actor_backward_evaluations": 0,
            "delight_or_actor_backward": False,
            "scientific_promotion_allowed": False,
        }
        if any(payload.get(name) != value for name, value in fixed.items()):
            raise ValueError("candidate-evidence producer fixed semantics differ")
        if type(payload["builder"]) is not dict:
            raise ValueError("candidate-evidence producer builder must be an exact dict")
        restored = cls(
            LearnableGRUStateBuilderConfig.from_config(
                cast(dict[str, Any], payload["builder"])
            )
        )
        if restored.to_config() != dict(payload):
            raise ValueError("candidate-evidence producer config is not canonical")
        return restored

    def _validate_source_static(
        self,
        state: ExternalLearnedStateRouterAuditCoordinatorState,
    ) -> None:
        if type(state) is not ExternalLearnedStateRouterAuditCoordinatorState:
            raise TypeError(
                "state must be an exact ExternalLearnedStateRouterAuditCoordinatorState"
            )
        if type(state.builder_state) is not LearnableGRUStateBuilderState:
            raise TypeError("state.builder_state must be an exact full-GRU state")
        checks = (
            (state.current_raw_observation, (self._config.observation_dim,), jnp.float32),
            (state.current_representation, (self._config.feature_dim(),), jnp.float32),
            (state.current_action, (), jnp.int32),
            (state.current_decision_id, (4,), jnp.uint32),
            (state.cached_builder_step_words, (2,), jnp.uint32),
            (state.cached_prototype_step_words, (2,), jnp.uint32),
            (state.cached_feature_generation_words, (2,), jnp.uint32),
            (state.event_count, (), jnp.int32),
            (state.event_words, (2,), jnp.uint32),
            (state.started, (), jnp.bool_),
            (state.schema_digest, (32,), jnp.uint8),
        )
        for value, shape, dtype in checks:
            _require_array(value, name="source state field", shape=shape, dtype=dtype)

    def _validate_probe_static(
        self,
        probes: ExternalBuilderRepresentationProbeEvidence,
    ) -> None:
        if type(probes) is not ExternalBuilderRepresentationProbeEvidence:
            raise TypeError(
                "probes must be an exact ExternalBuilderRepresentationProbeEvidence"
            )
        feature_dim = self._config.feature_dim()
        checks = (
            (probes.source_event_words, "source_event_words", (2,), jnp.uint32),
            (
                probes.source_builder_step_words,
                "source_builder_step_words",
                (2,),
                jnp.uint32,
            ),
            (
                probes.source_prototype_step_words,
                "source_prototype_step_words",
                (2,),
                jnp.uint32,
            ),
            (
                probes.source_feature_generation_words,
                "source_feature_generation_words",
                (2,),
                jnp.uint32,
            ),
            (probes.decision_id, "decision_id", (4,), jnp.uint32),
            (
                probes.objective_representation_gradient,
                "objective_representation_gradient",
                (feature_dim,),
                jnp.float32,
            ),
            (
                probes.retention_representation_gradient,
                "retention_representation_gradient",
                (feature_dim,),
                jnp.float32,
            ),
            (
                probes.safety_representation_gradient,
                "safety_representation_gradient",
                (feature_dim,),
                jnp.float32,
            ),
            (probes.objective_probe_available, "objective_probe_available", (), jnp.bool_),
            (probes.retention_probe_available, "retention_probe_available", (), jnp.bool_),
            (probes.safety_probe_available, "safety_probe_available", (), jnp.bool_),
            (
                probes.probe_independence_attested,
                "probe_independence_attested",
                (),
                jnp.bool_,
            ),
            (probes.advantage, "advantage", (), jnp.float32),
            (probes.action_surprisal, "action_surprisal", (), jnp.float32),
            (probes.safety_cost, "safety_cost", (), jnp.float32),
            (probes.advantage_available, "advantage_available", (), jnp.bool_),
            (
                probes.action_surprisal_available,
                "action_surprisal_available",
                (),
                jnp.bool_,
            ),
            (probes.safety_cost_available, "safety_cost_available", (), jnp.bool_),
        )
        for value, name, shape, dtype in checks:
            _require_array(value, name=name, shape=shape, dtype=dtype)

    def _source_valid(
        self,
        state: ExternalLearnedStateRouterAuditCoordinatorState,
    ) -> Array:
        return (
            state.started
            & self._builder.state_valid(state.builder_state)
            & jnp.array_equal(
                state.cached_builder_step_words,
                state.builder_state.step_words,
            )
            & jnp.all(jnp.isfinite(state.current_raw_observation))
            & jnp.all(jnp.isfinite(state.current_representation))
            & (state.current_action >= 0)
            & (state.current_action < self._config.n_actions)
            & (state.event_count >= 0)
        )

    @staticmethod
    def _identity_matches(
        state: ExternalLearnedStateRouterAuditCoordinatorState,
        probes: ExternalBuilderRepresentationProbeEvidence,
    ) -> Array:
        return (
            jnp.array_equal(probes.source_event_words, state.event_words)
            & jnp.array_equal(
                probes.source_builder_step_words,
                state.cached_builder_step_words,
            )
            & jnp.array_equal(
                probes.source_prototype_step_words,
                state.cached_prototype_step_words,
            )
            & jnp.array_equal(
                probes.source_feature_generation_words,
                state.cached_feature_generation_words,
            )
            & jnp.array_equal(probes.decision_id, state.current_decision_id)
        )

    def produce(
        self,
        state: ExternalLearnedStateRouterAuditCoordinatorState,
        probes: ExternalBuilderRepresentationProbeEvidence,
    ) -> ExternalBuilderCandidateEvidenceProducerResult:
        """Bind and pull back one independent-probe bundle without learning."""

        self._validate_source_static(state)
        self._validate_probe_static(probes)
        source_valid = self._source_valid(state)
        identity_matches = self._identity_matches(state, probes)
        source_gate = source_valid & identity_matches

        gradients = (
            probes.objective_representation_gradient,
            probes.retention_representation_gradient,
            probes.safety_representation_gradient,
        )
        gradient_validity = tuple(jnp.all(jnp.isfinite(value)) for value in gradients)
        hidden_gradients = tuple(value[-self._config.hidden_dim :] for value in gradients)
        sensitivity = state.builder_state.parameter_sensitivity
        raw_parameter_gradients = tuple(
            sensitivity.T @ hidden_gradient for hidden_gradient in hidden_gradients
        )
        parameter_gradient_validity = tuple(
            input_valid & jnp.all(jnp.isfinite(value))
            for input_valid, value in zip(
                gradient_validity,
                raw_parameter_gradients,
                strict=True,
            )
        )
        probe_pullbacks_valid = source_gate
        for valid in parameter_gradient_validity:
            probe_pullbacks_valid = probe_pullbacks_valid & valid

        zero_parameters = jnp.zeros(
            (self._config.parameter_count(),), dtype=jnp.float32
        )
        declarations = (
            probes.objective_probe_available,
            probes.retention_probe_available,
            probes.safety_probe_available,
        )
        availability = tuple(
            source_gate & declared & valid
            for declared, valid in zip(
                declarations,
                parameter_gradient_validity,
                strict=True,
            )
        )
        safe_parameter_gradients = tuple(
            jax.lax.stop_gradient(jnp.where(available, value, zero_parameters))
            for available, value in zip(
                availability,
                raw_parameter_gradients,
                strict=True,
            )
        )

        scalar_values = (probes.advantage, probes.action_surprisal, probes.safety_cost)
        scalar_declarations = (
            probes.advantage_available,
            probes.action_surprisal_available,
            probes.safety_cost_available,
        )
        scalar_finite = tuple(jnp.isfinite(value) for value in scalar_values)
        scalar_availability = tuple(
            source_gate & declared & finite
            for declared, finite in zip(
                scalar_declarations,
                scalar_finite,
                strict=True,
            )
        )
        safe_scalars = tuple(
            jax.lax.stop_gradient(
                jnp.where(available, value, jnp.asarray(0.0, dtype=jnp.float32))
            )
            for available, value in zip(
                scalar_availability,
                scalar_values,
                strict=True,
            )
        )
        scalar_channels_valid = source_gate
        for finite in scalar_finite:
            scalar_channels_valid = scalar_channels_valid & finite

        independence = (
            source_gate
            & probes.probe_independence_attested
            & parameter_gradient_validity[0]
            & parameter_gradient_validity[1]
            & parameter_gradient_validity[2]
        )
        evidence_ready = independence
        for available in (*availability, *scalar_availability):
            evidence_ready = evidence_ready & available

        evidence = ExternalBuilderCandidateAuditEvidence(
            source_event_words=state.event_words,
            source_builder_step_words=state.cached_builder_step_words,
            source_prototype_step_words=state.cached_prototype_step_words,
            source_feature_generation_words=state.cached_feature_generation_words,
            decision_id=state.current_decision_id,
            objective_probe_gradient=safe_parameter_gradients[0],
            retention_probe_gradient=safe_parameter_gradients[1],
            safety_cost_gradient=safe_parameter_gradients[2],
            objective_probe_available=availability[0],
            retention_probe_available=availability[1],
            safety_probe_available=availability[2],
            probe_independence_attested=independence,
            advantage=safe_scalars[0],
            action_surprisal=safe_scalars[1],
            safety_cost=safe_scalars[2],
            advantage_available=scalar_availability[0],
            action_surprisal_available=scalar_availability[1],
            safety_cost_available=scalar_availability[2],
        )
        return ExternalBuilderCandidateEvidenceProducerResult(
            evidence=evidence,
            diagnostics=ExternalBuilderCandidateEvidenceProducerDiagnostics(
                source_state_valid=source_valid,
                source_identity_matches=identity_matches,
                objective_probe_gradient_valid=parameter_gradient_validity[0],
                retention_probe_gradient_valid=parameter_gradient_validity[1],
                safety_probe_gradient_valid=parameter_gradient_validity[2],
                probe_pullbacks_valid=probe_pullbacks_valid,
                scalar_channels_valid=scalar_channels_valid,
                evidence_ready=evidence_ready,
                analytic_probe_pullback_evaluations=jnp.asarray(3, dtype=jnp.int32),
                additional_model_forward_evaluations=jnp.asarray(0, dtype=jnp.int32),
                actor_backward_evaluations=jnp.asarray(0, dtype=jnp.int32),
                delight_or_actor_backward=jnp.asarray(False, dtype=jnp.bool_),
            ),
        )


__all__ = [
    "EXTERNAL_BUILDER_CANDIDATE_EVIDENCE_PRODUCER_SCHEMA",
    "ExternalBuilderCandidateEvidenceProducer",
    "ExternalBuilderCandidateEvidenceProducerDiagnostics",
    "ExternalBuilderCandidateEvidenceProducerResourceBudget",
    "ExternalBuilderCandidateEvidenceProducerResult",
    "ExternalBuilderRepresentationProbeEvidence",
]
