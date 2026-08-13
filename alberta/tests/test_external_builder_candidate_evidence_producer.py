# mypy: disable-error-code="attr-defined,call-arg,type-var"
"""Source-bound candidate evidence for the external full-GRU owner."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.external_builder_candidate_evidence_producer import (
    ExternalBuilderCandidateEvidenceProducer,
    ExternalBuilderRepresentationProbeEvidence,
)
from alberta_framework.core.external_learned_state_router_audit_coordinator import (
    ExternalLearnedStateRouterAuditCoordinatorState,
)
from alberta_framework.core.state_builder import (
    LearnableGRUStateBuilder,
    LearnableGRUStateBuilderConfig,
)

pytestmark = pytest.mark.unit


def _source() -> tuple[
    LearnableGRUStateBuilderConfig,
    ExternalLearnedStateRouterAuditCoordinatorState,
]:
    config = LearnableGRUStateBuilderConfig(
        observation_dim=2,
        n_actions=2,
        hidden_dim=2,
        step_size=0.01,
        gradient_clip=10.0,
        initialization_scale=0.2,
        include_raw_observation=True,
    )
    builder = LearnableGRUStateBuilder(config)
    builder_state = builder.init(jr.key(7))
    builder_state, representation = builder.start(
        builder_state,
        jnp.asarray((0.25, -0.5), dtype=jnp.float32),
    )
    decision_id = jnp.asarray((3, 5, 7, 11), dtype=jnp.uint32)
    state = ExternalLearnedStateRouterAuditCoordinatorState(
        builder_state=builder_state,
        inner_state=None,  # The stateless producer owns or inspects no inner state.
        learning_value_router_state=None,
        current_raw_observation=jnp.asarray((0.25, -0.5), dtype=jnp.float32),
        current_representation=representation,
        current_action=jnp.asarray(1, dtype=jnp.int32),
        current_decision_id=decision_id,
        cached_builder_step_words=builder_state.step_words,
        cached_prototype_step_words=jnp.asarray((0, 13), dtype=jnp.uint32),
        cached_feature_generation_words=jnp.asarray((0, 17), dtype=jnp.uint32),
        event_count=jnp.asarray(19, dtype=jnp.int32),
        event_words=jnp.asarray((0, 19), dtype=jnp.uint32),
        started=jnp.asarray(True, dtype=jnp.bool_),
        schema_digest=jnp.zeros((32,), dtype=jnp.uint8),
    )
    return config, state


def _probes(
    state: ExternalLearnedStateRouterAuditCoordinatorState,
) -> ExternalBuilderRepresentationProbeEvidence:
    true = jnp.asarray(True, dtype=jnp.bool_)
    return ExternalBuilderRepresentationProbeEvidence(
        source_event_words=state.event_words,
        source_builder_step_words=state.cached_builder_step_words,
        source_prototype_step_words=state.cached_prototype_step_words,
        source_feature_generation_words=state.cached_feature_generation_words,
        decision_id=state.current_decision_id,
        objective_representation_gradient=jnp.asarray(
            (0.2, -0.1, 0.7, -0.4), dtype=jnp.float32
        ),
        retention_representation_gradient=jnp.asarray(
            (-0.3, 0.6, -0.2, 0.5), dtype=jnp.float32
        ),
        safety_representation_gradient=jnp.asarray(
            (0.4, 0.8, -0.9, 0.1), dtype=jnp.float32
        ),
        objective_probe_available=true,
        retention_probe_available=true,
        safety_probe_available=true,
        probe_independence_attested=true,
        advantage=jnp.asarray(0.75, dtype=jnp.float32),
        action_surprisal=jnp.asarray(0.5, dtype=jnp.float32),
        safety_cost=jnp.asarray(0.125, dtype=jnp.float32),
        advantage_available=true,
        action_surprisal_available=true,
        safety_cost_available=true,
    )


def test_exact_source_pullback_forms_candidate_audit_evidence_without_joy_claim() -> None:
    config, state = _source()
    producer = ExternalBuilderCandidateEvidenceProducer(config)
    probes = _probes(state)

    payload = producer.to_config()
    assert ExternalBuilderCandidateEvidenceProducer.from_config(
        payload
    ).to_config() == payload
    budget = producer.resource_budget
    assert budget.persistent_state_bytes == 0
    assert budget.full_gru_owner_count == 0
    assert budget.candidate_audit_owner_count == 0
    assert budget.analytic_probe_pullbacks_per_event == 3
    assert budget.actor_backward_evaluations_per_event == 0
    assert not budget.delight_or_actor_backward

    result = producer.produce(state, probes)
    compiled = jax.jit(producer.produce)(state, probes)
    sensitivity = state.builder_state.parameter_sensitivity
    hidden = config.hidden_dim
    np.testing.assert_allclose(
        result.evidence.objective_probe_gradient,
        sensitivity.T @ probes.objective_representation_gradient[-hidden:],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.evidence.retention_probe_gradient,
        sensitivity.T @ probes.retention_representation_gradient[-hidden:],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.evidence.safety_cost_gradient,
        sensitivity.T @ probes.safety_representation_gradient[-hidden:],
        rtol=0.0,
        atol=0.0,
    )
    assert bool(result.diagnostics.source_state_valid)
    assert bool(result.diagnostics.source_identity_matches)
    assert bool(result.diagnostics.probe_pullbacks_valid)
    assert bool(result.diagnostics.evidence_ready)
    assert int(result.diagnostics.analytic_probe_pullback_evaluations) == 3
    assert int(result.diagnostics.additional_model_forward_evaluations) == 0
    assert int(result.diagnostics.actor_backward_evaluations) == 0
    assert not bool(result.diagnostics.delight_or_actor_backward)
    assert not hasattr(result, "sparks_joy")
    assert not hasattr(result.evidence, "sparks_joy")
    np.testing.assert_array_equal(
        compiled.evidence.objective_probe_gradient,
        result.evidence.objective_probe_gradient,
    )
    np.testing.assert_array_equal(
        compiled.diagnostics.evidence_ready,
        result.diagnostics.evidence_ready,
    )


def test_stale_or_nonfinite_probes_fail_closed_to_unavailable_zero_gradients() -> None:
    config, state = _source()
    producer = ExternalBuilderCandidateEvidenceProducer(config)
    probes = _probes(state)
    stale = dataclasses.replace(
        probes,
        decision_id=probes.decision_id.at[3].add(jnp.asarray(1, dtype=jnp.uint32)),
        objective_representation_gradient=probes.objective_representation_gradient.at[
            0
        ].set(jnp.asarray(jnp.nan, dtype=jnp.float32)),
    )

    result = producer.produce(state, stale)
    assert not bool(result.diagnostics.source_identity_matches)
    assert not bool(result.diagnostics.probe_pullbacks_valid)
    assert not bool(result.diagnostics.evidence_ready)
    assert not bool(result.evidence.objective_probe_available)
    assert not bool(result.evidence.retention_probe_available)
    assert not bool(result.evidence.safety_probe_available)
    assert not bool(result.evidence.probe_independence_attested)
    np.testing.assert_array_equal(
        result.evidence.objective_probe_gradient,
        jnp.zeros((config.parameter_count(),), dtype=jnp.float32),
    )
    np.testing.assert_array_equal(
        result.evidence.retention_probe_gradient,
        jnp.zeros((config.parameter_count(),), dtype=jnp.float32),
    )
    np.testing.assert_array_equal(
        result.evidence.safety_cost_gradient,
        jnp.zeros((config.parameter_count(),), dtype=jnp.float32),
    )


def test_static_contract_rejects_shape_drift_and_nonexact_inputs() -> None:
    config, state = _source()
    producer = ExternalBuilderCandidateEvidenceProducer(config)
    probes = _probes(state)

    with pytest.raises(ValueError, match="objective_representation_gradient"):
        producer.produce(
            state,
            dataclasses.replace(
                probes,
                objective_representation_gradient=jnp.zeros((3,), dtype=jnp.float32),
            ),
        )
    with pytest.raises(TypeError, match="exact ExternalBuilderRepresentationProbeEvidence"):
        producer.produce(state, object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact LearnableGRUStateBuilderConfig"):
        ExternalBuilderCandidateEvidenceProducer(object())  # type: ignore[arg-type]
