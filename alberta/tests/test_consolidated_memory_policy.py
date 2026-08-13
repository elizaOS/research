# mypy: disable-error-code="arg-type,call-arg,type-var"
"""Unit contracts for the consolidated procedural-memory policy boundary."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from alberta_framework.core.consolidated_memory import (
    ProceduralMemoryRetrieval,
    canonical_memory_digest,
)
from alberta_framework.core.consolidated_memory_policy import (
    CONSOLIDATED_PROCEDURAL_MEMORY_POLICY_ACTION_DISPATCH_AUTHORITY,
    CONSOLIDATED_PROCEDURAL_MEMORY_POLICY_AGENT_MUTATION_AUTHORITY,
    CONSOLIDATED_PROCEDURAL_MEMORY_POLICY_CHECKPOINT_REQUIRED,
    CONSOLIDATED_PROCEDURAL_MEMORY_POLICY_MEMORY_MUTATION_AUTHORITY,
    CONSOLIDATED_PROCEDURAL_MEMORY_POLICY_PROMOTION_AUTHORITY,
    CONSOLIDATED_PROCEDURAL_MEMORY_POLICY_SCIENTIFIC_PROMOTION_ALLOWED,
    CONSOLIDATED_PROCEDURAL_MEMORY_POLICY_STATELESS,
    POLICY_REASON_COMPATIBILITY,
    POLICY_REASON_COUNT_INCONSISTENCY,
    POLICY_REASON_EVIDENCE,
    POLICY_REASON_LIFECYCLE,
    POLICY_REASON_NO_SAFE_ACTION,
    POLICY_REASON_NO_SAFE_POSITIVE_MASS,
    POLICY_REASON_OUTCOME,
    POLICY_REASON_RETRIEVAL_UNAVAILABLE,
    POLICY_REASON_SCORE_MASS,
    POLICY_REASON_SUCCESS_BOUND,
    POLICY_REASON_UNCERTAINTY,
    ConsolidatedProceduralMemoryPolicy,
    ConsolidatedProceduralMemoryPolicyConfig,
    ConsolidatedProceduralMemoryProposal,
)

pytestmark = pytest.mark.unit


def _digest(text: str) -> jax.Array:
    return canonical_memory_digest("test.consolidated-memory-policy", text)


def _config() -> ConsolidatedProceduralMemoryPolicyConfig:
    return ConsolidatedProceduralMemoryPolicyConfig(
        n_actions=4,
        outcome_dim=1,
        min_evidence_count=5,
        min_success_lower_bound=0.5,
        wilson_z=1.0,
        max_outcome_standard_error=0.5,
        max_abs_outcome_mean=10.0,
    )


def _policy() -> ConsolidatedProceduralMemoryPolicy:
    return ConsolidatedProceduralMemoryPolicy(_config())


def _retrieval(
    *,
    payload: tuple[float, float, float, float] = (0.2, 0.9, 0.9, 0.0),
    accepted: bool = True,
    transaction_applied: bool = True,
    compatible: bool = True,
    fresh: bool = True,
    evidence_count: int = 10,
    success_count: int = 8,
    failure_count: int = 2,
    outcome_mean: float = 1.0,
    outcome_m2: float = 0.9,
    lifecycle: str = "option-a",
    lifecycle_generation: int = 2,
    lifecycle_revision: int = 3,
) -> ProceduralMemoryRetrieval:
    if not accepted:
        return ProceduralMemoryRetrieval(
            accepted=jnp.asarray(False, dtype=jnp.bool_),
            transaction_applied=jnp.asarray(transaction_applied, dtype=jnp.bool_),
            slot=jnp.asarray(-1, dtype=jnp.int32),
            payload=jnp.zeros((4,), dtype=jnp.float32),
            confidence=jnp.asarray(0.0, dtype=jnp.float32),
            evidence_count=jnp.asarray(0, dtype=jnp.int32),
            evidence_mean=jnp.asarray(0.0, dtype=jnp.float32),
            evidence_m2=jnp.asarray(0.0, dtype=jnp.float32),
            success_count=jnp.asarray(0, dtype=jnp.int32),
            failure_count=jnp.asarray(0, dtype=jnp.int32),
            outcome_mean=jnp.zeros((1,), dtype=jnp.float32),
            outcome_m2=jnp.zeros((1,), dtype=jnp.float32),
            lifecycle_link_available=jnp.asarray(False, dtype=jnp.bool_),
            lifecycle_digest=jnp.zeros((32,), dtype=jnp.uint8),
            lifecycle_generation=jnp.asarray(-1, dtype=jnp.int32),
            lifecycle_revision=jnp.asarray(-1, dtype=jnp.int32),
            state_valid=jnp.asarray(True, dtype=jnp.bool_),
            request_valid=jnp.asarray(True, dtype=jnp.bool_),
            identity_found=jnp.asarray(False, dtype=jnp.bool_),
            compatible=jnp.asarray(False, dtype=jnp.bool_),
            fresh=jnp.asarray(False, dtype=jnp.bool_),
            confidence_ok=jnp.asarray(False, dtype=jnp.bool_),
        )
    return ProceduralMemoryRetrieval(
        accepted=jnp.asarray(True, dtype=jnp.bool_),
        transaction_applied=jnp.asarray(transaction_applied, dtype=jnp.bool_),
        slot=jnp.asarray(1, dtype=jnp.int32),
        payload=jnp.asarray(payload, dtype=jnp.float32),
        confidence=jnp.asarray(0.9, dtype=jnp.float32),
        evidence_count=jnp.asarray(evidence_count, dtype=jnp.int32),
        evidence_mean=jnp.asarray(1.0, dtype=jnp.float32),
        evidence_m2=jnp.asarray(0.2, dtype=jnp.float32),
        success_count=jnp.asarray(success_count, dtype=jnp.int32),
        failure_count=jnp.asarray(failure_count, dtype=jnp.int32),
        outcome_mean=jnp.asarray([outcome_mean], dtype=jnp.float32),
        outcome_m2=jnp.asarray([outcome_m2], dtype=jnp.float32),
        lifecycle_link_available=jnp.asarray(True, dtype=jnp.bool_),
        lifecycle_digest=_digest(lifecycle),
        lifecycle_generation=jnp.asarray(lifecycle_generation, dtype=jnp.int32),
        lifecycle_revision=jnp.asarray(lifecycle_revision, dtype=jnp.int32),
        state_valid=jnp.asarray(True, dtype=jnp.bool_),
        request_valid=jnp.asarray(True, dtype=jnp.bool_),
        identity_found=jnp.asarray(True, dtype=jnp.bool_),
        compatible=jnp.asarray(compatible, dtype=jnp.bool_),
        fresh=jnp.asarray(fresh, dtype=jnp.bool_),
        confidence_ok=jnp.asarray(True, dtype=jnp.bool_),
    )


def _propose(
    policy: ConsolidatedProceduralMemoryPolicy,
    retrieval: ProceduralMemoryRetrieval,
    *,
    mask: tuple[bool, bool, bool, bool] = (True, True, True, True),
    lifecycle: str = "option-a",
    generation: int = 2,
    revision: int = 3,
) -> ConsolidatedProceduralMemoryProposal:
    return policy.propose(
        retrieval,
        hard_safety_mask=jnp.asarray(mask, dtype=jnp.bool_),
        expected_lifecycle_digest=_digest(lifecycle),
        expected_lifecycle_generation=generation,
        expected_lifecycle_revision=revision,
    )


def test_safe_positive_argmax_is_deterministic_and_lowest_index_wins_ties() -> None:
    proposal = _propose(_policy(), _retrieval())
    assert bool(proposal.available)
    assert int(proposal.action) == 1
    assert float(proposal.selected_mass) == pytest.approx(0.9)
    assert float(proposal.safe_mass_total) == pytest.approx(2.0)
    assert proposal.safe_positive_mask.tolist() == [True, True, True, False]

    unsafe_highest = _propose(
        _policy(), _retrieval(payload=(0.2, 10.0, 0.8, 0.1)), mask=(True, False, True, True)
    )
    assert bool(unsafe_highest.available)
    assert int(unsafe_highest.action) == 2


def test_unavailable_stale_incompatible_and_lifecycle_mismatch_fail_closed() -> None:
    policy = _policy()
    unavailable = _propose(policy, _retrieval(accepted=False))
    assert not bool(unavailable.available)
    assert int(unavailable.action) == -1
    assert int(unavailable.diagnostics.unavailable_reason) == (POLICY_REASON_RETRIEVAL_UNAVAILABLE)

    stale = _propose(policy, _retrieval(fresh=False))
    assert not bool(stale.available)
    assert not bool(stale.diagnostics.retrieval_fresh)
    assert int(stale.diagnostics.unavailable_reason) == POLICY_REASON_COMPATIBILITY

    incompatible = _propose(policy, _retrieval(compatible=False))
    assert not bool(incompatible.available)
    assert int(incompatible.diagnostics.unavailable_reason) == POLICY_REASON_COMPATIBILITY

    wrong_lifecycle = _propose(policy, _retrieval(), lifecycle="option-b")
    assert not bool(wrong_lifecycle.available)
    assert not bool(wrong_lifecycle.diagnostics.lifecycle_matches)
    assert int(wrong_lifecycle.diagnostics.unavailable_reason) == POLICY_REASON_LIFECYCLE


def test_evidence_count_consistency_and_conservative_success_bound_are_separate() -> None:
    policy = _policy()
    low_evidence = _propose(
        policy,
        _retrieval(evidence_count=4, success_count=4, failure_count=0, outcome_m2=0.0),
    )
    assert not bool(low_evidence.available)
    assert bool(low_evidence.diagnostics.counts_consistent)
    assert not bool(low_evidence.diagnostics.evidence_ready)
    assert int(low_evidence.diagnostics.unavailable_reason) == POLICY_REASON_EVIDENCE

    inconsistent = _propose(policy, _retrieval(evidence_count=10, success_count=8, failure_count=1))
    assert not bool(inconsistent.available)
    assert not bool(inconsistent.diagnostics.counts_consistent)
    assert int(inconsistent.diagnostics.unavailable_reason) == (POLICY_REASON_COUNT_INCONSISTENCY)

    low_success = _propose(policy, _retrieval(evidence_count=10, success_count=2, failure_count=8))
    assert not bool(low_success.available)
    assert bool(low_success.diagnostics.evidence_ready)
    assert float(low_success.diagnostics.success_lower_bound) < 0.5
    assert int(low_success.diagnostics.unavailable_reason) == POLICY_REASON_SUCCESS_BOUND


def test_nonfinite_outcome_uncertainty_and_score_mass_are_rejected() -> None:
    policy = _policy()
    nonfinite_outcome = _propose(policy, _retrieval(outcome_mean=float("nan")))
    assert not bool(nonfinite_outcome.available)
    assert not bool(nonfinite_outcome.diagnostics.outcome_finite)
    assert int(nonfinite_outcome.diagnostics.unavailable_reason) == POLICY_REASON_OUTCOME

    uncertain = _propose(policy, _retrieval(outcome_m2=100.0))
    assert not bool(uncertain.available)
    assert not bool(uncertain.diagnostics.uncertainty_ready)
    assert int(uncertain.diagnostics.unavailable_reason) == POLICY_REASON_UNCERTAINTY

    negative_mass = _propose(policy, _retrieval(payload=(0.2, -0.1, 0.9, 0.0)))
    assert not bool(negative_mass.available)
    assert not bool(negative_mass.diagnostics.score_mass_nonnegative)
    assert int(negative_mass.diagnostics.unavailable_reason) == POLICY_REASON_SCORE_MASS

    nan_mass = _propose(policy, _retrieval(payload=(0.2, float("nan"), 0.9, 0.0)))
    assert not bool(nan_mass.available)
    assert not bool(nan_mass.diagnostics.score_mass_finite)
    assert int(nan_mass.diagnostics.unavailable_reason) == POLICY_REASON_SCORE_MASS


def test_hard_safety_distinguishes_no_safe_action_from_no_safe_positive_mass() -> None:
    policy = _policy()
    no_safe = _propose(policy, _retrieval(), mask=(False, False, False, False))
    assert not bool(no_safe.available)
    assert not bool(no_safe.diagnostics.any_safe_action)
    assert int(no_safe.diagnostics.unavailable_reason) == POLICY_REASON_NO_SAFE_ACTION

    zero_safe_mass = _propose(
        policy,
        _retrieval(payload=(1.0, 0.0, 0.0, 0.0)),
        mask=(False, True, True, True),
    )
    assert not bool(zero_safe_mass.available)
    assert bool(zero_safe_mass.diagnostics.any_safe_action)
    assert not bool(zero_safe_mass.diagnostics.any_safe_positive_mass)
    assert int(zero_safe_mass.diagnostics.unavailable_reason) == (
        POLICY_REASON_NO_SAFE_POSITIVE_MASS
    )


def test_stateless_identity_config_resources_and_authority_are_exact() -> None:
    config = _config()
    policy = ConsolidatedProceduralMemoryPolicy(config)
    restored = ConsolidatedProceduralMemoryPolicy.from_config(config.to_config())
    assert restored.identity_sha256 == policy.identity_sha256
    assert len(policy.identity_sha256) == 64
    assert config.to_config()["stateless"] is True
    assert config.to_config()["checkpoint_required"] is False
    budget = policy.resource_budget
    assert budget.persistent_logical_scalars == 0
    assert budget.persistent_state_bytes == 0
    assert budget.checkpoint_bytes == 0
    assert budget.score_cells_scanned_per_proposal == 4
    assert budget.outcome_cells_scanned_per_proposal == 1
    assert budget.wilson_square_roots_per_proposal == 1
    assert budget.deterministic_argmax_calls_per_proposal == 1
    assert budget.random_generator_calls_per_proposal == 0
    assert budget.memory_queries_per_proposal == 0
    assert budget.memory_writes_per_proposal == 0
    assert budget.action_dispatches_per_proposal == 0
    assert budget.agent_parameter_mutations_per_proposal == 0
    assert budget.proposal_authority
    assert not budget.action_dispatch_authority
    assert not budget.memory_mutation_authority
    assert not budget.agent_mutation_authority
    assert not budget.promotion_authority
    assert not budget.scientific_promotion_allowed
    assert not budget.checkpoint_required
    assert CONSOLIDATED_PROCEDURAL_MEMORY_POLICY_STATELESS
    assert not CONSOLIDATED_PROCEDURAL_MEMORY_POLICY_CHECKPOINT_REQUIRED
    assert not CONSOLIDATED_PROCEDURAL_MEMORY_POLICY_ACTION_DISPATCH_AUTHORITY
    assert not CONSOLIDATED_PROCEDURAL_MEMORY_POLICY_MEMORY_MUTATION_AUTHORITY
    assert not CONSOLIDATED_PROCEDURAL_MEMORY_POLICY_AGENT_MUTATION_AUTHORITY
    assert not CONSOLIDATED_PROCEDURAL_MEMORY_POLICY_PROMOTION_AUTHORITY
    assert not CONSOLIDATED_PROCEDURAL_MEMORY_POLICY_SCIENTIFIC_PROMOTION_ALLOWED


def test_static_shape_and_dtype_tampering_fails_before_compilation() -> None:
    policy = _policy()
    bad_payload = dataclasses.replace(_retrieval(), payload=jnp.ones((5,), dtype=jnp.float32))
    with pytest.raises(ValueError, match="retrieval.payload"):
        _propose(policy, bad_payload)
    with pytest.raises(TypeError, match="hard_safety_mask"):
        policy.propose(
            _retrieval(),
            hard_safety_mask=jnp.ones((4,), dtype=jnp.int32),
            expected_lifecycle_digest=_digest("option-a"),
            expected_lifecycle_generation=2,
            expected_lifecycle_revision=3,
        )
