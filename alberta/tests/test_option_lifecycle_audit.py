# mypy: disable-error-code="attr-defined,call-arg,type-var"
"""Unit contracts for the bounded, authority-free WP7.3 lifecycle auditor."""

from __future__ import annotations

import copy
import dataclasses
import json
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.option_lifecycle_audit import (
    OPTION_LIFECYCLE_AUDIT_CURATION_AUTHORITY,
    OPTION_LIFECYCLE_AUDIT_GO_NO_GO_AUTHORITY,
    OPTION_LIFECYCLE_AUDIT_PROMOTION_AUTHORITY,
    OPTION_LIFECYCLE_AUDIT_SCIENTIFIC_PROMOTION_ALLOWED,
    OptionLifecycleAudit,
    OptionLifecycleAuditArm,
    OptionLifecycleAuditConfig,
    OptionLifecycleAuditResult,
    OptionLifecycleAuditState,
    option_semantic_digest,
)

pytestmark = pytest.mark.unit

N_OPTIONS = 2
N_CONTEXTS = 2
OUTCOME_DIM = 1
SIGNATURE_DIM = 6
SOURCE = jnp.arange(1, 9, dtype=jnp.uint32)
REPRESENTATION = jnp.arange(21, 29, dtype=jnp.uint32)
SEMANTICS = jnp.stack(
    (option_semantic_digest({"option": 0}), option_semantic_digest({"option": 1}))
)


def _config(**changes: Any) -> OptionLifecycleAuditConfig:
    values: dict[str, Any] = {
        "n_options": N_OPTIONS,
        "n_contexts": N_CONTEXTS,
        "outcome_dim": OUTCOME_DIM,
        "fixed_horizon": 2,
        "maintenance_budget": 1,
        "signature_scales": (1.0,) * SIGNATURE_DIM,
        "initiation_opportunity_floor": 1,
        "completion_evidence_floor": 1,
        "model_error_evidence_floor": 1,
        "comparison_treatment_evidence_floor": 1,
        "comparison_primitive_evidence_floor": 1,
        "signature_evidence_floor_per_context": 1,
        "redundancy_shared_context_floor": 1,
        "max_observations": 32,
        "max_planning_uses_per_observation": 8,
        "max_compute_cost_per_observation": 100.0,
        "max_mean_compute_cost": 50.0,
        "max_resident_memory_bytes": 1_000,
    }
    values.update(changes)
    return OptionLifecycleAuditConfig(**values)


def _new(**config_changes: Any) -> tuple[OptionLifecycleAudit, OptionLifecycleAuditState]:
    audit = OptionLifecycleAudit(_config(**config_changes))
    state = audit.init(
        source_digest=SOURCE,
        representation_digest=REPRESENTATION,
        semantic_digests=SEMANTICS,
    )
    return audit, state


def _arm(
    audit: OptionLifecycleAudit,
    state: OptionLifecycleAuditState,
    step: int,
    *,
    option: int = 0,
    context: int = 0,
    eligible: bool = False,
    owner: int = -1,
    randomized: bool = False,
    propensity: float = 0.0,
    prediction: jax.Array | None = None,
) -> OptionLifecycleAuditArm:
    return audit.arm(
        state,
        transition_id=jnp.asarray([0xA17E, step], dtype=jnp.uint32),
        source_digest=SOURCE,
        representation_digest=REPRESENTATION,
        semantic_digests=state.semantic_digests,
        semantic_generations=state.semantic_generations,
        candidate_option=option,
        initiation_context=context,
        initiation_eligible=eligible,
        owner_option=owner,
        comparator_randomized=randomized,
        treatment_propensity=propensity,
        frozen_model_prediction=(
            jnp.zeros((SIGNATURE_DIM,), dtype=jnp.float32)
            if prediction is None
            else prediction
        ),
    )


def _observe(
    audit: OptionLifecycleAudit,
    state: OptionLifecycleAuditState,
    arm: OptionLifecycleAuditArm,
    step: int,
    *,
    reward: float = 1.0,
    pseudo: float = 0.5,
    baseline: float = 1.0,
    discount: float = 0.9,
    outcome: float = 0.25,
    goal: bool = False,
    timeout: bool = False,
    environment: bool = False,
    censored: bool = False,
    planning: tuple[int, int] = (0, 0),
    cost: float = 0.0,
    memory: int = 0,
) -> OptionLifecycleAuditResult:
    return audit.observe(
        state,
        arm,
        transition_id=jnp.asarray([0xA17E, step], dtype=jnp.uint32),
        external_reward=reward,
        pseudo_reward=pseudo,
        baseline_mass=baseline,
        discount=discount,
        outcome_delta=jnp.asarray([outcome], dtype=jnp.float32),
        goal_terminated=goal,
        timeout_terminated=timeout,
        environment_terminated=environment,
        censored=censored,
        planning_usage_delta=jnp.asarray(planning, dtype=jnp.int32),
        compute_cost=cost,
        resident_memory_bytes=memory,
    )


def test_config_schema_ceilings_and_authority_are_strict() -> None:
    config = _config()
    restored = OptionLifecycleAuditConfig.from_config(
        json.loads(json.dumps(config.to_config()))
    )
    assert restored == config
    payload = config.to_config()
    payload["scientific_promotion_allowed"] = True
    with pytest.raises(ValueError, match="cannot claim"):
        OptionLifecycleAuditConfig.from_config(payload)
    with pytest.raises(ValueError, match="exact Python int"):
        _config(n_options=True)
    with pytest.raises(ValueError, match="exactly"):
        _config(signature_scales=(1.0,))
    with pytest.raises(ValueError, match="n_options"):
        _config(n_options=1_025)
    assert OPTION_LIFECYCLE_AUDIT_CURATION_AUTHORITY is False
    assert OPTION_LIFECYCLE_AUDIT_PROMOTION_AUTHORITY is False
    assert OPTION_LIFECYCLE_AUDIT_GO_NO_GO_AUTHORITY is False
    assert OPTION_LIFECYCLE_AUDIT_SCIENTIFIC_PROMOTION_ALLOWED is False

    budget = OptionLifecycleAudit(config).resource_budget
    assert budget.rng_draws_per_observe == 0
    assert budget.backward_passes_per_observe == 0
    assert budget.consumer_calls_per_observe == 0
    assert budget.option_updates_per_observe == 0
    assert budget.curation_authority is False


def test_init_rejects_ambiguous_semantics_and_has_frozen_arrays() -> None:
    audit, state = _new()
    chex.assert_tree_all_finite(state)
    assert bool(audit.maintenance_report(state).state_valid)
    assert state.semantic_digests.dtype == jnp.uint32
    duplicate = jnp.repeat(SEMANTICS[:1], N_OPTIONS, axis=0)
    with pytest.raises(ValueError, match="unique"):
        audit.init(
            source_digest=SOURCE,
            representation_digest=REPRESENTATION,
            semantic_digests=duplicate,
        )
    with pytest.raises(TypeError, match="uint32"):
        audit.init(
            source_digest=SOURCE.astype(jnp.int32),
            representation_digest=REPRESENTATION,
            semantic_digests=SEMANTICS,
        )
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(state, "revision", jnp.asarray(4, dtype=jnp.int32))


def test_execution_records_reasons_returns_frozen_errors_usage_and_cost() -> None:
    audit, state = _new()
    prediction = jnp.asarray([3.0, 1.0, 2.0, 3.0, 0.5, 1.0], dtype=jnp.float32)
    first = _arm(
        audit,
        state,
        1,
        eligible=True,
        owner=0,
        prediction=prediction,
    )
    first_result = _observe(
        audit,
        state,
        first,
        1,
        reward=1.0,
        pseudo=0.25,
        baseline=1.0,
        discount=0.8,
        outcome=0.4,
        planning=(2, 1),
        cost=3.0,
        memory=200,
    )
    assert bool(first_result.applied)
    assert int(first_result.state.active_option) == 0

    # Continuations must not smuggle a new prediction into the frozen model audit.
    second = _arm(audit, first_result.state, 2, option=0, owner=0)
    result = _observe(
        audit,
        first_result.state,
        second,
        2,
        reward=2.0,
        pseudo=0.75,
        baseline=2.0,
        discount=0.5,
        outcome=0.6,
        goal=True,
        timeout=True,
        environment=True,
        censored=True,
        planning=(1, 0),
        cost=5.0,
        memory=300,
    )
    assert bool(result.execution_completed)
    assert not bool(result.censor_only_ending)
    state = result.state
    assert int(state.active_option) == -1
    assert int(state.goal_terminations[0]) == 1
    assert int(state.timeout_terminations[0]) == 1
    assert int(state.environment_terminations[0]) == 1
    assert int(state.censored_endings[0]) == 1
    assert int(state.censor_only_endings[0]) == 0
    expected = jnp.asarray([2.6, 1.0, 2.0, 3.0, 0.4, 1.0], dtype=jnp.float32)
    np.testing.assert_allclose(state.completion_signature_sums[0], expected)
    np.testing.assert_allclose(
        state.model_squared_error_sums[0],
        (expected - prediction) ** 2,
    )
    np.testing.assert_array_equal(state.planning_use_counts, jnp.asarray([3, 1]))
    assert float(state.compute_cost_sums[0]) == pytest.approx(8.0)
    assert int(state.resident_memory_max_bytes[0]) == 300


def test_censor_only_is_not_a_completion_or_model_target() -> None:
    audit, state = _new()
    arm = _arm(audit, state, 1, eligible=True, owner=0)
    result = _observe(audit, state, arm, 1, censored=True)
    assert bool(result.censor_only_ending)
    assert not bool(result.execution_completed)
    assert int(result.state.censor_only_endings[0]) == 1
    assert int(result.state.natural_completions[0]) == 0
    assert int(result.state.model_error_counts[0]) == 0
    assert int(result.state.completion_moment_counts[0]) == 0


def _finish_randomized_trial(
    audit: OptionLifecycleAudit,
    state: OptionLifecycleAuditState,
    start_step: int,
    *,
    context: int,
    treatment: bool,
    rewards: tuple[float, float],
) -> OptionLifecycleAuditState:
    first = _arm(
        audit,
        state,
        start_step,
        context=context,
        eligible=True,
        owner=0 if treatment else -1,
        randomized=True,
        propensity=0.25,
    )
    first_result = _observe(
        audit,
        state,
        first,
        start_step,
        reward=rewards[0],
        goal=treatment,
    )
    assert bool(first_result.state.trial_active)
    second = _arm(
        audit,
        first_result.state,
        start_step + 1,
        option=0,
        context=context,
        owner=-1,
    )
    second_result = _observe(
        audit,
        first_result.state,
        second,
        start_step + 1,
        reward=rewards[1],
    )
    assert bool(second_result.comparator_trial_completed)
    return second_result.state


def test_randomized_fixed_horizon_comparator_keeps_evidence_arms_separate() -> None:
    audit, state = _new()
    step = 1
    for context in range(N_CONTEXTS):
        state = _finish_randomized_trial(
            audit,
            state,
            step,
            context=context,
            treatment=True,
            rewards=(2.0, 3.0),
        )
        step += 2
        state = _finish_randomized_trial(
            audit,
            state,
            step,
            context=context,
            treatment=False,
            rewards=(0.5, 0.5),
        )
        step += 2
    np.testing.assert_array_equal(state.comparison_treatment_counts[0], [1, 1])
    np.testing.assert_array_equal(state.comparison_primitive_counts[0], [1, 1])
    np.testing.assert_allclose(state.comparison_treatment_return_sums[0], [5.0, 5.0])
    np.testing.assert_allclose(state.comparison_primitive_return_sums[0], [1.0, 1.0])
    report = audit.maintenance_report(state)
    assert bool(report.comparison_ready[0])
    assert float(report.marginal_improvement[0]) == pytest.approx(4.0)
    assert float(report.inverse_propensity_marginal_improvement[0]) == pytest.approx(4.0)


def test_missing_comparator_context_blocks_readiness_without_mass_renormalization() -> None:
    audit, state = _new()
    state = _finish_randomized_trial(
        audit,
        state,
        1,
        context=0,
        treatment=True,
        rewards=(2.0, 3.0),
    )
    state = _finish_randomized_trial(
        audit,
        state,
        3,
        context=0,
        treatment=False,
        rewards=(0.5, 0.5),
    )
    report = audit.maintenance_report(state)
    assert not bool(report.comparison_ready[0])
    np.testing.assert_allclose(report.marginal_improvement_by_context[0], [4.0, 0.0])
    np.testing.assert_allclose(
        report.inverse_propensity_marginal_improvement_by_context[0],
        [4.0, 0.0],
    )
    # Each configured context has fixed mass 1/2.  Context 0 cannot be
    # renormalized to full mass merely because context 1 has no evidence.
    assert float(report.marginal_improvement[0]) == pytest.approx(2.0)
    assert float(report.inverse_propensity_marginal_improvement[0]) == pytest.approx(2.0)


def test_observational_selection_and_bad_propensity_cannot_create_comparison_evidence() -> None:
    audit, state = _new()
    bad = _arm(
        audit,
        state,
        1,
        eligible=True,
        owner=0,
        randomized=True,
        propensity=0.001,
    )
    assert not bool(bad.available)
    result = _observe(audit, state, bad, 1, goal=True)
    chex.assert_trees_all_equal(result.state, state)
    assert not bool(result.applied)

    ordinary = _arm(audit, state, 2, eligible=True, owner=0)
    result = _observe(audit, state, ordinary, 2, goal=True)
    assert bool(result.applied)
    np.testing.assert_array_equal(result.state.comparison_treatment_counts[0], [0, 0])
    np.testing.assert_array_equal(result.state.comparison_primitive_counts[0], [0, 0])


@pytest.mark.parametrize("tamper", ["identity", "cache", "source", "revision", "nan"])
def test_two_phase_binding_and_runtime_failures_are_atomic(tamper: str) -> None:
    audit, state = _new()
    arm = _arm(audit, state, 1, eligible=True, owner=0)
    kwargs: dict[str, Any] = {}
    if tamper == "identity":
        kwargs["transition_id"] = jnp.asarray([0xA17E, 99], dtype=jnp.uint32)
    elif tamper == "cache":
        arm = dataclasses.replace(arm, owner_option=jnp.asarray(-1, dtype=jnp.int32))
    elif tamper == "source":
        arm = dataclasses.replace(arm, source_digest=arm.source_digest.at[0].add(1))
    elif tamper == "revision":
        arm = dataclasses.replace(arm, state_revision=arm.state_revision + 1)

    if tamper == "nan":
        result = audit.observe(
            state,
            arm,
            transition_id=jnp.asarray([0xA17E, 1], dtype=jnp.uint32),
            external_reward=jnp.asarray(jnp.nan, dtype=jnp.float32),
            pseudo_reward=0.0,
            baseline_mass=0.0,
            discount=1.0,
            outcome_delta=jnp.zeros((1,), dtype=jnp.float32),
            goal_terminated=False,
            timeout_terminated=False,
            environment_terminated=False,
            censored=False,
            planning_usage_delta=jnp.zeros((2,), dtype=jnp.int32),
            compute_cost=0.0,
            resident_memory_bytes=0,
        )
    elif "transition_id" in kwargs:
        result = audit.observe(
            state,
            arm,
            transition_id=kwargs["transition_id"],
            external_reward=1.0,
            pseudo_reward=0.0,
            baseline_mass=0.0,
            discount=1.0,
            outcome_delta=jnp.zeros((1,), dtype=jnp.float32),
            goal_terminated=False,
            timeout_terminated=False,
            environment_terminated=False,
            censored=False,
            planning_usage_delta=jnp.zeros((2,), dtype=jnp.int32),
            compute_cost=0.0,
            resident_memory_bytes=0,
        )
    else:
        result = _observe(audit, state, arm, 1)
    assert not bool(result.applied)
    chex.assert_trees_all_equal(result.state, state)


def test_rebind_preserves_identical_semantics_and_resets_changed_slot() -> None:
    audit, state = _new()
    arm = _arm(audit, state, 1, option=0, eligible=True, owner=0)
    state = _observe(audit, state, arm, 1, goal=True).state
    same = audit.rebind(
        state,
        source_digest=SOURCE,
        representation_digest=REPRESENTATION,
        semantic_digests=SEMANTICS,
    )
    assert not bool(same.applied)
    assert bool(jnp.all(same.preserved_slots))
    chex.assert_trees_all_equal(same.state, state)

    changed_semantics = SEMANTICS.at[0].set(option_semantic_digest({"option": "new"}))
    changed = audit.rebind(
        state,
        source_digest=SOURCE,
        representation_digest=REPRESENTATION,
        semantic_digests=changed_semantics,
    )
    assert bool(changed.applied)
    np.testing.assert_array_equal(changed.reset_slots, jnp.asarray([True, False]))
    assert int(changed.state.semantic_generations[0]) == 1
    assert int(changed.state.execution_starts[0]) == 0
    assert int(changed.state.execution_starts[1]) == int(state.execution_starts[1])

    rebound_source = audit.rebind(
        changed.state,
        source_digest=SOURCE + jnp.uint32(1),
        representation_digest=REPRESENTATION,
        semantic_digests=changed_semantics,
    )
    assert bool(jnp.all(rebound_source.reset_slots))
    np.testing.assert_array_equal(rebound_source.state.execution_starts, jnp.zeros((2,)))


def test_rebind_defers_for_execution_or_comparator_trial_in_flight() -> None:
    audit, state = _new()
    active = _observe(
        audit,
        state,
        _arm(audit, state, 1, eligible=True, owner=0),
        1,
    ).state
    changed_semantics = SEMANTICS.at[1].set(option_semantic_digest({"new": 1}))
    deferred = audit.rebind(
        active,
        source_digest=SOURCE,
        representation_digest=REPRESENTATION,
        semantic_digests=changed_semantics,
    )
    assert bool(deferred.deferred)
    assert not bool(deferred.applied)
    chex.assert_trees_all_equal(deferred.state, active)

    audit, state = _new()
    trial = _observe(
        audit,
        state,
        _arm(
            audit,
            state,
            1,
            eligible=True,
            owner=-1,
            randomized=True,
            propensity=0.5,
        ),
        1,
    ).state
    deferred = audit.rebind(
        trial,
        source_digest=SOURCE,
        representation_digest=REPRESENTATION,
        semantic_digests=changed_semantics,
    )
    assert bool(deferred.deferred)
    chex.assert_trees_all_equal(deferred.state, trial)


def test_maintenance_reports_shared_context_redundancy_without_authority() -> None:
    audit, state = _new(fixed_horizon=1, redundancy_distance_threshold=0.001)
    step = 1
    for option in range(2):
        for context in range(2):
            treatment = _arm(
                audit,
                state,
                step,
                option=option,
                context=context,
                eligible=True,
                owner=option,
                randomized=True,
                propensity=0.5,
            )
            state = _observe(
                audit,
                state,
                treatment,
                step,
                goal=True,
                planning=(1, 1),
            ).state
            step += 1
            control = _arm(
                audit,
                state,
                step,
                option=option,
                context=context,
                eligible=True,
                owner=-1,
                randomized=True,
                propensity=0.5,
            )
            state = _observe(audit, state, control, step).state
            step += 1
    report = audit.maintenance_report(state)
    assert bool(report.state_valid)
    assert bool(report.redundant_pairs[0, 1])
    assert bool(report.redundancy_loser[1])
    assert int(report.proposed_replacement_slots[0]) == 1
    assert not bool(report.curation_authority)
    assert not bool(report.promotion_authority)
    assert not bool(report.go_no_go_authority)
    chex.assert_trees_all_equal(state, state)


def test_checkpoint_is_exact_and_rejects_tamper_or_stale_binding() -> None:
    audit, state = _new()
    state = _observe(
        audit,
        state,
        _arm(audit, state, 1, eligible=True, owner=0),
        1,
        goal=True,
    ).state
    payload = audit.checkpoint_payload(state)
    restored = audit.restore_checkpoint(
        copy.deepcopy(payload),
        expected_source_digest=SOURCE,
        expected_representation_digest=REPRESENTATION,
        expected_semantic_digests=SEMANTICS,
    )
    chex.assert_trees_all_equal(restored, state)

    tampered = copy.deepcopy(payload)
    state_payload = tampered["state"]
    assert isinstance(state_payload, dict)
    observation = state_payload["observation_count"]
    assert isinstance(observation, dict)
    observation["bytes_hex"] = "00" * 4
    with pytest.raises(ValueError, match="payload digest"):
        audit.restore_checkpoint(
            tampered,
            expected_source_digest=SOURCE,
            expected_representation_digest=REPRESENTATION,
            expected_semantic_digests=SEMANTICS,
        )
    with pytest.raises(ValueError, match="stale"):
        audit.restore_checkpoint(
            payload,
            expected_source_digest=SOURCE + jnp.uint32(1),
            expected_representation_digest=REPRESENTATION,
            expected_semantic_digests=SEMANTICS,
        )


def test_static_array_contracts_and_observation_cap_fail_closed() -> None:
    audit, state = _new(max_observations=1, fixed_horizon=1)
    with pytest.raises(TypeError, match="uint32"):
        _ = audit.arm(
            state,
            transition_id=jnp.asarray([0, 1], dtype=jnp.int32),
            source_digest=SOURCE,
            representation_digest=REPRESENTATION,
            semantic_digests=SEMANTICS,
            semantic_generations=state.semantic_generations,
            candidate_option=0,
            initiation_context=0,
            initiation_eligible=True,
            owner_option=0,
            comparator_randomized=False,
            treatment_propensity=0.0,
            frozen_model_prediction=jnp.zeros((SIGNATURE_DIM,), dtype=jnp.float32),
        )
    state = _observe(
        audit,
        state,
        _arm(audit, state, 1, eligible=True, owner=0),
        1,
        goal=True,
    ).state
    exhausted = _arm(audit, state, 2, eligible=True, owner=0)
    assert not bool(exhausted.available)
    result = _observe(audit, state, exhausted, 2, goal=True)
    chex.assert_trees_all_equal(result.state, state)
