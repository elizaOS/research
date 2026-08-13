# mypy: disable-error-code="attr-defined,call-arg,no-untyped-def,type-var"
"""JIT/scan/resume contracts for the authority-free WP7.3 lifecycle audit.

These are L0 mechanism tests only.  They are not scientific evidence and do
not authorize option installation, selection, replacement, or promotion.
"""

from __future__ import annotations

import json

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.option_lifecycle_audit import (
    OptionLifecycleAudit,
    OptionLifecycleAuditConfig,
    OptionLifecycleAuditState,
    option_semantic_digest,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

SOURCE = option_semantic_digest({"source": "integration"})
REPRESENTATION = option_semantic_digest({"representation": "fixed-v1"})
SEMANTICS = jnp.stack(
    (
        option_semantic_digest({"option": "left"}),
        option_semantic_digest({"option": "right"}),
    )
)


def _audit() -> OptionLifecycleAudit:
    return OptionLifecycleAudit(
        OptionLifecycleAuditConfig(
            n_options=2,
            n_contexts=2,
            outcome_dim=1,
            fixed_horizon=1,
            maintenance_budget=1,
            signature_scales=(1.0,) * 6,
            initiation_opportunity_floor=1,
            completion_evidence_floor=1,
            model_error_evidence_floor=1,
            comparison_treatment_evidence_floor=1,
            comparison_primitive_evidence_floor=1,
            signature_evidence_floor_per_context=1,
            redundancy_shared_context_floor=1,
            redundancy_distance_threshold=0.01,
            max_observations=64,
            max_planning_uses_per_observation=4,
            max_compute_cost_per_observation=10.0,
            max_mean_compute_cost=10.0,
            max_resident_memory_bytes=1_024,
        )
    )


def _init(audit: OptionLifecycleAudit) -> OptionLifecycleAuditState:
    return audit.init(
        source_digest=SOURCE,
        representation_digest=REPRESENTATION,
        semantic_digests=SEMANTICS,
    )


def _scan_step(audit: OptionLifecycleAudit):
    def step(
        state: OptionLifecycleAuditState,
        inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ):
        transition, option, treatment, reward = inputs
        owner = jnp.where(treatment, option, jnp.int32(-1))
        prediction = jnp.concatenate(
            (
                jnp.stack(
                    (
                        reward,
                        reward * 0.5,
                        jnp.float32(1.0),
                        jnp.float32(1.0),
                        jnp.float32(0.9),
                    )
                ),
                jnp.asarray([0.25], dtype=jnp.float32),
            )
        )
        prediction = jnp.where(treatment, prediction, jnp.zeros_like(prediction))
        arm = audit.arm(
            state,
            transition_id=jnp.stack((jnp.uint32(0xBEEF), transition.astype(jnp.uint32))),
            source_digest=SOURCE,
            representation_digest=REPRESENTATION,
            semantic_digests=state.semantic_digests,
            semantic_generations=state.semantic_generations,
            candidate_option=option,
            initiation_context=jnp.int32(0),
            initiation_eligible=jnp.asarray(True, dtype=jnp.bool_),
            owner_option=owner,
            comparator_randomized=jnp.asarray(True, dtype=jnp.bool_),
            treatment_propensity=jnp.float32(0.5),
            frozen_model_prediction=prediction,
        )
        planning = jax.nn.one_hot(option, 2, dtype=jnp.int32)
        result = audit.observe(
            state,
            arm,
            transition_id=arm.transition_id,
            external_reward=reward,
            pseudo_reward=reward * 0.5,
            baseline_mass=jnp.float32(1.0),
            discount=jnp.float32(0.9),
            outcome_delta=jnp.asarray([0.25], dtype=jnp.float32),
            goal_terminated=treatment,
            timeout_terminated=jnp.asarray(False, dtype=jnp.bool_),
            environment_terminated=jnp.asarray(False, dtype=jnp.bool_),
            censored=jnp.asarray(False, dtype=jnp.bool_),
            planning_usage_delta=planning,
            compute_cost=jnp.where(treatment, jnp.float32(2.0), jnp.float32(0.0)),
            resident_memory_bytes=jnp.where(treatment, jnp.int32(128), jnp.int32(0)),
        )
        facts = jnp.stack(
            (
                result.applied,
                result.execution_completed,
                result.comparator_trial_completed,
            )
        )
        return result.state, facts

    return step


def test_eager_jit_and_scan_have_exact_transaction_parity() -> None:
    audit = _audit()
    transitions = jnp.arange(1, 9, dtype=jnp.int32)
    options = jnp.asarray([0, 0, 1, 1, 0, 0, 1, 1], dtype=jnp.int32)
    treatments = jnp.asarray([True, False, True, False, True, False, True, False])
    rewards = jnp.asarray([2.0, 0.5, 2.0, 0.5, 3.0, 1.0, 3.0, 1.0], jnp.float32)
    inputs = (transitions, options, treatments, rewards)
    step = _scan_step(audit)

    eager_state = _init(audit)
    eager_facts = []
    for row in zip(*inputs, strict=True):
        eager_state, facts = step(eager_state, row)
        eager_facts.append(facts)

    scan_state, scan_facts = jax.jit(lambda state, xs: jax.lax.scan(step, state, xs))(
        _init(audit),
        inputs,
    )
    chex.assert_trees_all_equal(scan_state, eager_state)
    np.testing.assert_array_equal(scan_facts, jnp.stack(eager_facts))
    assert bool(jnp.all(scan_facts[:, 0]))
    np.testing.assert_array_equal(scan_facts[:, 1], treatments)
    assert bool(jnp.all(scan_facts[:, 2]))
    np.testing.assert_array_equal(
        scan_state.comparison_treatment_counts,
        [[2, 0], [2, 0]],
    )
    np.testing.assert_array_equal(
        scan_state.comparison_primitive_counts,
        [[2, 0], [2, 0]],
    )
    np.testing.assert_array_equal(scan_state.model_squared_error_sums, jnp.zeros((2, 6)))
    np.testing.assert_array_equal(scan_state.planning_use_counts, [4, 4])


def test_jitted_maintenance_and_rebind_are_proposal_only_and_slot_exact() -> None:
    audit = _audit()
    transitions = jnp.arange(1, 5, dtype=jnp.int32)
    inputs = (
        transitions,
        jnp.asarray([0, 0, 1, 1], dtype=jnp.int32),
        jnp.asarray([True, False, True, False]),
        jnp.asarray([2.0, 1.0, 2.0, 1.0], dtype=jnp.float32),
    )
    state, _ = jax.lax.scan(_scan_step(audit), _init(audit), inputs)
    before = state
    report = jax.jit(audit.maintenance_report)(state)
    assert bool(report.state_valid)
    assert bool(report.redundant_pairs[0, 1])
    assert not bool(report.curation_authority)
    chex.assert_trees_all_equal(state, before)

    changed = SEMANTICS.at[1].set(option_semantic_digest({"option": "replacement"}))
    rebound = jax.jit(
        lambda current, semantics: audit.rebind(
            current,
            source_digest=SOURCE,
            representation_digest=REPRESENTATION,
            semantic_digests=semantics,
        )
    )(state, changed)
    assert bool(rebound.applied)
    np.testing.assert_array_equal(rebound.preserved_slots, [True, False])
    np.testing.assert_array_equal(rebound.reset_slots, [False, True])
    np.testing.assert_array_equal(
        rebound.state.comparison_treatment_counts[0],
        [1, 0],
    )
    np.testing.assert_array_equal(
        rebound.state.comparison_treatment_counts[1],
        [0, 0],
    )
    assert int(rebound.state.semantic_generations[1]) == 1


def test_json_checkpoint_resume_preserves_next_transaction_exactly() -> None:
    audit = _audit()
    inputs = (
        jnp.arange(1, 3, dtype=jnp.int32),
        jnp.asarray([0, 0], dtype=jnp.int32),
        jnp.asarray([True, False]),
        jnp.asarray([2.0, 1.0], dtype=jnp.float32),
    )
    state, _ = jax.lax.scan(_scan_step(audit), _init(audit), inputs)
    payload = json.loads(json.dumps(audit.checkpoint_payload(state)))
    resumed = audit.restore_checkpoint(
        payload,
        expected_source_digest=SOURCE,
        expected_representation_digest=REPRESENTATION,
        expected_semantic_digests=SEMANTICS,
    )
    chex.assert_trees_all_equal(resumed, state)

    next_inputs = (
        jnp.asarray(3, dtype=jnp.int32),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(True, dtype=jnp.bool_),
        jnp.asarray(4.0, dtype=jnp.float32),
    )
    direct, direct_facts = _scan_step(audit)(state, next_inputs)
    after_resume, resume_facts = jax.jit(_scan_step(audit))(resumed, next_inputs)
    chex.assert_trees_all_equal(after_resume, direct)
    np.testing.assert_array_equal(resume_facts, direct_facts)
