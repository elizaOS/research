# mypy: disable-error-code="arg-type,attr-defined,call-arg,no-untyped-call"
"""Focused contracts for the development-only v6 intervention audit."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.integrated_hidden_partner import (
    BASE_FEATURE_DIM,
    RAW_OBSERVATION_DIM,
    IntegratedHiddenPartnerAgent,
)
from alberta_framework.evaluation import (
    hidden_partner_lifecycle_world_v6_intervention_audit as audit_module,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6 import (
    PRIMARY_CONDITION_ORDER,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_controls import (
    V6_DIAGNOSTIC_ORDER,
    build_v6_diagnostic_controls,
    build_v6_primary_controls,
)
from alberta_framework.evaluation.hidden_partner_world_online_bridge import (
    HiddenPartnerWorldOnlineBridge,
)
from alberta_framework.streams.hidden_partner_world_feedback import (
    HiddenPartnerWorldFeedbackWorld,
)

pytestmark = pytest.mark.unit


EXPECTED_AUDIT_ORDER = (
    "behavior_credit_replay",
    "grounded_credit_replay",
    "gradient_mix_mode_bounded_replay",
    "gradient_chain_bounded_replay",
    "state_learning_gate_bounded_replay",
    "grounded_learning_gate_exact",
    "memory_mask_exact",
    "planner_reward_source_exact",
    "planning_application_exact",
    "partner_belief_exact",
    "lifecycle_commit_gate_exact",
    "identity_carry_mode_exact",
    "retention_floor_exact",
    "retirement_gate_exact",
    "random_curation_exact",
    "uniform_action_exact",
    "cue_sampling_exact",
    "row_bias_exact",
)

EXPECTED_WITNESS_ORDER = (
    "behavior_credit_nonzero",
    "grounded_credit_nonzero",
    "state_parameter_proposal_nonzero",
    "grounded_parameter_proposal_nonzero",
    "lifecycle_proposal_event",
    "applied_descriptor_change",
    "retention_floor_counterfactual_bind",
    "retirement_eligible",
    "random_selection_differs_from_utility_selection",
    "masked_hidden_state_downstream_learning_effect",
    "table_and_grounded_rewards_disagree",
    "planner_model_term_nonzero",
    "partner_prediction_nonuniform",
    "forced_action_differs_from_ordinary",
    "equal_cue_differs_from_base_counterfactual",
    "row_bias_proposal_nonzero",
)


def _controls() -> tuple[object, ...]:
    return (*build_v6_primary_controls(), *build_v6_diagnostic_controls())


@functools.lru_cache(maxsize=1)
def _coherent_nonzero_q_case() -> tuple[object, ...]:
    control = build_v6_primary_controls()[0]
    assert control.agent_config is not None
    agent = IntegratedHiddenPartnerAgent(control.agent_config)
    bridge = HiddenPartnerWorldOnlineBridge(
        world=HiddenPartnerWorldFeedbackWorld(control.world_config),
        agent=agent,
    )
    pre_state = bridge.initialize(jr.key(81_001), jr.key(81_002))

    # The current decision cache is intentionally historical: change the
    # committed Q parameters while binding the exact fresh-minus-cached delta.
    q_weights = pre_state.agent.control.q_weights.at[0, 0].set(jnp.float32(0.125))
    control_state = pre_state.agent.control.replace(q_weights=q_weights)
    fresh = agent.evaluate_models(
        pre_state.agent.behavior,
        pre_state.agent.joint_world,
        control_state,
        pre_state.agent.chi,
        pre_state.agent.grounded_world,
    )
    q_delta = fresh.q_values - pre_state.agent.current_evaluation.q_values
    assert bool(jnp.any(q_delta != 0.0))
    agent_state = pre_state.agent.replace(
        control=control_state,
        current_q_value_delta=q_delta,
    )
    pre_state = pre_state.replace(agent=agent_state)
    result = bridge.step(pre_state)
    assert bool(result.trace.accepted)
    audit = audit_module.audit_v6_intervention_step(
        control,
        agent,
        pre_state,
        result,
    )
    return control, agent, pre_state, result, jax.block_until_ready(audit)


def test_fixed_orders_shapes_dtypes_and_no_compact_blockers() -> None:
    assert audit_module.V6_INTERVENTION_AUDIT_ORDER == EXPECTED_AUDIT_ORDER
    assert audit_module.V6_INTERVENTION_WITNESS_ORDER == EXPECTED_WITNESS_ORDER
    assert audit_module.V6_INTERVENTION_COMPACT_FIELD_BLOCKERS == ()
    assert audit_module.V6_FLOAT32_REPLAY_RTOL == 2.0**-20
    assert audit_module.V6_FLOAT32_REPLAY_ATOL == 2.0**-22
    audit = audit_module.V6InterventionStepAudit(
        checks=jnp.ones((18,), dtype=jnp.bool_),
        witnesses=jnp.zeros((16,), dtype=jnp.bool_),
    )
    assert audit_module.validate_v6_intervention_step_audit(audit) is audit
    assert audit.checks.shape == (18,)
    assert audit.checks.dtype == jnp.bool_
    assert audit.witnesses.shape == (16,)
    assert audit.witnesses.dtype == jnp.bool_
    traced_checks = jax.jit(
        lambda checks: (
            audit_module.validate_v6_intervention_step_audit(
                audit_module.V6InterventionStepAudit(
                    checks=checks,
                    witnesses=jnp.zeros((16,), dtype=jnp.bool_),
                )
            ).checks
        )
    )(audit.checks)
    np.testing.assert_array_equal(traced_checks, audit.checks)

    with pytest.raises(ValueError, match="shape"):
        audit_module.validate_v6_intervention_step_audit(
            audit.replace(checks=jnp.ones((17,), dtype=jnp.bool_))
        )
    with pytest.raises(TypeError, match="dtype"):
        audit_module.validate_v6_intervention_step_audit(
            audit.replace(witnesses=jnp.zeros((16,), dtype=jnp.int32))
        )
    with pytest.raises(TypeError, match="JAX array"):
        audit_module.validate_v6_intervention_step_audit(
            audit_module.V6InterventionStepAudit(
                checks=np.ones((18,), dtype=bool),
                witnesses=np.zeros((16,), dtype=bool),
            )
        )


def test_required_witness_mapping_is_complete_ordered_and_fail_closed() -> None:
    controls = _controls()
    assert tuple(control.name for control in controls) == (
        *PRIMARY_CONDITION_ORDER,
        *V6_DIAGNOSTIC_ORDER,
    )
    assert tuple(name for name, _ in audit_module.V6_CONTROL_REQUIRED_WITNESSES) == (
        *PRIMARY_CONDITION_ORDER,
        *V6_DIAGNOSTIC_ORDER,
    )
    all_supported = jnp.ones((16,), dtype=jnp.int32)
    no_support = jnp.zeros((16,), dtype=jnp.int32)
    for control in controls:
        names = audit_module.required_v6_intervention_witness_names(control)
        mask = audit_module.required_v6_intervention_witness_mask(control)
        assert mask.shape == (16,)
        assert mask.dtype == jnp.bool_
        assert int(jnp.sum(mask)) == len(names)
        assert bool(audit_module.v6_required_witnesses_satisfied(control, all_supported))
        assert bool(audit_module.v6_required_witnesses_satisfied(control, no_support)) is (
            not names
        )
        assert audit_module.missing_v6_required_witnesses(control, no_support) == names

    required_control = build_v6_primary_controls()[1]
    assert not bool(
        jax.jit(
            lambda counts: audit_module.v6_required_witnesses_satisfied(
                required_control,
                counts,
            )
        )(no_support)
    )
    with pytest.raises(ValueError, match="shape"):
        audit_module.v6_required_witnesses_satisfied(
            required_control,
            jnp.zeros((15,), dtype=jnp.int32),
        )
    with pytest.raises(TypeError, match="dtype"):
        audit_module.v6_required_witnesses_satisfied(
            required_control,
            jnp.zeros((16,), dtype=jnp.float32),
        )
    with pytest.raises(ValueError, match="non-negative"):
        audit_module.missing_v6_required_witnesses(
            required_control,
            -np.ones((16,), dtype=np.int32),
        )


def test_row_isolation_uses_dynamic_one_hot_under_jit() -> None:
    def check(index: jax.Array, corrupt_index: jax.Array) -> jax.Array:
        weight = jnp.zeros((4,), dtype=jnp.bool_).at[index].set(True)
        bias = jnp.zeros((4,), dtype=jnp.bool_).at[index].set(True)
        valid = audit_module._row_update_isolation_exact(  # noqa: SLF001
            weight,
            bias,
            index,
        )
        corrupt = weight.at[corrupt_index].set(True)
        invalid = audit_module._row_update_isolation_exact(  # noqa: SLF001
            corrupt,
            bias,
            index,
        )
        return jnp.stack((valid, invalid))

    observed = jax.jit(check)(jnp.int32(2), jnp.int32(1))
    np.testing.assert_array_equal(observed, np.asarray((True, False)))


def test_memory_witness_detects_unmasked_learning_from_zero_downstream_weights() -> None:
    control = next(
        item for item in build_v6_primary_controls() if item.name == "recurrent_memory_masked"
    )
    assert control.agent_config is not None
    agent = IntegratedHiddenPartnerAgent(control.agent_config)
    bridge = HiddenPartnerWorldOnlineBridge(
        world=HiddenPartnerWorldFeedbackWorld(control.world_config),
        agent=agent,
    )
    state = bridge.initialize(jr.key(83_001), jr.key(83_002))
    hidden = jnp.asarray((0.25, -0.5, 0.75, 1.0), dtype=jnp.float32)
    phi = state.agent.phi.at[RAW_OBSERVATION_DIM:BASE_FEATURE_DIM].set(hidden)
    read_mask = jnp.ones((12,), dtype=jnp.bool_)
    masked_chi = agent.build_chi(phi, state.agent.router.descriptors, read_mask)
    unmasked_chi = audit_module._unmasked_chi_counterfactual(  # noqa: SLF001
        phi,
        state.agent.router.descriptors,
        read_mask,
    )
    masked_update = agent.behavior_model.update(
        state.agent.behavior,
        masked_chi,
        jnp.asarray(1, dtype=jnp.int32),
    )
    unmasked_update = agent.behavior_model.update(
        state.agent.behavior,
        unmasked_chi,
        jnp.asarray(1, dtype=jnp.int32),
    )
    hidden_slice = slice(RAW_OBSERVATION_DIM, BASE_FEATURE_DIM)
    pre_hidden = state.agent.behavior.weights[:, hidden_slice]
    masked_hidden = masked_update.state.weights[:, hidden_slice]
    unmasked_hidden = unmasked_update.state.weights[:, hidden_slice]

    assert bool(jnp.all(pre_hidden == 0.0))
    assert bool(jnp.all(masked_hidden == 0.0))
    assert bool(jnp.any(unmasked_hidden != 0.0))
    assert bool(
        audit_module._masked_memory_downstream_learning_effect(  # noqa: SLF001
            hidden,
            pre_hidden,
            masked_hidden,
            unmasked_hidden,
        )
    )


def test_disabled_state_gate_rejects_one_ulp_parameter_persistence() -> None:
    _, _, pre_state, result, _ = _coherent_nonzero_q_case()
    pre_builder = pre_state.agent.state_builder
    replay_builder = result.state.agent.state_builder.replace(
        parameters=pre_builder.parameters,
        update_count=pre_builder.update_count,
        last_gradient_norm=pre_builder.last_gradient_norm,
    )
    drifted = replay_builder.replace(
        parameters=replay_builder.parameters.at[0].set(
            jnp.nextafter(
                replay_builder.parameters[0],
                jnp.asarray(jnp.inf, dtype=jnp.float32),
            )
        )
    )
    # The tight replay tolerance alone accepts one ULP; the disabled gate must not.
    assert bool(audit_module._tree_replay_equal(drifted, replay_builder))  # noqa: SLF001
    assert not bool(
        audit_module._state_learning_persistence_exact(  # noqa: SLF001
            enabled=False,
            pre_state=pre_builder,
            post_state=drifted,
            replay_state=drifted,
        )
    )


def test_lifecycle_frozen_rejects_one_ulp_archive_mutation() -> None:
    _, _, pre_state, result, _ = _coherent_nonzero_q_case()
    pre_interaction = pre_state.agent.interaction
    replay_interaction = result.state.agent.interaction
    assert bool(
        audit_module._lifecycle_persistence_exact(  # noqa: SLF001
            enabled=False,
            pre_state=pre_interaction,
            post_state=replay_interaction,
            replay_state=replay_interaction,
        )
    )
    mutated = replay_interaction.replace(
        candidate_utilities=replay_interaction.candidate_utilities.at[0].set(
            jnp.nextafter(
                replay_interaction.candidate_utilities[0],
                jnp.asarray(jnp.inf, dtype=jnp.float32),
            )
        )
    )
    assert not bool(
        audit_module._lifecycle_persistence_exact(  # noqa: SLF001
            enabled=False,
            pre_state=pre_interaction,
            post_state=mutated,
            replay_state=replay_interaction,
        )
    )


def test_real_step_accepts_historical_q_cache_with_nonzero_exact_delta() -> None:
    _, _, pre_state, _, audit = _coherent_nonzero_q_case()
    assert bool(jnp.any(pre_state.agent.current_q_value_delta != 0.0))
    np.testing.assert_array_equal(audit.checks, np.ones((18,), dtype=bool))


def test_adversarial_mix_and_random_rank_corruptions_fail_their_checks() -> None:
    control, agent, pre_state, result, baseline = _coherent_nonzero_q_case()
    np.testing.assert_array_equal(baseline.checks, np.ones((18,), dtype=bool))

    mechanism = result.trace.mechanism
    bad_mix = mechanism.replace(
        mixed_credit_gradient_chi=mechanism.mixed_credit_gradient_chi.at[0].add(jnp.float32(1e-3))
    )
    mix_result = result.replace(trace=result.trace.replace(mechanism=bad_mix))
    mix_audit = audit_module.audit_v6_intervention_step(
        control,
        agent,
        pre_state,
        mix_result,
    )
    assert not bool(mix_audit.checks[2])

    bad_rank = mechanism.replace(
        random_curation_active_priorities=(
            mechanism.random_curation_active_priorities.at[0].add(jnp.float32(1.0))
        )
    )
    rank_result = result.replace(trace=result.trace.replace(mechanism=bad_rank))
    rank_audit = audit_module.audit_v6_intervention_step(
        control,
        agent,
        pre_state,
        rank_result,
    )
    assert not bool(rank_audit.checks[14])


def test_descriptor_identity_recomputation_detects_source_corruption() -> None:
    old = jnp.asarray(
        ((0, 1), (1, 2), (2, 3), *[(-1, -1)] * 9),
        dtype=jnp.int32,
    )
    new = jnp.asarray(
        ((2, 3), (0, 1), (3, 4), *[(-1, -1)] * 9),
        dtype=jnp.int32,
    )
    source, survivor, new_mask, evicted = jax.jit(
        audit_module._expected_route_identity  # noqa: SLF001
    )(old, new)
    np.testing.assert_array_equal(source[:3], np.asarray((2, 0, -1)))
    np.testing.assert_array_equal(survivor[:3], np.asarray((True, True, False)))
    np.testing.assert_array_equal(new_mask[:3], np.asarray((False, False, True)))
    np.testing.assert_array_equal(evicted[:3], np.asarray((False, True, False)))
    corrupted = source.at[0].set(jnp.int32(1))
    assert not bool(audit_module._exact(corrupted, source))  # noqa: SLF001
