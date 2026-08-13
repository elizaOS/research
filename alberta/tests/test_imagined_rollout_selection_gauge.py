# mypy: disable-error-code="attr-defined,call-arg"
"""Grounded authorization and real training contracts for imagined rollouts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.ensemble_short_rollouts import (
    EnsembleShortRolloutConfig,
    EnsembleShortRolloutPlanner,
    EnsembleShortRolloutState,
    ImaginedRolloutBatch,
    RolloutPolicyValueAuthority,
)
from alberta_framework.core.imagined_rollout_selection_gauge import (
    IMAGINED_ROLLOUT_COMPETENT_REAL_TRUTH_AUTHENTICATED,
    IMAGINED_ROLLOUT_CONTENT_INTEGRITY_SCOPE,
    IMAGINED_ROLLOUT_GAUGE_EVIDENCE_LEVEL,
    IMAGINED_ROLLOUT_GAUGE_MECHANISM_STATUS,
    IMAGINED_ROLLOUT_PLANNER_ISSUANCE_AUTHENTICATED,
    IMAGINED_ROLLOUT_SCIENTIFIC_PROMOTION_ALLOWED,
    AuthorizedImaginedRolloutActorCritic,
    GroundedRolloutAuditRecord,
    ImaginedRolloutActorCriticConfig,
    ImaginedRolloutSelectionGauge,
    ImaginedRolloutSelectionGaugeConfig,
    ImaginedRolloutSelectionGaugeState,
    load_imagined_rollout_actor_critic_checkpoint,
    load_imagined_rollout_selection_gauge_checkpoint,
    save_imagined_rollout_actor_critic_checkpoint,
    save_imagined_rollout_selection_gauge_checkpoint,
)
from alberta_framework.core.learning_signals import LearningSignalEstimatorConfig
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig
from alberta_framework.core.world_model_ensemble import (
    WorldModelEnsemble,
    WorldModelEnsembleConfig,
    WorldModelEnsembleState,
)

pytestmark = pytest.mark.unit


def _ensemble() -> WorldModelEnsemble:
    model = ActionConditionedWorldModelConfig(
        observation_dim=2,
        n_actions=2,
        gamma=0.95,
        hidden_sizes=(),
        step_size=0.05,
        sparsity=0.0,
        use_layer_norm=False,
        error_decay=0.8,
    )
    signals = LearningSignalEstimatorConfig(
        ensemble_size=2,
        target_dim=4,
        progress_warmup_steps=2,
        change_calibration_steps=2,
        max_input_magnitude=1_000.0,
        max_predicted_variance=10_000.0,
        max_observed_loss=10_000.0,
    )
    return WorldModelEnsemble(
        WorldModelEnsembleConfig(
            model=model,
            signal_estimator=signals,
            ensemble_size=2,
            bootstrap_probability=0.5,
            residual_variance_warmup_steps=1,
            residual_variance_floor=1.0e-6,
        )
    )


def _set_outputs(
    ensemble: WorldModelEnsemble,
    state: WorldModelEnsembleState,
    *,
    terminal: bool = False,
) -> WorldModelEnsembleState:
    members = []
    output = (0.1, -0.1, 1.0, 0.0 if terminal else 0.5)
    for member in state.member_states:
        learner = member.learner_state
        weights = []
        biases = []
        for head_index, value in enumerate(output):
            weight = jnp.zeros_like(learner.head_params.weights[head_index])
            weight = weight.at[0, 2].set(value)
            weight = weight.at[0, 3].set(value)
            weights.append(weight)
            biases.append(jnp.zeros((1,), dtype=jnp.float32))
        heads = learner.head_params.replace(weights=tuple(weights), biases=tuple(biases))
        members.append(member.replace(learner_state=learner.replace(head_params=heads)))
    return cast(
        WorldModelEnsembleState,
        state.replace(member_states=tuple(members)),
    )


def _planner_system(
    *,
    terminal: bool = False,
) -> tuple[
    EnsembleShortRolloutPlanner,
    WorldModelEnsembleState,
    RolloutPolicyValueAuthority,
    EnsembleShortRolloutState,
]:
    ensemble = _ensemble()
    model_state = _set_outputs(
        ensemble,
        ensemble.init(jr.key(1, impl="threefry2x32")),
        terminal=terminal,
    )
    planner = EnsembleShortRolloutPlanner(
        ensemble,
        EnsembleShortRolloutConfig(
            rollout_horizon=2,
            rollout_budget=1,
            require_residual_proxy_ready=False,
            max_epistemic_disagreement=100.0,
            max_residual_variance=100.0,
            max_proposal_calls=16,
            max_rollout_attempts=16,
            max_imagined_steps=32,
        ),
    )
    revision = jnp.asarray((0, 1), dtype=jnp.uint32)
    authority = planner.bind_authority(
        policy_weights=jnp.zeros((2, 2), dtype=jnp.float32),
        policy_bias=jnp.asarray((20.0, -20.0), dtype=jnp.float32),
        value_weights=jnp.zeros((2,), dtype=jnp.float32),
        value_bias=jnp.asarray(0.0, dtype=jnp.float32),
        action_support_counts=jnp.asarray((20, 20), dtype=jnp.int32),
        source_revision_words=revision,
        model_state=model_state,
        policy_revision_words=revision,
        value_revision_words=revision,
    )
    state = planner.init(
        jr.key(2, impl="threefry2x32"),
        model_state,
        authority,
    )
    return planner, model_state, authority, state


def _proposals(
    planner: EnsembleShortRolloutPlanner,
    model_state: WorldModelEnsembleState,
    authority: RolloutPolicyValueAuthority,
    state: EnsembleShortRolloutState,
    decision_ids: tuple[int, ...],
) -> tuple[EnsembleShortRolloutState, tuple[ImaginedRolloutBatch, ...]]:
    batches = []
    for decision_id in decision_ids:
        anchor = planner.bind_real_anchor(
            jnp.asarray((float(decision_id), 0.0), dtype=jnp.float32),
            jnp.asarray((0, decision_id), dtype=jnp.uint32),
            authority,
        )
        result = planner.propose(state, model_state, authority, anchor)
        assert bool(result.diagnostics.transaction_applied)
        assert bool(jnp.any(result.proposals.transition_valid))
        state = result.state
        batches.append(result.proposals)
    return state, tuple(batches)


def _gauge(
    planner: EnsembleShortRolloutPlanner,
    **overrides: object,
) -> ImaginedRolloutSelectionGauge:
    defaults: dict[str, object] = {
        "audit_capacity": 8,
        "n_regions": 2,
        "min_evidence_count": 2,
        "min_realized_valid_fraction": 1.0,
        "max_mean_abs_reward_error": 0.01,
        "max_root_mean_square_next_observation_error": 0.01,
        "min_termination_accuracy": 1.0,
        "require_success_lcb": True,
        "success_lcb_z": 1.0,
        "min_success_lcb": 0.2,
        "require_top_quantile_purity": True,
        "top_quantile_fraction": 0.5,
        "min_top_quantile_purity": 1.0,
        "max_authorizations": 32,
    }
    defaults.update(overrides)
    return ImaginedRolloutSelectionGauge(
        planner,
        ImaginedRolloutSelectionGaugeConfig(**defaults),  # type: ignore[arg-type]
    )


def _perfect_record(
    gauge: ImaginedRolloutSelectionGauge,
    batch: ImaginedRolloutBatch,
    *,
    step_index: int,
    record_id: int,
    success: bool = True,
    reward_offset: float = 0.0,
    termination_flip: bool = False,
) -> GroundedRolloutAuditRecord:
    predicted_terminated = bool(batch.terminated[0, step_index])
    return gauge.bind_grounded_record(
        batch,
        rollout_index=jnp.asarray(0, dtype=jnp.int32),
        step_index=jnp.asarray(step_index, dtype=jnp.int32),
        region_id=jnp.asarray(0, dtype=jnp.int32),
        record_id_words=jnp.asarray((0, record_id), dtype=jnp.uint32),
        realized_valid=jnp.asarray(True),
        realized_reward=(
            batch.rewards[0, step_index]
            + jnp.asarray(reward_offset, dtype=jnp.float32)
        ),
        realized_next_observation=batch.next_observations[0, step_index],
        realized_terminated=jnp.asarray(
            not predicted_terminated if termination_flip else predicted_terminated
        ),
        realized_success=jnp.asarray(success),
    )


def _calibrated_system(*, terminal: bool = False) -> tuple[
    ImaginedRolloutSelectionGauge,
    ImaginedRolloutSelectionGaugeState,
    ImaginedRolloutBatch,
    ImaginedRolloutBatch,
]:
    planner, model_state, authority, planner_state = _planner_system(terminal=terminal)
    _, (audit_batch, second_audit_batch, candidate_batch) = _proposals(
        planner,
        model_state,
        authority,
        planner_state,
        (1, 2, 3),
    )
    gauge = _gauge(planner)
    gauge_state = gauge.init(audit_batch)
    for index, grounded_batch in enumerate((audit_batch, second_audit_batch)):
        result = gauge.record_grounded_outcome(
            gauge_state,
            _perfect_record(
                gauge,
                grounded_batch,
                step_index=0,
                record_id=index + 1,
            ),
        )
        assert bool(result.diagnostics.applied)
        gauge_state = result.state
    return gauge, gauge_state, audit_batch, candidate_batch


def _authorization(
    gauge: ImaginedRolloutSelectionGauge,
    gauge_state: ImaginedRolloutSelectionGaugeState,
    batch: ImaginedRolloutBatch,
) -> object:
    return gauge.authorize(
        gauge_state,
        batch,
        region_ids=jnp.zeros(batch.actions.shape, dtype=jnp.int32),
        safety_admitted=jnp.ones(batch.actions.shape, dtype=jnp.bool_),
        protected=jnp.zeros(batch.actions.shape, dtype=jnp.bool_),
    )


def _materialize_keys(tree: object) -> object:
    def convert(value: object) -> object:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            return jr.key_data(value)  # type: ignore[arg-type]
        return value

    return jax.tree.map(convert, tree)


def test_gauge_is_strict_l0_fixed_resource_and_never_mutates_planner_batch() -> None:
    planner, model_state, authority, planner_state = _planner_system()
    _, (batch,) = _proposals(
        planner,
        model_state,
        authority,
        planner_state,
        (1,),
    )
    gauge = _gauge(planner)
    state = gauge.init(batch)
    config = gauge.to_config()
    batch_before = jax.tree.map(lambda value: value.copy(), batch)
    planner_state_before = _materialize_keys(planner_state)

    assert config["mechanism_status"] == IMAGINED_ROLLOUT_GAUGE_MECHANISM_STATUS
    assert config["evidence_level"] == IMAGINED_ROLLOUT_GAUGE_EVIDENCE_LEVEL == "L0"
    assert config["scientific_promotion_allowed"] is False
    assert config["content_integrity_scope"] == IMAGINED_ROLLOUT_CONTENT_INTEGRITY_SCOPE
    assert config["planner_issuance_authenticated"] is False
    assert IMAGINED_ROLLOUT_PLANNER_ISSUANCE_AUTHENTICATED is False
    assert IMAGINED_ROLLOUT_COMPETENT_REAL_TRUTH_AUTHENTICATED is False
    assert IMAGINED_ROLLOUT_SCIENTIFIC_PROMOTION_ALLOWED is False
    assert gauge.resource_budget.persistent_state_bytes == sum(
        np.asarray(value).nbytes for value in jax.tree.leaves(state)
    )
    assert gauge.resource_budget.planner_or_model_state_owned == 0
    assert gauge.resource_budget.dispatch_authority == 0
    assert gauge.resource_budget.max_grounded_records == 8
    assert ImaginedRolloutSelectionGauge.from_config(config).to_config() == config

    record = _perfect_record(gauge, batch, step_index=0, record_id=1)
    _ = gauge.record_grounded_outcome(state, record)
    chex.assert_trees_all_equal(batch, batch_before)
    chex.assert_trees_all_equal(_materialize_keys(planner_state), planner_state_before)


def test_audited_candidate_separation_and_noncompensating_evidence_floors() -> None:
    gauge, state, audit_batch, candidate = _calibrated_system()
    authorized = _authorization(gauge, state, candidate)

    assert bool(authorized.diagnostics.transaction_applied)
    assert bool(authorized.diagnostics.post_mint_content_integrity_only)
    assert not bool(authorized.diagnostics.planner_issuance_authenticated)
    assert bool(authorized.receipt.authorized)
    assert bool(jnp.all(authorized.receipt.evidence_floor_passed[candidate.transition_valid]))
    assert bool(jnp.all(authorized.receipt.realized_validity_passed[candidate.transition_valid]))
    assert bool(jnp.all(authorized.receipt.reward_error_passed[candidate.transition_valid]))
    assert bool(jnp.all(authorized.receipt.next_state_error_passed[candidate.transition_valid]))
    assert bool(jnp.all(authorized.receipt.termination_passed[candidate.transition_valid]))
    assert bool(jnp.all(authorized.receipt.success_lcb_passed[candidate.transition_valid]))
    assert bool(jnp.all(authorized.receipt.top_quantile_purity_passed[candidate.transition_valid]))

    self_authorization = _authorization(gauge, state, audit_batch)
    assert not bool(self_authorization.receipt.authorized)
    assert not bool(self_authorization.diagnostics.audit_candidate_separated)

    planner, model_state, authority, planner_state = _planner_system()
    _, (bad_audit, bad_candidate) = _proposals(
        planner,
        model_state,
        authority,
        planner_state,
        (3, 4),
    )
    bad_gauge = _gauge(planner)
    bad_state = bad_gauge.init(bad_audit)
    for index in range(2):
        outcome = bad_gauge.record_grounded_outcome(
            bad_state,
            _perfect_record(
                bad_gauge,
                bad_audit,
                step_index=index,
                record_id=index + 1,
                reward_offset=1.0,
            ),
        )
        bad_state = outcome.state
    rejected = _authorization(bad_gauge, bad_state, bad_candidate)
    assert not bool(rejected.receipt.authorized)
    assert bool(jnp.all(rejected.receipt.next_state_error_passed[bad_candidate.transition_valid]))
    assert not bool(jnp.any(rejected.receipt.reward_error_passed[bad_candidate.transition_valid]))


def test_irrelevant_batch_mutation_cannot_launder_an_audited_transition() -> None:
    gauge, state, audit_batch, _ = _calibrated_system()
    exact = _authorization(gauge, state, audit_batch)
    laundered = cast(
        ImaginedRolloutBatch,
        audit_batch.replace(
            bootstrap_values=audit_batch.bootstrap_values.at[0].add(
                jnp.asarray(1.0e-3, dtype=jnp.float32)
            )
        ),
    )

    assert gauge.proposal_content_tag(laundered) != gauge.proposal_content_tag(audit_batch)
    bypass = _authorization(gauge, state, laundered)
    assert not bool(exact.diagnostics.audit_candidate_separated)
    assert bool(bypass.diagnostics.batch_valid)
    assert not bool(bypass.diagnostics.audit_candidate_separated)
    assert not bool(bypass.diagnostics.transaction_applied)
    assert not bool(bypass.receipt.authorized)
    chex.assert_trees_all_equal(bypass.state, state)

    duplicate_record = _perfect_record(
        gauge,
        laundered,
        step_index=0,
        record_id=3,
    )
    duplicate = gauge.record_grounded_outcome(state, duplicate_record)
    assert not bool(duplicate.diagnostics.proposal_slot_fresh)
    assert not bool(duplicate.diagnostics.applied)
    chex.assert_trees_all_equal(duplicate.state, state)


def test_receipt_binds_every_batch_word_masks_revision_and_calibration_content() -> None:
    gauge, state, audit_batch, candidate = _calibrated_system()
    authorization = _authorization(gauge, state, candidate)
    state = authorization.state
    receipt = authorization.receipt
    assert bool(gauge.receipt_valid(state, candidate, receipt))

    tampered_batch = cast(
        ImaginedRolloutBatch,
        candidate.replace(rewards=candidate.rewards.at[0, 0].add(1.0e-3)),
    )
    assert not bool(gauge.receipt_valid(state, tampered_batch, receipt))
    aliased_model = cast(
        ImaginedRolloutBatch,
        candidate.replace(
            model_integrity_tags=candidate.model_integrity_tags.at[0].add(
                jnp.asarray(1, dtype=jnp.uint32)
            )
        ),
    )
    assert not bool(gauge.receipt_valid(state, aliased_model, receipt))

    tampered_receipt = cast(
        Any,
        receipt,
    ).replace(protected=receipt.protected.at[0, 0].set(True))
    assert not bool(gauge.receipt_valid(state, candidate, tampered_receipt))

    record = gauge.record_grounded_outcome(
        state,
        _perfect_record(
            gauge,
            audit_batch,
            step_index=1,
            record_id=3,
        ),
    )
    assert bool(record.diagnostics.applied)
    assert not bool(gauge.receipt_valid(record.state, candidate, receipt))


def test_termination_and_full_safety_protected_masks_are_preserved() -> None:
    gauge, state, _, candidate = _calibrated_system(terminal=True)
    assert bool(candidate.terminated[0, 0])
    assert float(candidate.continuations[0, 0]) == 0.0
    safety = jnp.ones(candidate.actions.shape, dtype=jnp.bool_)
    protected = jnp.zeros(candidate.actions.shape, dtype=jnp.bool_).at[0, 0].set(True)
    result = gauge.authorize(
        state,
        candidate,
        region_ids=jnp.zeros(candidate.actions.shape, dtype=jnp.int32),
        safety_admitted=safety,
        protected=protected,
    )

    chex.assert_trees_all_equal(result.receipt.safety_admitted, safety)
    chex.assert_trees_all_equal(result.receipt.protected, protected)
    assert not bool(result.receipt.transition_authorized[0, 0])
    assert not bool(jnp.any(result.receipt.transition_authorized & protected))
    assert not bool(
        jnp.any(result.receipt.transition_authorized & ~candidate.transition_valid)
    )
    assert bool(result.diagnostics.terminal_semantics_valid)


def test_authorization_is_prefix_closed_for_safety_and_calibration_gates() -> None:
    gauge, state, _, candidate = _calibrated_system()
    assert candidate.transition_valid.tolist() == [[True, True]]
    zeros_i = jnp.zeros(candidate.actions.shape, dtype=jnp.int32)
    zeros_b = jnp.zeros(candidate.actions.shape, dtype=jnp.bool_)
    safety = jnp.ones(candidate.actions.shape, dtype=jnp.bool_).at[0, 0].set(False)
    safety_result = gauge.authorize(
        state,
        candidate,
        region_ids=zeros_i,
        safety_admitted=safety,
        protected=zeros_b,
    )
    assert not bool(jnp.any(safety_result.receipt.transition_authorized))
    assert not bool(safety_result.receipt.authorized)

    mixed_regions = zeros_i.at[0, 0].set(1)
    calibration_result = gauge.authorize(
        state,
        candidate,
        region_ids=mixed_regions,
        safety_admitted=jnp.ones(candidate.actions.shape, dtype=jnp.bool_),
        protected=zeros_b,
    )
    assert not bool(calibration_result.receipt.evidence_floor_passed[0, 0])
    assert bool(calibration_result.receipt.evidence_floor_passed[0, 1])
    assert not bool(jnp.any(calibration_result.receipt.transition_authorized))
    assert not bool(calibration_result.receipt.authorized)


def test_post_terminal_valid_bit_resurrection_is_rejected_fail_closed() -> None:
    gauge, state, _, candidate = _calibrated_system(terminal=True)
    assert candidate.transition_valid.tolist() == [[True, False]]
    assert candidate.terminated.tolist() == [[True, False]]
    resurrected = cast(
        ImaginedRolloutBatch,
        candidate.replace(
            transition_valid=candidate.transition_valid.at[0, 1].set(True)
        ),
    )

    result = _authorization(gauge, state, resurrected)
    assert not bool(result.diagnostics.terminal_semantics_valid)
    assert not bool(result.diagnostics.batch_valid)
    assert not bool(result.diagnostics.transaction_applied)
    assert not bool(result.receipt.authorized)
    assert not bool(jnp.any(result.receipt.transition_authorized))
    chex.assert_trees_all_equal(result.state, state)


def test_dream_consumer_exact_gradient_commit_replay_and_unauthorized_zero_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gauge, gauge_state, audit_batch, candidate = _calibrated_system(terminal=True)
    authorization = _authorization(gauge, gauge_state, candidate)
    gauge_state = authorization.state
    consumer = AuthorizedImaginedRolloutActorCritic(
        gauge,
        ImaginedRolloutActorCriticConfig(
            actor_step_size=0.1,
            critic_step_size=0.2,
            momentum_decay=0.0,
            gradient_clip=100.0,
            initialization_scale=0.0,
            max_update_calls=8,
            max_backward_transitions=16,
        ),
    )
    state = consumer.init(jr.key(20, impl="threefry2x32"))
    original_value_and_grad = jax.value_and_grad
    autodiff_calls = 0

    def counted_value_and_grad(*args: object, **kwargs: object) -> Any:
        nonlocal autodiff_calls
        autodiff_calls += 1
        return original_value_and_grad(*args, **kwargs)

    monkeypatch.setattr(jax, "value_and_grad", counted_value_and_grad)
    proposal = consumer.propose_dream_update(
        state,
        candidate,
        authorization.receipt,
        gauge_state,
    )

    assert bool(proposal.valid)
    assert int(proposal.eligible_transition_count) > 0
    assert not hasattr(proposal, "actor_gradient")
    assert autodiff_calls == 0

    tampered = cast(
        Any,
        proposal,
    ).replace(
        source_content_tag=proposal.source_content_tag
        + jnp.asarray(1, dtype=jnp.uint32)
    )
    tampered_result = consumer.commit_dream_update(
        state,
        tampered,
        candidate,
        authorization.receipt,
        gauge_state,
    )
    assert autodiff_calls == 0
    assert not bool(tampered_result.diagnostics.preflight_valid)
    assert not bool(tampered_result.trace.backward_work_performed)
    chex.assert_trees_all_equal(tampered_result.state, state)

    committed = consumer.commit_dream_update(
        state,
        proposal,
        candidate,
        authorization.receipt,
        gauge_state,
    )
    assert autodiff_calls == 1
    assert bool(committed.diagnostics.preflight_valid)
    assert int(committed.diagnostics.autodiff_pass_count) == 1
    assert bool(committed.diagnostics.applied)
    trace = committed.trace
    assert bool(trace.backward_work_performed)
    assert int(trace.backward_transition_count) == int(
        proposal.eligible_transition_count
    )
    chex.assert_trees_all_equal(
        trace.critic_targets[candidate.terminated],
        candidate.rewards[candidate.terminated],
    )
    assert bool(jnp.all(trace.positive_advantages >= 0.0))
    assert bool(jnp.any(trace.imitation_weights > 0.0))
    observation = candidate.observations[0, 0]
    target = candidate.rewards[0, 0]
    advantage = target
    expected_actor_bias_gradient = jnp.asarray(
        (-0.5 * advantage, 0.5 * advantage),
        dtype=jnp.float32,
    )
    expected_actor_weight_gradient = expected_actor_bias_gradient[:, None] * observation
    np.testing.assert_allclose(
        trace.actor_gradient.bias,
        expected_actor_bias_gradient,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        trace.actor_gradient.weights,
        expected_actor_weight_gradient,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        trace.critic_gradient.weights,
        -target * observation,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        trace.critic_gradient.bias,
        -target,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    expected_actor = jax.tree.map(
        lambda parameter, update: parameter + update,
        state.actor_parameters,
        trace.actor_parameter_update,
    )
    expected_critic = jax.tree.map(
        lambda parameter, update: parameter + update,
        state.critic_parameters,
        trace.critic_parameter_update,
    )
    chex.assert_trees_all_close(committed.state.actor_parameters, expected_actor)
    chex.assert_trees_all_close(committed.state.critic_parameters, expected_critic)
    assert int(committed.state.update_count_words[1]) == 1
    assert int(committed.state.dream_update_count_words[1]) == 1
    assert int(committed.state.backward_transition_count_words[1]) == int(
        trace.backward_transition_count
    )

    replay = consumer.commit_dream_update(
        committed.state,
        proposal,
        candidate,
        authorization.receipt,
        gauge_state,
    )
    assert autodiff_calls == 1
    assert not bool(replay.diagnostics.applied)
    assert int(replay.diagnostics.autodiff_pass_count) == 0
    assert not bool(replay.trace.backward_work_performed)
    chex.assert_trees_all_equal(replay.state, committed.state)

    self_receipt = _authorization(gauge, gauge_state, audit_batch)
    unauthorized = consumer.propose_dream_update(
        state,
        audit_batch,
        self_receipt.receipt,
        self_receipt.state,
    )
    assert not bool(unauthorized.valid)
    rejected = consumer.commit_dream_update(
        state,
        unauthorized,
        audit_batch,
        self_receipt.receipt,
        self_receipt.state,
    )
    assert autodiff_calls == 1
    assert not bool(rejected.diagnostics.applied)
    assert int(rejected.diagnostics.autodiff_pass_count) == 0
    assert not bool(rejected.trace.backward_work_performed)
    assert rejected.state.update_count_words.tolist() == [0, 0]
    assert rejected.state.backward_transition_count_words.tolist() == [0, 0]


def test_nonfinite_backward_candidate_is_an_atomic_zero_count_rollback() -> None:
    gauge, gauge_state, _, candidate = _calibrated_system()
    maximum = jnp.asarray(np.finfo(np.float32).max, dtype=jnp.float32)
    extreme = cast(
        ImaginedRolloutBatch,
        candidate.replace(
            observations=jnp.full_like(candidate.observations, maximum),
            return_targets=jnp.full_like(candidate.return_targets, maximum),
        ),
    )
    authorization = _authorization(gauge, gauge_state, extreme)
    assert bool(authorization.receipt.authorized)
    assert not bool(authorization.diagnostics.planner_issuance_authenticated)
    consumer = AuthorizedImaginedRolloutActorCritic(
        gauge,
        ImaginedRolloutActorCriticConfig(
            actor_step_size=0.1,
            critic_step_size=0.1,
            momentum_decay=0.0,
            gradient_clip=100.0,
            initialization_scale=0.0,
            max_update_calls=8,
            max_backward_transitions=16,
        ),
    )
    state = consumer.init(jr.key(24, impl="threefry2x32"))
    proposal = consumer.propose_dream_update(
        state,
        extreme,
        authorization.receipt,
        authorization.state,
    )
    assert bool(proposal.valid)
    result = consumer.commit_dream_update(
        state,
        proposal,
        extreme,
        authorization.receipt,
        authorization.state,
    )
    assert bool(result.trace.backward_work_performed)
    assert not bool(result.trace.candidate_finite)
    assert int(result.diagnostics.autodiff_pass_count) == 1
    assert not bool(result.diagnostics.applied)
    chex.assert_trees_all_equal(result.state, state)
    assert result.state.update_count_words.tolist() == [0, 0]
    assert result.state.backward_transition_count_words.tolist() == [0, 0]


def test_competent_real_control_is_matched_bounded_and_source_mode_isolated() -> None:
    gauge, gauge_state, _, candidate = _calibrated_system()
    authorization = _authorization(gauge, gauge_state, candidate)
    consumer = AuthorizedImaginedRolloutActorCritic(
        gauge,
        ImaginedRolloutActorCriticConfig(
            actor_step_size=0.1,
            critic_step_size=0.1,
            momentum_decay=0.0,
            gradient_clip=100.0,
            initialization_scale=0.0,
            max_update_calls=8,
            max_backward_transitions=16,
        ),
    )
    state = consumer.init(jr.key(21, impl="threefry2x32"))
    real = consumer.bind_competent_real_episode(
        observations=candidate.observations,
        actions=candidate.actions,
        rewards=candidate.rewards,
        continuations=candidate.continuations,
        next_observations=candidate.next_observations,
        return_targets=candidate.return_targets,
        terminated=candidate.terminated,
        transition_valid=candidate.transition_valid,
        competent=jnp.ones(candidate.actions.shape, dtype=jnp.bool_),
        safety_admitted=jnp.ones(candidate.actions.shape, dtype=jnp.bool_),
        protected=jnp.zeros(candidate.actions.shape, dtype=jnp.bool_),
        episode_revision_words=jnp.asarray((0, 1), dtype=jnp.uint32),
        source_revision_words=candidate.source_revision_words[0],
        source_integrity_tag=candidate.source_integrity_tags[0],
    )
    real_proposal = consumer.propose_competent_real_update(state, real)
    assert bool(real_proposal.valid)
    assert int(real_proposal.eligible_transition_count) <= consumer.max_transition_budget
    assert real_proposal.source_mode == consumer.COMPETENT_REAL_SOURCE_MODE

    wrong_mode = consumer.commit_dream_update(
        state,
        real_proposal,
        candidate,
        authorization.receipt,
        authorization.state,
    )
    assert not bool(wrong_mode.diagnostics.applied)
    assert not bool(wrong_mode.trace.backward_work_performed)
    committed = consumer.commit_competent_real_update(state, real_proposal, real)
    assert bool(committed.diagnostics.applied)
    assert not bool(committed.diagnostics.source_truth_authenticated)
    assert consumer.to_config()["competent_real_truth_authenticated"] is False
    assert bool(
        jnp.all(committed.trace.imitation_weights[real.transition_valid] == 1.0)
    )
    assert int(committed.diagnostics.autodiff_pass_count) == 1
    assert int(committed.state.real_update_count_words[1]) == 1
    assert int(committed.state.dream_update_count_words[1]) == 0

    blocked_real = consumer.bind_competent_real_episode(
        observations=candidate.observations,
        actions=candidate.actions,
        rewards=candidate.rewards,
        continuations=candidate.continuations,
        next_observations=candidate.next_observations,
        return_targets=candidate.return_targets,
        terminated=candidate.terminated,
        transition_valid=candidate.transition_valid,
        competent=jnp.asarray(((False, True),), dtype=jnp.bool_),
        safety_admitted=jnp.ones(candidate.actions.shape, dtype=jnp.bool_),
        protected=jnp.zeros(candidate.actions.shape, dtype=jnp.bool_),
        episode_revision_words=jnp.asarray((0, 2), dtype=jnp.uint32),
        source_revision_words=candidate.source_revision_words[0],
        source_integrity_tag=candidate.source_integrity_tags[0],
    )
    blocked_proposal = consumer.propose_competent_real_update(state, blocked_real)
    assert bool(blocked_proposal.source_authorized)
    assert int(blocked_proposal.eligible_transition_count) == 0
    assert not bool(blocked_proposal.valid)
    blocked = consumer.commit_competent_real_update(
        state,
        blocked_proposal,
        blocked_real,
    )
    assert not bool(blocked.trace.backward_work_performed)
    assert int(blocked.diagnostics.autodiff_pass_count) == 0
    assert not bool(blocked.diagnostics.applied)
    chex.assert_trees_all_equal(blocked.state, state)


def test_competent_real_commit_has_eager_jit_scan_parity() -> None:
    gauge, _, _, candidate = _calibrated_system()
    consumer = AuthorizedImaginedRolloutActorCritic(
        gauge,
        ImaginedRolloutActorCriticConfig(
            max_update_calls=8,
            max_backward_transitions=16,
        ),
    )
    initial = consumer.init(jr.key(25, impl="threefry2x32"))

    def real_batch(revision: int) -> object:
        return consumer.bind_competent_real_episode(
            observations=candidate.observations,
            actions=candidate.actions,
            rewards=candidate.rewards,
            continuations=candidate.continuations,
            next_observations=candidate.next_observations,
            return_targets=candidate.return_targets,
            terminated=candidate.terminated,
            transition_valid=candidate.transition_valid,
            competent=jnp.ones(candidate.actions.shape, dtype=jnp.bool_),
            safety_admitted=jnp.ones(candidate.actions.shape, dtype=jnp.bool_),
            protected=jnp.zeros(candidate.actions.shape, dtype=jnp.bool_),
            episode_revision_words=jnp.asarray((0, revision), dtype=jnp.uint32),
            source_revision_words=candidate.source_revision_words[0],
            source_integrity_tag=candidate.source_integrity_tags[0],
        )

    batches = (real_batch(1), real_batch(2))
    stacked = jax.tree.map(lambda *values: jnp.stack(values), *batches)

    def step(state: object, batch: object) -> tuple[object, tuple[jax.Array, jax.Array]]:
        proposal = consumer.propose_competent_real_update(state, batch)
        result = consumer.commit_competent_real_update(state, proposal, batch)
        return result.state, (
            result.diagnostics.applied,
            result.diagnostics.autodiff_pass_count,
        )

    scanned_state, (applied, autodiff_passes) = jax.jit(
        lambda state, items: jax.lax.scan(step, state, items)
    )(initial, stacked)
    eager_state = initial
    for batch in batches:
        proposal = consumer.propose_competent_real_update(eager_state, batch)
        eager_state = consumer.commit_competent_real_update(
            eager_state,
            proposal,
            batch,
        ).state

    assert applied.tolist() == [True, True]
    assert autodiff_passes.tolist() == [1, 1]
    chex.assert_trees_all_close(scanned_state, eager_state)


def test_checkpoints_resources_jit_and_audit_scan_are_exact(tmp_path: Path) -> None:
    planner, model_state, authority, planner_state = _planner_system()
    _, (audit_batch, candidate) = _proposals(
        planner,
        model_state,
        authority,
        planner_state,
        (1, 2),
    )
    gauge = _gauge(planner)
    initial = gauge.init(audit_batch)
    records = tuple(
        _perfect_record(
            gauge,
            audit_batch,
            step_index=index,
            record_id=index + 1,
        )
        for index in range(2)
    )
    stacked = jax.tree.map(lambda *values: jnp.stack(values), *records)

    def audit_step(
        state: ImaginedRolloutSelectionGaugeState,
        record: GroundedRolloutAuditRecord,
    ) -> tuple[ImaginedRolloutSelectionGaugeState, jax.Array]:
        result = gauge.record_grounded_outcome(state, record)
        return result.state, result.diagnostics.applied

    scanned_state, applied = jax.jit(
        lambda state, items: jax.lax.scan(audit_step, state, items)
    )(initial, stacked)
    assert bool(jnp.all(applied))
    eager_state = initial
    for record in records:
        eager_state = gauge.record_grounded_outcome(eager_state, record).state
    chex.assert_trees_all_equal(scanned_state, eager_state)

    eager_authorization = _authorization(gauge, eager_state, candidate)
    compiled_authorization = jax.jit(
        lambda state: _authorization(gauge, state, candidate)
    )(eager_state)
    chex.assert_trees_all_equal(compiled_authorization, eager_authorization)

    gauge_path = tmp_path / "gauge"
    save_imagined_rollout_selection_gauge_checkpoint(gauge, eager_state, gauge_path)
    restored_gauge, restored_gauge_state = (
        load_imagined_rollout_selection_gauge_checkpoint(gauge_path)
    )
    assert restored_gauge.to_config() == gauge.to_config()
    chex.assert_trees_all_equal(restored_gauge_state, eager_state)

    consumer = AuthorizedImaginedRolloutActorCritic(
        gauge,
        ImaginedRolloutActorCriticConfig(
            max_update_calls=8,
            max_backward_transitions=16,
        ),
    )
    consumer_state = consumer.init(jr.key(23, impl="threefry2x32"))
    proposal = consumer.propose_dream_update(
        consumer_state,
        candidate,
        eager_authorization.receipt,
        eager_authorization.state,
    )
    committed = consumer.commit_dream_update(
        consumer_state,
        proposal,
        candidate,
        eager_authorization.receipt,
        eager_authorization.state,
    )
    assert bool(committed.diagnostics.applied)
    compiled_proposal = jax.jit(
        lambda state: consumer.propose_dream_update(
            state,
            candidate,
            eager_authorization.receipt,
            eager_authorization.state,
        )
    )(consumer_state)
    compiled_commit = jax.jit(
        lambda state, pending: consumer.commit_dream_update(
            state,
            pending,
            candidate,
            eager_authorization.receipt,
            eager_authorization.state,
        )
    )(consumer_state, compiled_proposal)
    chex.assert_trees_all_equal(compiled_proposal, proposal)
    chex.assert_trees_all_close(compiled_commit, committed)
    assert consumer.resource_budget.max_transitions_per_update == 2
    assert consumer.resource_budget.proposal_autodiff_passes == 0
    assert (
        consumer.resource_budget.max_autodiff_passes_per_preflight_valid_commit
        == 1
    )
    assert consumer.resource_budget.rejected_preflight_autodiff_passes == 0
    assert consumer.resource_budget.backward_clock_counts_accepted_transitions
    assert consumer.resource_budget.discarded_functional_state_can_repeat_pure_calls
    assert consumer.resource_budget.dispatch_authority == 0
    consumer_path = tmp_path / "consumer"
    save_imagined_rollout_actor_critic_checkpoint(
        consumer,
        committed.state,
        consumer_path,
    )
    restored_consumer, restored_consumer_state = (
        load_imagined_rollout_actor_critic_checkpoint(consumer_path)
    )
    assert restored_consumer.to_config() == consumer.to_config()
    chex.assert_trees_all_equal(restored_consumer_state, committed.state)


def test_public_roots_export_the_complete_gauge_surface() -> None:
    import alberta_framework as package_root
    import alberta_framework.core as core_root
    from alberta_framework.core import imagined_rollout_selection_gauge

    for name in imagined_rollout_selection_gauge.__all__:
        implementation = getattr(imagined_rollout_selection_gauge, name)
        assert getattr(core_root, name) is implementation
        assert getattr(package_root, name) is implementation
        assert name in core_root.__all__
        assert name in package_root.__all__
