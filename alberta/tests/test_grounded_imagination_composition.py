# mypy: disable-error-code="attr-defined,call-arg,no-any-return"
"""Strict end-to-end contracts for grounded imagination composition."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework as public_root
import alberta_framework.core as public_core
import alberta_framework.core.grounded_imagination_composition as composition_module
from alberta_framework.core.ensemble_short_rollouts import (
    EnsembleShortRolloutConfig,
    EnsembleShortRolloutPlanner,
    ImaginedRolloutBatch,
)
from alberta_framework.core.grounded_imagination_composition import (
    GROUNDED_IMAGINATION_COMPOSITION_EVIDENCE_LEVEL,
    GROUNDED_IMAGINATION_COMPOSITION_MECHANISM_STATUS,
    GROUNDED_IMAGINATION_COMPOSITION_SCIENTIFIC_PROMOTION_ALLOWED,
    GROUNDED_IMAGINATION_MODEL_SUPPORT_AUTHENTICATED,
    GROUNDED_IMAGINATION_REAL_ENVIRONMENT_AUTHENTICATED,
    GROUNDED_IMAGINATION_REGION_ASSIGNMENTS_AUTHENTICATED,
    GROUNDED_IMAGINATION_SAFETY_PROTECTION_AUTHENTICATED,
    GroundedImaginationComposition,
    GroundedImaginationCompositionState,
    load_grounded_imagination_composition_checkpoint,
    save_grounded_imagination_composition_checkpoint,
)
from alberta_framework.core.imagined_rollout_selection_gauge import (
    AuthorizedImaginedRolloutActorCritic,
    ImaginedRolloutActorCriticConfig,
    ImaginedRolloutSelectionGauge,
    ImaginedRolloutSelectionGaugeConfig,
    ImaginedRolloutSelectionGaugeState,
)
from alberta_framework.core.learning_signals import LearningSignalEstimatorConfig
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig
from alberta_framework.core.world_model_ensemble import (
    WorldModelEnsemble,
    WorldModelEnsembleConfig,
    WorldModelEnsembleState,
)

pytestmark = pytest.mark.unit

REVISION_ONE = jnp.asarray((0, 1), dtype=jnp.uint32)
SUPPORT = jnp.asarray((20, 20), dtype=jnp.int32)


def _materialize_keys(tree: object) -> object:
    def materialize(value: object) -> object:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            return jr.key_data(value)  # type: ignore[arg-type]
        return value

    return jax.tree.map(materialize, tree)


def _assert_tree_equal(left: object, right: object) -> None:
    chex.assert_trees_all_equal(_materialize_keys(left), _materialize_keys(right))


def _assert_tree_close(left: object, right: object) -> None:
    chex.assert_trees_all_close(
        _materialize_keys(left),
        _materialize_keys(right),
        rtol=1.0e-6,
        atol=1.0e-7,
    )


def _ensemble() -> WorldModelEnsemble:
    return WorldModelEnsemble(
        WorldModelEnsembleConfig(
            model=ActionConditionedWorldModelConfig(
                observation_dim=2,
                n_actions=2,
                gamma=0.95,
                hidden_sizes=(),
                step_size=0.05,
                sparsity=0.0,
                use_layer_norm=False,
                error_decay=0.8,
            ),
            signal_estimator=LearningSignalEstimatorConfig(
                ensemble_size=2,
                target_dim=4,
                progress_warmup_steps=2,
                change_calibration_steps=2,
                max_input_magnitude=1_000.0,
                max_predicted_variance=10_000.0,
                max_observed_loss=10_000.0,
            ),
            ensemble_size=2,
            bootstrap_probability=0.5,
            residual_variance_warmup_steps=1,
            residual_variance_floor=1.0e-6,
        )
    )


def _set_constant_outputs(
    ensemble: WorldModelEnsemble,
    state: WorldModelEnsembleState,
) -> WorldModelEnsembleState:
    members = []
    output = (0.1, -0.1, 1.0, 0.5)
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
        heads = learner.head_params.replace(
            weights=tuple(weights),
            biases=tuple(biases),
        )
        members.append(member.replace(learner_state=learner.replace(head_params=heads)))
    result = cast(
        WorldModelEnsembleState,
        state.replace(member_states=tuple(members)),
    )
    assert bool(ensemble.state_valid(result))
    return result


def _planner_and_model() -> tuple[EnsembleShortRolloutPlanner, WorldModelEnsembleState]:
    ensemble = _ensemble()
    model_state = _set_constant_outputs(
        ensemble,
        ensemble.init(jr.key(1, impl="threefry2x32")),
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
    return planner, model_state


def _calibration_batch(
    planner: EnsembleShortRolloutPlanner,
    model_state: WorldModelEnsembleState,
    *,
    action: int,
    decision: int,
) -> ImaginedRolloutBatch:
    revision = jnp.asarray((0, decision), dtype=jnp.uint32)
    bias = (20.0, -20.0) if action == 0 else (-20.0, 20.0)
    authority = planner.bind_authority(
        policy_weights=jnp.zeros((2, 2), dtype=jnp.float32),
        policy_bias=jnp.asarray(bias, dtype=jnp.float32),
        value_weights=jnp.zeros((2,), dtype=jnp.float32),
        value_bias=jnp.asarray(0.0, dtype=jnp.float32),
        action_support_counts=SUPPORT,
        source_revision_words=REVISION_ONE,
        model_state=model_state,
        policy_revision_words=revision,
        value_revision_words=revision,
    )
    planner_state = planner.init(
        jr.key(20 + decision, impl="threefry2x32"),
        model_state,
        authority,
    )
    anchor = planner.bind_real_anchor(
        jnp.asarray((float(decision), 0.0), dtype=jnp.float32),
        jnp.asarray((0, decision), dtype=jnp.uint32),
        authority,
    )
    result = planner.propose(planner_state, model_state, authority, anchor)
    assert bool(result.diagnostics.transaction_applied)
    assert bool(jnp.all(result.proposals.actions[result.proposals.transition_valid] == action))
    return result.proposals


def _grounded_gauge(
    planner: EnsembleShortRolloutPlanner,
    model_state: WorldModelEnsembleState,
) -> tuple[ImaginedRolloutSelectionGauge, ImaginedRolloutSelectionGaugeState]:
    action_zero = _calibration_batch(
        planner,
        model_state,
        action=0,
        decision=1,
    )
    action_one = _calibration_batch(
        planner,
        model_state,
        action=1,
        decision=2,
    )
    gauge = ImaginedRolloutSelectionGauge(
        planner,
        ImaginedRolloutSelectionGaugeConfig(
            audit_capacity=4,
            n_regions=1,
            min_evidence_count=1,
            min_realized_valid_fraction=1.0,
            max_mean_abs_reward_error=0.0,
            max_root_mean_square_next_observation_error=0.0,
            min_termination_accuracy=1.0,
            require_success_lcb=False,
            require_top_quantile_purity=False,
            max_authorizations=16,
        ),
    )
    state = gauge.init(action_zero)
    for record_id, batch in enumerate((action_zero, action_one), start=1):
        record = gauge.bind_grounded_record(
            batch,
            rollout_index=jnp.asarray(0, dtype=jnp.int32),
            step_index=jnp.asarray(0, dtype=jnp.int32),
            region_id=jnp.asarray(0, dtype=jnp.int32),
            record_id_words=jnp.asarray((0, record_id), dtype=jnp.uint32),
            realized_valid=jnp.asarray(True),
            realized_reward=batch.rewards[0, 0],
            realized_next_observation=batch.next_observations[0, 0],
            realized_terminated=batch.terminated[0, 0],
            realized_success=jnp.asarray(True),
        )
        result = gauge.record_grounded_outcome(state, record)
        assert bool(result.diagnostics.applied)
        state = result.state
    return gauge, state


def _system() -> tuple[
    GroundedImaginationComposition,
    GroundedImaginationCompositionState,
    WorldModelEnsembleState,
]:
    planner, model_state = _planner_and_model()
    gauge, gauge_state = _grounded_gauge(planner, model_state)
    actor_critic = AuthorizedImaginedRolloutActorCritic(
        gauge,
        ImaginedRolloutActorCriticConfig(
            initialization_scale=0.0,
            max_update_calls=16,
            max_backward_transitions=32,
        ),
    )
    composition = GroundedImaginationComposition(planner, gauge, actor_critic)
    state = composition.init(
        planner_key=jr.key(30, impl="threefry2x32"),
        actor_critic_key=jr.key(31, impl="threefry2x32"),
        model_state=model_state,
        action_support_counts=SUPPORT,
        source_revision_words=REVISION_ONE,
        grounded_gauge_state=gauge_state,
    )
    return composition, state, model_state


def _inputs(model_state: WorldModelEnsembleState, *, decision: int = 3) -> dict[str, object]:
    return {
        "model_state": model_state,
        "action_support_counts": SUPPORT,
        "source_revision_words": REVISION_ONE,
        "real_observation": jnp.asarray((float(decision), 0.0), dtype=jnp.float32),
        "decision_id_words": jnp.asarray((0, decision), dtype=jnp.uint32),
        "region_ids": jnp.zeros((1, 2), dtype=jnp.int32),
        "safety_admitted": jnp.ones((1, 2), dtype=jnp.bool_),
        "protected": jnp.zeros((1, 2), dtype=jnp.bool_),
    }


def test_exact_local_batch_is_authorized_and_committed_with_one_backward() -> None:
    composition, state, model_state = _system()
    assert "batch" not in inspect.signature(composition.step).parameters
    result = composition.step(state, **_inputs(model_state))  # type: ignore[arg-type]
    diagnostics = result.diagnostics

    assert bool(diagnostics.transaction_applied)
    assert bool(diagnostics.state_valid_before)
    assert bool(diagnostics.state_valid_after)
    assert bool(diagnostics.config_fingerprints_valid)
    assert bool(diagnostics.live_actor_policy_value_bound)
    assert bool(diagnostics.planner_transaction_applied)
    assert bool(diagnostics.planner_batch_nonempty)
    assert bool(diagnostics.planner_call_delta_exact)
    assert bool(diagnostics.planner_output_forwarded_directly)
    assert bool(diagnostics.exact_planner_batch_receipt_bound)
    assert not bool(diagnostics.caller_rollout_batch_input_available)
    assert bool(diagnostics.authorization_transaction_applied)
    assert bool(diagnostics.authorization_receipt_valid)
    assert bool(diagnostics.authorization_granted)
    assert bool(diagnostics.authorization_call_delta_exact)
    assert bool(diagnostics.proposal_valid)
    assert int(diagnostics.proposal_autodiff_pass_count) == 0
    assert bool(diagnostics.commit_preflight_valid)
    assert bool(diagnostics.commit_backward_work_performed)
    assert int(diagnostics.commit_autodiff_pass_count) == 1
    assert bool(diagnostics.actor_update_delta_exact)
    assert bool(diagnostics.actor_dream_update_delta_exact)
    assert bool(diagnostics.actor_backward_delta_exact)
    assert bool(diagnostics.commit_applied)
    assert bool(diagnostics.child_candidate_valid)
    assert not bool(diagnostics.real_environment_authenticated)
    assert not bool(diagnostics.model_support_authenticated)
    assert not bool(diagnostics.region_assignments_authenticated)
    assert not bool(diagnostics.safety_protection_authenticated)
    assert not bool(diagnostics.scientific_promotion_allowed)
    assert bool(composition.state_valid(result.state))

    assert jnp.array_equal(
        result.authorization_receipt.proposal_content_tag,
        composition.gauge.proposal_content_tag(result.imagined_batch),
    )
    assert jnp.array_equal(
        result.state.transaction_count_words,
        jnp.asarray((0, 1), dtype=jnp.uint32),
    )
    assert jnp.array_equal(
        result.state.planner_state.proposal_call_count_words,
        state.planner_state.proposal_call_count_words + jnp.asarray((0, 1), dtype=jnp.uint32),
    )
    assert jnp.array_equal(
        result.state.gauge_state.authorization_count_words,
        state.gauge_state.authorization_count_words + jnp.asarray((0, 1), dtype=jnp.uint32),
    )
    assert jnp.array_equal(
        result.state.actor_critic_state.update_count_words,
        jnp.asarray((0, 1), dtype=jnp.uint32),
    )
    assert int(result.commit_trace.backward_transition_count) == int(
        result.update_proposal.eligible_transition_count
    )
    assert not jnp.array_equal(
        result.state.actor_critic_state.actor_parameters.bias,
        state.actor_critic_state.actor_parameters.bias,
    )


def test_any_failed_stage_rolls_back_all_child_states_keys_and_clocks() -> None:
    composition, state, model_state = _system()
    blocked_inputs = _inputs(model_state)
    blocked_inputs["protected"] = jnp.ones((1, 2), dtype=jnp.bool_)
    blocked = composition.step(state, **blocked_inputs)  # type: ignore[arg-type]
    assert bool(blocked.diagnostics.planner_transaction_applied)
    assert bool(blocked.diagnostics.authorization_transaction_applied)
    assert not bool(blocked.diagnostics.authorization_granted)
    assert int(blocked.diagnostics.proposal_autodiff_pass_count) == 0
    assert int(blocked.diagnostics.commit_autodiff_pass_count) == 0
    assert not bool(blocked.diagnostics.transaction_applied)
    _assert_tree_equal(blocked.state, state)

    aliased_model = cast(
        WorldModelEnsembleState,
        model_state.replace(
            residual_variances=(
                model_state.residual_variances
                + jnp.asarray(1.0, dtype=jnp.float32)
            )
        ),
    )
    aliased = composition.step(
        state,
        **_inputs(aliased_model),  # type: ignore[arg-type]
    )
    assert not bool(aliased.diagnostics.planner_transaction_applied)
    assert int(aliased.diagnostics.commit_autodiff_pass_count) == 0
    assert not bool(aliased.diagnostics.transaction_applied)
    _assert_tree_equal(aliased.state, state)

    tampered = cast(
        GroundedImaginationCompositionState,
        state.replace(
            actor_critic_state=state.actor_critic_state.replace(
                actor_parameters=state.actor_critic_state.actor_parameters.replace(
                    bias=state.actor_critic_state.actor_parameters.bias.at[0].set(9.0)
                )
            )
        ),
    )
    assert not bool(composition.state_valid(tampered))
    rejected = composition.step(
        tampered,
        **_inputs(model_state),  # type: ignore[arg-type]
    )
    assert not bool(rejected.diagnostics.state_valid_before)
    assert not bool(rejected.diagnostics.planner_transaction_applied)
    assert int(rejected.diagnostics.commit_autodiff_pass_count) == 0
    _assert_tree_equal(rejected.state, tampered)


def test_eager_jit_and_repeated_lineage_are_exact() -> None:
    composition, state, model_state = _system()
    inputs = _inputs(model_state)
    with jax.disable_jit():
        eager = composition.step(state, **inputs)  # type: ignore[arg-type]
    compiled = jax.jit(
        lambda current: composition.step(
            current,
            **inputs,  # type: ignore[arg-type]
        )
    )(state)
    _assert_tree_close(eager, compiled)
    assert bool(compiled.diagnostics.transaction_applied)

    second = composition.step(
        compiled.state,
        **_inputs(model_state, decision=4),  # type: ignore[arg-type]
    )
    assert bool(second.diagnostics.transaction_applied)
    assert jnp.array_equal(
        second.state.transaction_count_words,
        jnp.asarray((0, 2), dtype=jnp.uint32),
    )
    assert jnp.array_equal(
        second.state.planner_state.bound_policy_revision_words,
        jnp.asarray((0, 2), dtype=jnp.uint32),
    )
    assert jnp.array_equal(
        second.state.actor_critic_state.update_count_words,
        jnp.asarray((0, 2), dtype=jnp.uint32),
    )


def test_config_resources_and_complete_checkpoint_are_fail_closed(tmp_path: Path) -> None:
    composition, state, model_state = _system()
    result = composition.step(state, **_inputs(model_state))  # type: ignore[arg-type]
    assert bool(result.diagnostics.transaction_applied)

    config = composition.to_config()
    assert config["mechanism_status"] == GROUNDED_IMAGINATION_COMPOSITION_MECHANISM_STATUS
    assert config["evidence_level"] == GROUNDED_IMAGINATION_COMPOSITION_EVIDENCE_LEVEL
    assert config["scientific_promotion_allowed"] is False
    assert config["caller_rollout_batch_input_available"] is False
    assert (
        config["policy_value_snapshot_revision_rule"]
        == "actor_critic_update_count_plus_one_no_update"
    )
    assert config["real_environment_authenticated"] is False
    assert config["model_support_authenticated"] is False
    assert config["region_assignments_authenticated"] is False
    assert config["safety_protection_authenticated"] is False
    assert GROUNDED_IMAGINATION_COMPOSITION_SCIENTIFIC_PROMOTION_ALLOWED is False
    assert GROUNDED_IMAGINATION_REAL_ENVIRONMENT_AUTHENTICATED is False
    assert GROUNDED_IMAGINATION_MODEL_SUPPORT_AUTHENTICATED is False
    assert GROUNDED_IMAGINATION_REGION_ASSIGNMENTS_AUTHENTICATED is False
    assert GROUNDED_IMAGINATION_SAFETY_PROTECTION_AUTHENTICATED is False
    restored_config = GroundedImaginationComposition.from_config(config)
    assert restored_config.to_config() == config
    with pytest.raises(ValueError, match="fields"):
        GroundedImaginationComposition.from_config({**config, "extra": True})

    budget = composition.resource_budget
    state_nbytes = sum(
        np.asarray(leaf).nbytes
        for leaf in jax.tree.leaves(_materialize_keys(result.state))
    )
    assert budget.persistent_state_bytes == state_nbytes
    assert budget.persistent_typed_prng_keys == 1
    assert budget.config_fingerprint_uint32_scalars == 32
    assert budget.max_planner_calls_per_call == 1
    assert budget.max_authorization_calls_per_call == 1
    assert budget.max_actor_critic_proposals_per_call == 1
    assert budget.proposal_autodiff_passes == 0
    assert budget.max_actor_critic_commits_per_call == 1
    assert budget.max_autodiff_passes_per_call == 1
    assert budget.accepted_call_autodiff_passes == 1
    assert budget.caller_rollout_batch_inputs == 0
    assert budget.model_state_owned == 0
    assert budget.dispatch_authority == 0
    assert budget.safety_authority == 0
    assert budget.output_authority == 0
    assert budget.scientific_promotion_allowed is False

    path = tmp_path / "grounded-imagination"
    save_grounded_imagination_composition_checkpoint(composition, result.state, path)
    loaded_composition, loaded_state = (
        load_grounded_imagination_composition_checkpoint(path)
    )
    assert loaded_composition.to_config() == composition.to_config()
    _assert_tree_equal(loaded_state, result.state)
    assert bool(loaded_composition.state_valid(loaded_state))

    corrupt = cast(
        GroundedImaginationCompositionState,
        result.state.replace(
            composition_config_fingerprint=(
                result.state.composition_config_fingerprint.at[0].add(1)
            )
        ),
    )
    assert not bool(composition.state_valid(corrupt))
    with pytest.raises(ValueError, match="invalid"):
        save_grounded_imagination_composition_checkpoint(composition, corrupt, tmp_path / "bad")


def test_complete_composition_is_exported_from_both_public_roots() -> None:
    expected = {
        "GROUNDED_IMAGINATION_COMPOSITION_CHECKPOINT_SCHEMA",
        "GROUNDED_IMAGINATION_COMPOSITION_CONFIG_SCHEMA",
        "GROUNDED_IMAGINATION_COMPOSITION_EVIDENCE_LEVEL",
        "GROUNDED_IMAGINATION_COMPOSITION_MECHANISM_STATUS",
        "GROUNDED_IMAGINATION_COMPOSITION_SCIENTIFIC_PROMOTION_ALLOWED",
        "GROUNDED_IMAGINATION_MODEL_SUPPORT_AUTHENTICATED",
        "GROUNDED_IMAGINATION_REAL_ENVIRONMENT_AUTHENTICATED",
        "GROUNDED_IMAGINATION_REGION_ASSIGNMENTS_AUTHENTICATED",
        "GROUNDED_IMAGINATION_SAFETY_PROTECTION_AUTHENTICATED",
        "GroundedImaginationComposition",
        "GroundedImaginationCompositionDiagnostics",
        "GroundedImaginationCompositionResourceBudget",
        "GroundedImaginationCompositionResult",
        "GroundedImaginationCompositionState",
        "load_grounded_imagination_composition_checkpoint",
        "save_grounded_imagination_composition_checkpoint",
    }
    for public_module in (public_root, public_core):
        assert expected <= set(public_module.__all__)
        for name in expected:
            assert getattr(public_module, name) is getattr(composition_module, name)
