"""Causal online bridge contracts for the noisy world and integrated agent."""

from __future__ import annotations

import copy
import dataclasses
import functools

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.grounded_joint_world_model import (
    GroundedJointWorldModelConfig,
)
from alberta_framework.core.integrated_hidden_partner import (
    ACTIVE_PAIR_SLOTS,
    CANDIDATE_PAIR_SLOTS,
    DEPLOYED_FEATURE_DIM,
    INTEGRATED_DECISION_CACHE_CHECK_ORDER,
    RAW_OBSERVATION_DIM,
    IntegratedHiddenPartnerAgent,
    IntegratedHiddenPartnerConfig,
    IntegratedUpdateResult,
)
from alberta_framework.core.representation_gradient_mixer import (
    RepresentationGradientMixerConfig,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_controls import (
    build_v6_full_agent_config,
)
from alberta_framework.evaluation.hidden_partner_world_filter import (
    HiddenPartnerWorldBayesFilter,
    HiddenPartnerWorldFilterConfig,
)
from alberta_framework.evaluation.hidden_partner_world_online_bridge import (
    HIDDEN_PARTNER_WORLD_ONLINE_BRIDGE_CONFIG_SCHEMA,
    HiddenPartnerWorldOnlineBridge,
    HiddenPartnerWorldOnlineResourceBudget,
    HiddenPartnerWorldOnlineState,
    HiddenPartnerWorldOnlineStep,
    LearnerHiddenPartnerWorldTransition,
    strip_hidden_partner_world_oracle,
)
from alberta_framework.streams.hidden_partner_world_feedback import (
    CUE_1_INDEX,
    CUE_2_INDEX,
    HiddenPartnerWorldFeedbackConfig,
    HiddenPartnerWorldFeedbackWorld,
)

pytestmark = pytest.mark.unit

_INT32_MAX = 2**31 - 1


@functools.lru_cache(maxsize=1)
def _shared_bridge() -> HiddenPartnerWorldOnlineBridge:
    return HiddenPartnerWorldOnlineBridge()


def _balanced_bridge(*, initial_action: int = 0) -> HiddenPartnerWorldOnlineBridge:
    agent = IntegratedHiddenPartnerAgent(
        dataclasses.replace(
            IntegratedHiddenPartnerConfig(),
            action_selection_mode="externally_forced",
        )
    )
    return HiddenPartnerWorldOnlineBridge(
        agent=agent,
        focal_action_policy="balanced_external",
        initial_external_action=initial_action,
    )


def _grounded_bridge() -> HiddenPartnerWorldOnlineBridge:
    agent = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(
            grounded_world_model=GroundedJointWorldModelConfig(
                representation_dim=DEPLOYED_FEATURE_DIM,
                target_observation_dim=RAW_OBSERVATION_DIM,
                n_focal_actions=2,
                n_partner_actions=2,
                step_size=0.2,
                initialization_scale=0.05,
                max_input_magnitude=100.0,
                max_parameter_magnitude=100.0,
            ),
            representation_gradient_mixer=RepresentationGradientMixerConfig(
                representation_dim=DEPLOYED_FEATURE_DIM,
                mode="full",
            ),
            feature_lifecycle_enabled=False,
            replacement_interval=0,
        )
    )
    return HiddenPartnerWorldOnlineBridge(agent=agent)


def _agent_result_for_state(
    bridge: HiddenPartnerWorldOnlineBridge,
    state: HiddenPartnerWorldOnlineState,
) -> IntegratedUpdateResult:
    world_transition, _ = bridge.world.step(state.world, state.action)
    learner_transition = strip_hidden_partner_world_oracle(world_transition)
    if bridge.focal_action_policy == "balanced_external":
        next_action = jnp.bitwise_xor(
            jnp.asarray(bridge.initial_external_action, dtype=jnp.int32),
            jnp.bitwise_and(state.step_count + 1, jnp.asarray(1, dtype=jnp.int32)),
        )
        return bridge.agent.update_with_forced_next_action(
            state.agent,
            learner_transition,
            next_action,
        )
    return bridge.agent.update(state.agent, learner_transition)


def _force_next_interaction_promotion(
    bridge: HiddenPartnerWorldOnlineBridge,
    state: HiddenPartnerWorldOnlineState,
    pair: tuple[int, int] = (4, 5),
) -> HiddenPartnerWorldOnlineState:
    clock_63 = jnp.asarray((0, 63), dtype=jnp.uint32)
    clock_64 = jnp.asarray((0, 64), dtype=jnp.uint32)
    # Build one valid non-birth world observation, then restart the otherwise
    # untrained agent on exactly that ordinary observation.  This keeps the
    # decision cache causal while the test moves every clock owner to the
    # synthetic pre-step-64 curation boundary.
    world_state = state.world.replace(
        step_count=jnp.asarray(63, dtype=jnp.int32),
        step_words=clock_63,
        previous_outcome=jnp.asarray(1.0, dtype=jnp.float32),
        previous_partner_action=jnp.asarray(1, dtype=jnp.int32),
        has_partner_history=jnp.asarray(True, dtype=jnp.bool_),
    )
    restarted = bridge.agent.start(
        bridge.world.observe(world_state),
        state.agent.current_selection.rng_key_before,
    )
    interaction = restarted.state.interaction
    matching = (interaction.candidate_left == pair[0]) & (
        interaction.candidate_right == pair[1]
    )
    assert int(jnp.sum(matching)) == 1
    candidate_index = int(jnp.argmax(matching))
    interaction = interaction.replace(
        step_count=jnp.asarray(63, dtype=jnp.int32),
        step_words=clock_63,
        replacement_phase=jnp.asarray(63, dtype=jnp.int32),
        ages=jnp.full((ACTIVE_PAIR_SLOTS,), 256, dtype=jnp.int32),
        utilities=jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.float32),
        candidate_ages=jnp.full(
            (CANDIDATE_PAIR_SLOTS,),
            128,
            dtype=jnp.int32,
        ),
        candidate_utilities=(
            jnp.zeros((CANDIDATE_PAIR_SLOTS,), dtype=jnp.float32)
            .at[candidate_index]
            .set(10.0)
        ),
    )
    agent_state = restarted.state.replace(
        interaction=interaction,
        state_builder=restarted.state.state_builder.replace(
            step_count=jnp.asarray(64, dtype=jnp.int32),
            step_words=clock_64,
            update_count=jnp.asarray(63, dtype=jnp.int32),
            update_words=clock_63,
        ),
        behavior=restarted.state.behavior.replace(
            step_count=jnp.asarray(63, dtype=jnp.int32),
            step_words=clock_63,
        ),
        joint_world=restarted.state.joint_world.replace(
            step_count=jnp.asarray(63, dtype=jnp.int32),
            step_words=clock_63,
        ),
        control=restarted.state.control.replace(
            step_count=jnp.asarray(63, dtype=jnp.int32),
            step_words=clock_63,
        ),
        router=dataclasses.replace(
            restarted.state.router,
            route_count=jnp.asarray(63, dtype=jnp.int32),
            route_words=clock_63,
        ),
        step_count=jnp.asarray(63, dtype=jnp.int32),
        step_words=clock_63,
    )
    # The bridge authenticates one shared transition identity.  Move every
    # owner to the same synthetic pre-step-64 point; changing only the child
    # lifecycle clock would correctly fail the global transaction.
    return state.replace(
        world=world_state,
        agent=agent_state,
        world_filter=state.world_filter.replace(
            step_count=jnp.asarray(63, dtype=jnp.int32)
        ),
        action=restarted.action,
        step_count=jnp.asarray(63, dtype=jnp.int32),
    )


def _assert_grounded_mechanism_neutral(mechanism: object) -> None:
    assert not bool(mechanism.grounded_enabled)  # type: ignore[attr-defined]
    assert int(mechanism.grounded_executed_joint_index) == -1  # type: ignore[attr-defined]
    for name in (
        "grounded_feature_contribution",
        "grounded_row_bias",
        "grounded_raw_predictions",
        "grounded_targets",
        "grounded_errors",
        "grounded_fit_loss_by_head",
        "grounded_representation_loss_by_head",
        "grounded_representation_gradient",
        "grounded_representation_gradient_by_head",
        "grounded_representation_gradient_norm_by_head",
        "grounded_executed_weight_row_delta_norm_by_head",
        "grounded_executed_bias_row_delta_by_head",
        "grounded_credit_gradient_chi",
        "mixed_credit_gradient_chi",
        "mixed_credit_gradient_phi",
    ):
        value = np.asarray(getattr(mechanism, name))
        np.testing.assert_array_equal(value, np.zeros_like(value))
    for name in (
        "grounded_proposed_weight_row_bit_change_mask",
        "grounded_proposed_bias_row_bit_change_mask",
    ):
        assert not bool(jnp.any(getattr(mechanism, name)))
    for name in (
        "grounded_prediction_valid",
        "grounded_target_valid",
        "grounded_gradient_valid",
        "grounded_learning_enabled",
        "grounded_prediction_matches_decision",
        "grounded_row_update_isolated",
        "grounded_update_applied",
        "grounded_credit_valid",
        "mixed_credit_valid",
        "mixed_credit_applied",
        "mixed_credit_conflict",
    ):
        assert not bool(getattr(mechanism, name))


def _assert_mechanism_fully_neutral(mechanism: object) -> None:
    assert not bool(mechanism.valid)  # type: ignore[attr-defined]
    _assert_grounded_mechanism_neutral(mechanism)
    for name in (
        "lifecycle_pre_descriptors",
        "lifecycle_proposal_descriptors",
        "lifecycle_applied_descriptors",
    ):
        np.testing.assert_array_equal(
            getattr(mechanism, name),
            np.full((ACTIVE_PAIR_SLOTS, 2), -1, dtype=np.int32),
        )
    for name in (
        "lifecycle_proposal_replaced_slot",
        "lifecycle_proposal_promoted_candidate",
        "lifecycle_proposal_refreshed_candidate",
        "lifecycle_proposal_retired_slot",
        "lifecycle_proposal_retired_left",
        "lifecycle_proposal_retired_right",
        "lifecycle_applied_replaced_slot",
        "lifecycle_applied_promoted_candidate",
        "lifecycle_applied_refreshed_candidate",
        "lifecycle_applied_retired_slot",
        "lifecycle_applied_retired_left",
        "lifecycle_applied_retired_right",
        "random_curation_selected_active_worst_slot",
        "random_curation_selected_promotion_candidate",
        "random_curation_selected_refresh_candidate",
    ):
        assert int(getattr(mechanism, name)) == -1
    np.testing.assert_array_equal(
        mechanism.router_source_slots,  # type: ignore[attr-defined]
        np.full((ACTIVE_PAIR_SLOTS,), -1, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        mechanism.random_curation_active_priorities,  # type: ignore[attr-defined]
        np.zeros((ACTIVE_PAIR_SLOTS,), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        mechanism.random_curation_candidate_priorities,  # type: ignore[attr-defined]
        np.zeros((CANDIDATE_PAIR_SLOTS,), dtype=np.float32),
    )
    for name, shape, dtype in (
        ("lifecycle_relevance_probe_scores", (ACTIVE_PAIR_SLOTS,), np.float32),
        (
            "lifecycle_relevance_probe_errors",
            (1, ACTIVE_PAIR_SLOTS),
            np.float32,
        ),
        (
            "lifecycle_candidate_promotion_signal",
            (CANDIDATE_PAIR_SLOTS,),
            np.float32,
        ),
        (
            "lifecycle_candidate_promotion_evidence_streak_pre",
            (CANDIDATE_PAIR_SLOTS,),
            np.int32,
        ),
        (
            "lifecycle_candidate_promotion_evidence_streak_updated",
            (CANDIDATE_PAIR_SLOTS,),
            np.int32,
        ),
        (
            "lifecycle_candidate_promotion_evidence_streak_proposal_post",
            (CANDIDATE_PAIR_SLOTS,),
            np.int32,
        ),
        (
            "lifecycle_candidate_promotion_evidence_streak_post",
            (CANDIDATE_PAIR_SLOTS,),
            np.int32,
        ),
    ):
        np.testing.assert_array_equal(
            getattr(mechanism, name),
            np.zeros(shape, dtype=dtype),
        )
    for leaf in jax.tree_util.tree_leaves(mechanism):
        value = np.asarray(leaf)
        if value.dtype == np.bool_:
            assert not bool(np.any(value))
    absent_fields = {
        "grounded_executed_joint_index",
        "lifecycle_pre_descriptors",
        "lifecycle_proposal_descriptors",
        "lifecycle_applied_descriptors",
        "lifecycle_proposal_replaced_slot",
        "lifecycle_proposal_promoted_candidate",
        "lifecycle_proposal_refreshed_candidate",
        "lifecycle_proposal_retired_slot",
        "lifecycle_proposal_retired_left",
        "lifecycle_proposal_retired_right",
        "lifecycle_applied_replaced_slot",
        "lifecycle_applied_promoted_candidate",
        "lifecycle_applied_refreshed_candidate",
        "lifecycle_applied_retired_slot",
        "lifecycle_applied_retired_left",
        "lifecycle_applied_retired_right",
        "random_curation_selected_active_worst_slot",
        "random_curation_selected_promotion_candidate",
        "random_curation_selected_refresh_candidate",
        "router_source_slots",
    }
    for field in dataclasses.fields(mechanism):
        value = np.asarray(getattr(mechanism, field.name))
        if value.dtype == np.bool_:
            assert not bool(np.any(value))
        elif field.name in absent_fields:
            np.testing.assert_array_equal(value, np.full_like(value, -1))
        else:
            np.testing.assert_array_equal(value, np.zeros_like(value))


def _assert_oracle_schedule_neutral(trace: object) -> None:
    for name in (
        "oracle_step_count",
        "oracle_cycle_index",
        "oracle_cycle_step",
        "oracle_cycle_length",
        "oracle_segment_index",
        "oracle_segment_step",
        "oracle_segment_length",
        "oracle_regime_id",
        "oracle_next_cycle_index",
        "oracle_next_segment_index",
        "oracle_next_regime_id",
        "oracle_partner_intended_action",
    ):
        assert int(getattr(trace, name)) == -1
    for name in (
        "oracle_world_sign",
        "oracle_next_world_sign",
        "oracle_partner_intended_sign",
        "oracle_focal_action_sign",
        "oracle_partner_action_sign",
    ):
        assert float(getattr(trace, name)) == 0.0
    assert not bool(trace.oracle_schedule_switched)  # type: ignore[attr-defined]
    assert not bool(trace.oracle_partner_flipped)  # type: ignore[attr-defined]
    assert not bool(jnp.any(trace.oracle_world_cue_flipped))  # type: ignore[attr-defined]
    assert not bool(jnp.any(trace.oracle_next_world_cue_flipped))  # type: ignore[attr-defined]


def _unwrap_prng_keys(tree: object) -> object:
    def unwrap(value: object) -> object:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            return jr.key_data(value)  # type: ignore[arg-type]
        return value

    return jax.tree_util.tree_map(unwrap, tree)


def _assert_atomic_rejection(
    before: HiddenPartnerWorldOnlineState,
    rejected: HiddenPartnerWorldOnlineStep,
) -> None:
    assert not bool(rejected.trace.accepted)
    assert not bool(rejected.state.valid)
    expected = before.replace(valid=jnp.asarray(False, dtype=jnp.bool_))
    chex.assert_trees_all_equal(
        _unwrap_prng_keys(rejected.state),
        _unwrap_prng_keys(expected),
    )
    assert int(rejected.trace.committed_world_step_delta) == 0
    assert int(rejected.trace.committed_agent_step_delta) == 0
    assert int(rejected.trace.committed_filter_step_delta) == 0
    assert int(rejected.trace.committed_bridge_step_delta) == 0


def test_oracle_strip_is_exact_and_oracle_mutation_cannot_change_learner_transition() -> None:
    world = HiddenPartnerWorldFeedbackWorld()
    state = world.init(jr.key(1))
    transition, _ = world.step(state, jnp.asarray(1, dtype=jnp.int32))
    stripped = strip_hidden_partner_world_oracle(transition)
    adversarial = transition.replace(
        oracle=transition.oracle.replace(
            world_sign=-transition.oracle.world_sign,
            regime_id=jnp.asarray(99, dtype=jnp.int32),
            counterfactual_rewards=1.0 - transition.oracle.counterfactual_rewards,
        )
    )
    adversarial_stripped = strip_hidden_partner_world_oracle(adversarial)

    assert isinstance(stripped, LearnerHiddenPartnerWorldTransition)
    assert not hasattr(stripped, "oracle")
    assert set(field.name for field in dataclasses.fields(stripped)) == {
        "observation",
        "focal_action",
        "partner_action",
        "reward",
        "outcome",
        "next_observation",
        "terminated",
        "discount",
    }
    chex.assert_trees_all_equal(stripped, adversarial_stripped)


def test_online_bridge_advances_world_agent_and_filter_once_in_causal_order() -> None:
    bridge = _shared_bridge()
    state = bridge.initialize(jr.key(2), jr.key(3))
    pre_cells = bridge.world_filter.expected_reward_cells(
        state.world_filter.posterior_mean
    )
    pre_decision = bridge.world_filter.marginalize_partner(
        pre_cells,
        state.agent.current_evaluation.partner_probabilities,
    )
    step = bridge.step(state)

    assert bool(state.valid)
    assert bool(step.trace.active)
    assert bool(step.trace.accepted)
    assert int(step.trace.step) == 0
    assert int(step.trace.focal_action) == int(state.action)
    assert int(step.trace.next_action) == int(step.state.action)
    assert int(step.trace.committed_world_step_delta) == 1
    assert int(step.trace.committed_agent_step_delta) == 1
    assert int(step.trace.committed_filter_step_delta) == 1
    assert int(step.trace.committed_bridge_step_delta) == 1
    assert int(step.trace.proposed_world_step_delta) == 1
    assert int(step.trace.proposed_agent_step_delta) == 1
    assert int(step.trace.proposed_filter_step_delta) == 1
    assert int(step.trace.proposed_bridge_step_delta) == 1
    assert int(step.state.step_count) == 1
    assert int(step.state.world.step_count) == 1
    assert int(step.state.agent.step_count) == 1
    assert int(step.state.world_filter.step_count) == 1
    assert bool(step.trace.entry_state_contract_valid)
    assert bool(step.trace.config_token_valid)
    assert bool(step.trace.counters_synchronized)
    assert bool(step.trace.action_valid)
    assert bool(step.trace.filter_entry_valid)
    assert bool(step.trace.proposed_agent_update_valid)
    assert bool(step.trace.proposed_filter_update_valid)
    assert bool(step.trace.proposed_filter_decision_valid)
    assert bool(step.trace.next_selection_diagnostics_valid)
    assert bool(step.trace.oracle_trace_valid)
    assert bool(step.trace.all_finite)
    chex.assert_trees_all_equal(step.trace.observation_pre, state.agent.raw_observation)
    chex.assert_trees_all_equal(step.trace.next_observation, step.state.agent.raw_observation)
    corrected = (
        step.trace.outcome
        * (2.0 * step.trace.focal_action.astype(jnp.float32) - 1.0)
        * (2.0 * step.trace.partner_action.astype(jnp.float32) - 1.0)
    )
    chex.assert_trees_all_equal(step.trace.corrected_outcome, corrected)
    expected_filter = bridge.world_filter.advance(
        state.world_filter,
        corrected,
        step.trace.next_observation[jnp.asarray((CUE_1_INDEX, CUE_2_INDEX))],
    )
    chex.assert_trees_all_close(
        step.state.world_filter,
        expected_filter.state,
        atol=1e-6,
        rtol=0.0,
    )
    chex.assert_trees_all_equal(
        step.trace.agent_partner_belief_conditioned_reward_cells,
        pre_cells.rewards,
    )
    chex.assert_trees_all_equal(
        step.trace.agent_partner_belief_conditioned_expected_rewards,
        pre_decision.expected_rewards,
    )
    assert (
        int(step.trace.agent_partner_belief_conditioned_greedy_action)
        == int(pre_decision.greedy_action)
    )
    assert 0.0 <= float(
        step.trace.agent_partner_belief_conditioned_selected_regret
    ) <= 1.0
    assert step.trace.agent_partner_belief_conditioned_reward_cells.shape == (2, 2)
    assert step.trace.agent_partner_belief_conditioned_expected_rewards.shape == (2,)
    np.testing.assert_allclose(
        np.sum(np.asarray(step.trace.agent_applied_partner_probabilities)),
        1.0,
        atol=1e-6,
        rtol=0.0,
    )


def test_v6_full_agent_cache_contract_survives_fused_probability_reduction() -> None:
    world = HiddenPartnerWorldFeedbackWorld()
    agent = IntegratedHiddenPartnerAgent(build_v6_full_agent_config())
    world_state = world.init(jr.key(303))
    start = agent.start(world.observe(world_state), jr.key(404))
    agent_state = start.state
    action = start.action

    for step_index in range(12):
        transition, world_state = world.step(world_state, action)
        update = agent.update(
            agent_state,
            strip_hidden_partner_world_oracle(transition),
        )
        checks = np.asarray(update.diagnostics.decision_cache_check_vector)
        failed = [
            name
            for name, valid in zip(
                INTEGRATED_DECISION_CACHE_CHECK_ORDER,
                checks,
                strict=True,
            )
            if not valid
        ]
        assert checks.shape == (len(INTEGRATED_DECISION_CACHE_CHECK_ORDER),)
        assert not failed, f"step {step_index} cache failures: {failed}"
        assert bool(update.diagnostics.transition_semantics_valid)
        agent_state = update.state
        action = update.action

    assert int(agent_state.step_count) == 12


def test_bridge_constructor_rejects_types_before_reading_nested_properties() -> None:
    with pytest.raises(TypeError, match="world"):
        HiddenPartnerWorldOnlineBridge(world=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="agent"):
        HiddenPartnerWorldOnlineBridge(agent=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="world_filter"):
        HiddenPartnerWorldOnlineBridge(world_filter=object())  # type: ignore[arg-type]


def test_bridge_constructor_rejects_world_filter_probability_mismatch() -> None:
    world = HiddenPartnerWorldFeedbackWorld(
        HiddenPartnerWorldFeedbackConfig(world_flip_probability=0.02)
    )
    mismatched = HiddenPartnerWorldBayesFilter(HiddenPartnerWorldFilterConfig())
    with pytest.raises(ValueError, match="probability contracts"):
        HiddenPartnerWorldOnlineBridge(world=world, world_filter=mismatched)


@pytest.mark.parametrize("key_name", ("world_key", "agent_key"))
def test_bridge_initialization_requires_exact_threefry_key_shape(key_name: str) -> None:
    keys = {
        "world_key": jr.key(8),
        "agent_key": jr.key(9),
    }
    keys[key_name] = jr.key(7, impl="rbg")

    with pytest.raises(TypeError, match="threefry2x32 uint32\\[2\\]"):
        HiddenPartnerWorldOnlineBridge().initialize(**keys)


def test_agent_cache_rejection_atomically_latches_bridge_without_advancing_world() -> None:
    bridge = _shared_bridge()
    state = bridge.initialize(jr.key(4), jr.key(5))
    corrupt_agent = state.agent.replace(
        behavior=state.agent.behavior.replace(
            weights=state.agent.behavior.weights.at[0, 0].add(0.25)
        )
    )
    corrupt = state.replace(agent=corrupt_agent)
    rejected = bridge.step(corrupt)

    assert not bool(rejected.trace.accepted)
    assert not bool(rejected.trace.proposed_agent_update_valid)
    _assert_atomic_rejection(corrupt, rejected)
    assert int(rejected.trace.proposed_world_step_delta) == 1
    assert int(rejected.trace.proposed_filter_step_delta) == 1
    assert int(rejected.trace.proposed_agent_step_delta) == 0
    np.testing.assert_array_equal(
        rejected.trace.agent_partner_belief_conditioned_reward_cells,
        np.full((2, 2), 0.5, dtype=np.float32),
    )
    assert int(rejected.trace.oracle_regime_id) == -1

    blocked = jax.jit(bridge.step)(rejected.state)
    assert not bool(blocked.trace.active)
    assert not bool(blocked.trace.accepted)
    chex.assert_trees_all_equal(
        _unwrap_prng_keys(blocked.state),
        _unwrap_prng_keys(rejected.state),
    )


def test_bridge_config_roundtrip_token_authority_and_tamper_rejection() -> None:
    bridge = _shared_bridge()
    payload = bridge.to_config()
    restored = HiddenPartnerWorldOnlineBridge.from_config(payload)

    assert payload["schema"] == HIDDEN_PARTNER_WORLD_ONLINE_BRIDGE_CONFIG_SCHEMA
    assert payload["development_only"] is True
    assert payload["execution_authorized"] is False
    assert payload["evidence_authorized"] is False
    assert payload["scientific_promotion_allowed"] is False
    assert restored.to_config() == payload
    assert restored.config_token_hex == bridge.config_token_hex
    assert len(bridge.config_token_hex) == 64

    for field in (
        "execution_authorized",
        "evidence_authorized",
        "scientific_promotion_allowed",
    ):
        hostile = copy.deepcopy(payload)
        hostile[field] = True
        with pytest.raises(ValueError):
            HiddenPartnerWorldOnlineBridge.from_config(hostile)
    extra = copy.deepcopy(payload)
    extra["extra"] = False
    with pytest.raises(ValueError, match="fields"):
        HiddenPartnerWorldOnlineBridge.from_config(extra)


def test_resource_budget_is_exact_includes_bridge_metadata_and_zero_replay() -> None:
    bridge = _shared_bridge()
    state = bridge.initialize(jr.key(10), jr.key(11))
    budget = bridge.resource_budget(state)

    assert isinstance(budget, HiddenPartnerWorldOnlineResourceBudget)
    assert budget.world_state_nbytes == bridge.world.resource_budget.state_nbytes
    assert budget.agent_state_nbytes == bridge.agent.resource_budget(state.agent).total_state_nbytes
    assert budget.filter_state_nbytes == 9
    assert budget.config_token_nbytes == 32
    assert budget.action_nbytes == 4
    assert budget.valid_nbytes == 1
    assert budget.step_count_nbytes == 4
    assert budget.bridge_metadata_nbytes == 41
    assert budget.component_state_nbytes == (
        budget.world_state_nbytes + budget.agent_state_nbytes + budget.filter_state_nbytes
    )
    assert budget.total_state_nbytes == budget.component_state_nbytes + 41
    assert budget.world_replay_capacity == 0
    assert budget.agent_replay_capacity == 0
    assert budget.replay_capacity == 0
    assert budget.to_dict()["total_state_nbytes"] == budget.total_state_nbytes


def test_bridge_v3_policy_config_is_strict_token_bound_and_mode_matched() -> None:
    balanced = _balanced_bridge(initial_action=1)
    payload = balanced.to_config()
    restored = HiddenPartnerWorldOnlineBridge.from_config(payload)

    assert payload["focal_action_policy"] == "balanced_external"
    assert payload["initial_external_action"] == 1
    assert restored.to_config() == payload
    assert restored.config_token_hex == balanced.config_token_hex
    assert restored.focal_action_policy == "balanced_external"
    assert restored.initial_external_action == 1
    assert restored.config_token_hex != _shared_bridge().config_token_hex

    with pytest.raises(ValueError, match="action_selection_mode"):
        HiddenPartnerWorldOnlineBridge(focal_action_policy="balanced_external")
    forced_agent = balanced.agent
    with pytest.raises(ValueError, match="action_selection_mode"):
        HiddenPartnerWorldOnlineBridge(agent=forced_agent)
    with pytest.raises(ValueError, match="canonical"):
        HiddenPartnerWorldOnlineBridge(initial_external_action=1)
    for hostile_initial in (True, 0.0, np.int32(0), 2):
        with pytest.raises(ValueError, match="initial_external_action"):
            HiddenPartnerWorldOnlineBridge(
                initial_external_action=hostile_initial,  # type: ignore[arg-type]
            )

    for field, value in (
        ("focal_action_policy", "agent"),
        ("initial_external_action", False),
    ):
        hostile = copy.deepcopy(payload)
        hostile[field] = value
        with pytest.raises(ValueError):
            HiddenPartnerWorldOnlineBridge.from_config(hostile)


def test_balanced_initialization_preserves_ordinary_policy_rng_and_state_budget() -> None:
    world_key = jr.key(210)
    agent_key = jr.key(211)
    ordinary = _shared_bridge()
    balanced = _balanced_bridge(initial_action=1)
    ordinary_state = ordinary.initialize(world_key, agent_key)
    balanced_state = balanced.initialize(world_key, agent_key)
    ordinary_selection = ordinary_state.agent.current_selection
    balanced_selection = balanced_state.agent.current_selection

    assert int(balanced_state.action) == 1
    assert bool(balanced_selection.externally_forced)
    assert not bool(ordinary_selection.externally_forced)
    for field in (
        "noisy_greedy_action",
        "random_action",
        "explored",
    ):
        chex.assert_trees_all_equal(
            getattr(balanced_selection, field),
            getattr(ordinary_selection, field),
        )
    np.testing.assert_array_equal(
        jr.key_data(balanced_selection.rng_key_before),
        jr.key_data(ordinary_selection.rng_key_before),
    )
    np.testing.assert_array_equal(
        jr.key_data(balanced_selection.rng_key_after),
        jr.key_data(ordinary_selection.rng_key_after),
    )

    ordinary_budget = ordinary.resource_budget(ordinary_state)
    balanced_budget = balanced.resource_budget(balanced_state)
    # Exact stream/agent lifetime words intentionally raised the fixed v2
    # footprint; keep the measured golden explicit so later hidden growth is
    # still caught.
    assert ordinary_budget.total_state_nbytes == 7_040
    assert balanced_budget.total_state_nbytes == ordinary_budget.total_state_nbytes
    assert balanced_budget.bridge_metadata_nbytes == 41
    assert _tree_signature(balanced_state) == _tree_signature(ordinary_state)


def _tree_signature(tree: object) -> tuple[tuple[tuple[int, ...], str], ...]:
    return tuple(
        (tuple(leaf.shape), str(leaf.dtype))
        for leaf in jax.tree_util.tree_leaves(tree)
    )


def test_balanced_external_actions_are_exact_under_jit_scan_with_rng_replay() -> None:
    bridge = _balanced_bridge()
    initial = bridge.initialize(jr.key(212), jr.key(213))

    def body(state, _):
        result = bridge.step(state)
        return result.state, result.trace

    final, traces = jax.jit(
        lambda state: jax.lax.scan(
            body,
            state,
            xs=None,
            length=8,
        )
    )(initial)
    expected = np.asarray((0, 1, 0, 1, 0, 1, 0, 1), dtype=np.int32)
    np.testing.assert_array_equal(traces.accepted, np.ones((8,), dtype=bool))
    np.testing.assert_array_equal(traces.focal_action, expected)
    np.testing.assert_array_equal(traces.expected_focal_action, expected)
    np.testing.assert_array_equal(traces.next_action, 1 - expected)
    np.testing.assert_array_equal(traces.expected_next_action, 1 - expected)
    np.testing.assert_array_equal(traces.action_policy_valid, np.ones((8,), dtype=bool))
    np.testing.assert_array_equal(traces.policy_replay_valid, np.ones((8,), dtype=bool))
    np.testing.assert_array_equal(
        traces.next_selection_diagnostics_valid,
        np.ones((8,), dtype=bool),
    )
    np.testing.assert_array_equal(
        traces.selection_binding_valid,
        np.ones((8,), dtype=bool),
    )
    np.testing.assert_array_equal(
        traces.focal_action_externally_forced,
        np.ones((8,), dtype=bool),
    )
    np.testing.assert_array_equal(
        traces.focal_action_ordinary_policy_action,
        np.where(
            np.asarray(traces.focal_action_explored),
            np.asarray(traces.focal_action_random_action),
            np.asarray(traces.focal_action_noisy_greedy_action),
        ),
    )
    assert np.bincount(np.asarray(traces.focal_action), minlength=2).tolist() == [4, 4]
    assert int(final.action) == 0
    assert int(final.step_count) == 8

    for before, after in zip(
        np.asarray(traces.focal_action_policy_rng_before),
        np.asarray(traces.focal_action_policy_rng_after),
        strict=True,
    ):
        replayed_after = jr.key_data(jr.split(jr.wrap_key_data(before), 4)[0])
        np.testing.assert_array_equal(after, replayed_after)


@pytest.mark.parametrize(
    "mutation",
    ("parity", "external_flag", "rng_replay"),
)
def test_balanced_policy_tampering_rejects_before_component_work(mutation: str) -> None:
    bridge = _balanced_bridge()
    state = bridge.initialize(jr.key(214), jr.key(215))
    selection = state.agent.current_selection
    control = state.agent.control
    if mutation == "parity":
        wrong = jnp.asarray(1, dtype=jnp.int32)
        corrupt_agent = state.agent.replace(
            control=control.replace(last_action=wrong),
            current_selection=selection.replace(action=wrong),
        )
        corrupt = state.replace(action=wrong, agent=corrupt_agent)
    elif mutation == "external_flag":
        corrupt = state.replace(
            agent=state.agent.replace(
                current_selection=selection.replace(
                    externally_forced=jnp.asarray(False, dtype=jnp.bool_)
                )
            )
        )
    else:
        hostile_key = jr.key(999)
        corrupt = state.replace(
            agent=state.agent.replace(
                control=control.replace(rng_key=hostile_key),
                current_selection=selection.replace(rng_key_after=hostile_key),
            )
        )

    with jax.disable_jit():
        rejected = bridge.step(corrupt)
    assert bool(rejected.trace.active)
    assert not bool(rejected.trace.entry_state_contract_valid)
    assert int(rejected.trace.proposed_world_step_delta) == 0
    assert int(rejected.trace.proposed_agent_step_delta) == 0
    assert int(rejected.trace.proposed_filter_step_delta) == 0
    if mutation == "parity":
        assert bool(rejected.trace.selection_binding_valid)
        assert bool(rejected.trace.policy_replay_valid)
        assert not bool(rejected.trace.action_policy_valid)
    elif mutation == "external_flag":
        assert not bool(rejected.trace.action_policy_valid)
    else:
        assert bool(rejected.trace.selection_binding_valid)
        assert not bool(rejected.trace.policy_replay_valid)
    _assert_atomic_rejection(corrupt, rejected)


@pytest.mark.parametrize(
    "mutation",
    (
        "bridge_negative",
        "bridge_saturated",
        "world_desynchronized",
        "agent_desynchronized",
    ),
)
def test_counter_contract_rejects_atomically_before_component_work(mutation: str) -> None:
    bridge = _shared_bridge()
    state = bridge.initialize(jr.key(12), jr.key(13))
    if mutation == "bridge_negative":
        corrupt = state.replace(step_count=jnp.asarray(-1, dtype=jnp.int32))
    elif mutation == "bridge_saturated":
        corrupt = state.replace(step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32))
    elif mutation == "world_desynchronized":
        corrupt = state.replace(
            world=state.world.replace(step_count=jnp.asarray(1, dtype=jnp.int32))
        )
    else:
        corrupt = state.replace(
            agent=state.agent.replace(step_count=jnp.asarray(1, dtype=jnp.int32))
        )

    with jax.disable_jit():
        rejected = bridge.step(corrupt)
    assert bool(rejected.trace.active)
    assert not bool(rejected.trace.entry_state_contract_valid)
    assert not bool(rejected.trace.counters_synchronized)
    assert int(rejected.trace.proposed_world_step_delta) == 0
    assert int(rejected.trace.proposed_agent_step_delta) == 0
    assert int(rejected.trace.proposed_filter_step_delta) == 0
    _assert_atomic_rejection(corrupt, rejected)


def test_invalid_action_rejects_before_component_work() -> None:
    bridge = _shared_bridge()
    state = bridge.initialize(jr.key(14), jr.key(15))
    corrupt = state.replace(action=jnp.asarray(2, dtype=jnp.int32))

    with jax.disable_jit():
        rejected = bridge.step(corrupt)
    assert bool(rejected.trace.active)
    assert not bool(rejected.trace.entry_state_contract_valid)
    assert not bool(rejected.trace.action_valid)
    _assert_atomic_rejection(corrupt, rejected)


@pytest.mark.parametrize("mutation", ("posterior_nan", "invalid_flag", "counter_desync"))
def test_filter_corruption_is_diagnostic_only_and_cannot_change_learning_path(
    mutation: str,
) -> None:
    bridge = _shared_bridge()
    state = bridge.initialize(jr.key(14), jr.key(15))
    if mutation == "posterior_nan":
        corrupt_filter = state.world_filter.replace(
            posterior_mean=jnp.asarray(jnp.nan, dtype=jnp.float32)
        )
    elif mutation == "invalid_flag":
        corrupt_filter = state.world_filter.replace(
            valid=jnp.asarray(False, dtype=jnp.bool_)
        )
    else:
        corrupt_filter = state.world_filter.replace(
            step_count=jnp.asarray(1, dtype=jnp.int32)
        )
    corrupt = state.replace(world_filter=corrupt_filter)

    with jax.disable_jit():
        ordinary_first = bridge.step(state)
        hostile_first = bridge.step(corrupt)
        ordinary_second = bridge.step(ordinary_first.state)
        hostile_second = bridge.step(hostile_first.state)

    for hostile in (hostile_first, hostile_second):
        assert bool(hostile.trace.active)
        assert bool(hostile.trace.accepted)
        assert bool(hostile.trace.entry_state_contract_valid)
        assert bool(hostile.trace.counters_synchronized)
        assert not bool(hostile.trace.filter_entry_valid)
        assert not bool(hostile.trace.all_finite)
        assert bool(hostile.state.valid)
        assert int(hostile.trace.committed_world_step_delta) == 1
        assert int(hostile.trace.committed_agent_step_delta) == 1
        assert int(hostile.trace.committed_bridge_step_delta) == 1
        np.testing.assert_array_equal(
            hostile.trace.agent_partner_belief_conditioned_reward_cells,
            np.full((2, 2), 0.5, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            hostile.trace.agent_partner_belief_conditioned_expected_rewards,
            np.full((2,), 0.5, dtype=np.float32),
        )

    for ordinary, hostile in (
        (ordinary_first, hostile_first),
        (ordinary_second, hostile_second),
    ):
        chex.assert_trees_all_equal(
            _unwrap_prng_keys(hostile.state.world),
            _unwrap_prng_keys(ordinary.state.world),
        )
        chex.assert_trees_all_equal(
            _unwrap_prng_keys(hostile.state.agent),
            _unwrap_prng_keys(ordinary.state.agent),
        )
        assert int(hostile.state.action) == int(ordinary.state.action)
        assert int(hostile.state.step_count) == int(ordinary.state.step_count)


def test_filter_failure_cannot_change_jitted_scan_actions_or_updates() -> None:
    bridge = _shared_bridge()
    initial = bridge.initialize(jr.key(140), jr.key(150))
    hostile_initial = initial.replace(
        world_filter=initial.world_filter.replace(
            valid=jnp.asarray(False, dtype=jnp.bool_)
        )
    )

    @jax.jit
    def run(state: HiddenPartnerWorldOnlineState):
        def body(carry, _):
            result = bridge.step(carry)
            return result.state, result.trace

        return jax.lax.scan(body, state, xs=None, length=3)

    ordinary_final, ordinary_traces = run(initial)
    hostile_final, hostile_traces = run(hostile_initial)

    np.testing.assert_array_equal(ordinary_traces.accepted, np.ones((3,), dtype=bool))
    np.testing.assert_array_equal(hostile_traces.accepted, np.ones((3,), dtype=bool))
    np.testing.assert_array_equal(hostile_traces.filter_entry_valid, np.zeros((3,), dtype=bool))
    np.testing.assert_array_equal(hostile_traces.all_finite, np.zeros((3,), dtype=bool))
    np.testing.assert_array_equal(
        hostile_traces.focal_action,
        ordinary_traces.focal_action,
    )
    np.testing.assert_array_equal(
        hostile_traces.next_action,
        ordinary_traces.next_action,
    )
    chex.assert_trees_all_equal(
        _unwrap_prng_keys(hostile_final.world),
        _unwrap_prng_keys(ordinary_final.world),
    )
    chex.assert_trees_all_equal(
        _unwrap_prng_keys(hostile_final.agent),
        _unwrap_prng_keys(ordinary_final.agent),
    )
    assert int(hostile_final.action) == int(ordinary_final.action)
    assert int(hostile_final.step_count) == int(ordinary_final.step_count) == 3


def test_valid_filter_posterior_perturbation_changes_only_evaluator_trace() -> None:
    bridge = _shared_bridge()
    initial = bridge.initialize(jr.key(141), jr.key(151))
    perturbed_initial = initial.replace(
        world_filter=initial.world_filter.replace(
            posterior_mean=jnp.asarray(0.25, dtype=jnp.float32)
        )
    )

    with jax.disable_jit():
        ordinary_first = bridge.step(initial)
        perturbed_first = bridge.step(perturbed_initial)
        ordinary_second = bridge.step(ordinary_first.state)
        perturbed_second = bridge.step(perturbed_first.state)

    for ordinary, perturbed in (
        (ordinary_first, perturbed_first),
        (ordinary_second, perturbed_second),
    ):
        assert bool(ordinary.trace.accepted)
        assert bool(perturbed.trace.accepted)
        assert bool(perturbed.trace.filter_entry_valid)
        assert bool(perturbed.trace.proposed_filter_update_valid)
        assert bool(perturbed.trace.proposed_filter_decision_valid)
        chex.assert_trees_all_equal(
            _unwrap_prng_keys(perturbed.state.world),
            _unwrap_prng_keys(ordinary.state.world),
        )
        chex.assert_trees_all_equal(
            _unwrap_prng_keys(perturbed.state.agent),
            _unwrap_prng_keys(ordinary.state.agent),
        )
        assert int(perturbed.state.action) == int(ordinary.state.action)
        assert int(perturbed.state.step_count) == int(ordinary.state.step_count)

    assert not np.array_equal(
        np.asarray(
            perturbed_first.trace.agent_partner_belief_conditioned_reward_cells
        ),
        np.asarray(ordinary_first.trace.agent_partner_belief_conditioned_reward_cells),
    )


def test_cross_bridge_config_token_rejects_and_latches_invalid() -> None:
    source = _shared_bridge()
    source_state = source.initialize(jr.key(16), jr.key(17))
    other = HiddenPartnerWorldOnlineBridge(
        world=HiddenPartnerWorldFeedbackWorld(
            HiddenPartnerWorldFeedbackConfig(world_flip_probability=0.02)
        )
    )
    other.initialize(jr.key(18), jr.key(19))

    with jax.disable_jit():
        rejected = other.step(source_state)
    assert not bool(rejected.trace.config_token_valid)
    assert not bool(rejected.trace.entry_state_contract_valid)
    _assert_atomic_rejection(source_state, rejected)


def test_static_component_tree_shape_and_dtype_contracts_fail_before_computation() -> None:
    bridge = _shared_bridge()
    state = bridge.initialize(jr.key(20), jr.key(21))
    wrong_dtype = state.replace(action=state.action.astype(jnp.float32))
    with pytest.raises((TypeError, ValueError), match="static state contract"):
        bridge.step(wrong_dtype)

    wrong_component_shape = state.replace(
        world=state.world.replace(current_cues=jnp.ones((3,), dtype=jnp.float32))
    )
    with pytest.raises((TypeError, ValueError), match="static state contract"):
        bridge.step(wrong_component_shape)


def test_filter_decision_is_pretransition_and_two_step_filter_state_is_continuous() -> None:
    bridge = _shared_bridge()
    initial = bridge.initialize(jr.key(22), jr.key(23))
    initial = initial.replace(
        world_filter=initial.world_filter.replace(
            posterior_mean=jnp.asarray(0.0, dtype=jnp.float32)
        )
    )
    pre_cells = bridge.world_filter.expected_reward_cells(
        initial.world_filter.posterior_mean
    )
    first = bridge.step(initial)
    post_cells = bridge.world_filter.expected_reward_cells(
        first.state.world_filter.posterior_mean
    )
    second = bridge.step(first.state)

    assert bool(first.trace.accepted)
    assert bool(second.trace.accepted)
    chex.assert_trees_all_equal(
        first.trace.agent_partner_belief_conditioned_reward_cells,
        pre_cells.rewards,
    )
    assert not np.allclose(
        np.asarray(first.trace.agent_partner_belief_conditioned_reward_cells),
        np.asarray(post_cells.rewards),
    )
    chex.assert_trees_all_equal(
        first.trace.filter_mean_post,
        second.trace.filter_mean_pre,
    )


def test_lax_scan_midstream_rejection_latches_remaining_steps() -> None:
    bridge = _shared_bridge()
    initial = bridge.initialize(jr.key(24), jr.key(25))

    def body(state, corrupt_action):
        attempted = jax.lax.cond(
            corrupt_action,
            lambda current: current.replace(action=jnp.asarray(2, dtype=jnp.int32)),
            lambda current: current,
            state,
        )
        result = bridge.step(attempted)
        return result.state, result.trace

    final, traces = jax.lax.scan(
        body,
        initial,
        jnp.asarray((False, True, False), dtype=jnp.bool_),
    )
    np.testing.assert_array_equal(traces.active, np.asarray((True, True, False)))
    np.testing.assert_array_equal(traces.accepted, np.asarray((True, False, False)))
    assert int(final.step_count) == 1
    assert not bool(final.valid)


class _OracleCorruptingWorld(HiddenPartnerWorldFeedbackWorld):
    def step(self, state, focal_action):
        transition, next_state = super().step(state, focal_action)
        corrupted = transition.replace(
            oracle=transition.oracle.replace(
                world_sign=jnp.asarray(jnp.nan, dtype=jnp.float32),
                counterfactual_rewards=jnp.full((2,), jnp.nan, dtype=jnp.float32),
            )
        )
        return corrupted, next_state


def test_whole_bridge_oracle_corruption_cannot_change_committed_learning_path() -> None:
    world_key = jr.key(26)
    agent_key = jr.key(27)
    ordinary = _shared_bridge()
    hostile = HiddenPartnerWorldOnlineBridge(world=_OracleCorruptingWorld())
    ordinary_state = ordinary.initialize(world_key, agent_key)
    hostile_state = hostile.initialize(world_key, agent_key)

    ordinary_step = ordinary.step(ordinary_state)
    hostile_step = hostile.step(hostile_state)

    assert bool(ordinary_step.trace.accepted)
    assert bool(hostile_step.trace.accepted)
    assert bool(ordinary_step.trace.oracle_trace_valid)
    assert not bool(hostile_step.trace.oracle_trace_valid)
    assert not bool(hostile_step.trace.all_finite)
    assert float(hostile_step.trace.oracle_world_sign) == 0.0
    assert int(hostile_step.trace.oracle_regime_id) == -1
    chex.assert_trees_all_equal(
        _unwrap_prng_keys(hostile_step.state.world),
        _unwrap_prng_keys(ordinary_step.state.world),
    )
    chex.assert_trees_all_equal(
        _unwrap_prng_keys(hostile_step.state.agent),
        _unwrap_prng_keys(ordinary_step.state.agent),
    )
    chex.assert_trees_all_close(
        hostile_step.state.world_filter,
        ordinary_step.state.world_filter,
        atol=1e-6,
        rtol=0.0,
    )
    assert int(hostile_step.state.action) == int(ordinary_step.state.action)
    assert int(hostile_step.state.step_count) == int(ordinary_step.state.step_count)


def test_default_mechanism_trace_projects_lifecycle_router_credit_and_counters() -> None:
    bridge = _shared_bridge()
    state = bridge.initialize(jr.key(300), jr.key(301))
    budget_before = bridge.resource_budget(state)
    expected = _agent_result_for_state(bridge, state)
    step = bridge.step(state)
    mechanism = step.trace.mechanism
    diagnostics = expected.diagnostics

    assert bool(step.trace.accepted)
    assert bool(mechanism.valid)
    _assert_grounded_mechanism_neutral(mechanism)
    chex.assert_trees_all_equal(
        mechanism.lifecycle_pre_descriptors,
        state.agent.router.descriptors,
    )
    chex.assert_trees_all_equal(
        mechanism.lifecycle_proposal_descriptors,
        diagnostics.interaction_proposal_descriptors,
    )
    chex.assert_trees_all_equal(
        mechanism.lifecycle_applied_descriptors,
        diagnostics.interaction_applied_descriptors,
    )
    for projected, source in (
        (mechanism.lifecycle_active_evidence_refreshed, diagnostics.interaction_evidence_refreshed),
        (
            mechanism.lifecycle_retention_evidence_refreshed,
            diagnostics.interaction_retention_evidence_refreshed,
        ),
        (mechanism.lifecycle_durable_read_mask, diagnostics.interaction_durable_read_mask),
        (
            mechanism.lifecycle_relevance_probe_scores,
            diagnostics.interaction_relevance_probe_scores,
        ),
        (
            mechanism.lifecycle_relevance_probe_errors,
            diagnostics.interaction_relevance_probe_errors,
        ),
        (
            mechanism.lifecycle_candidate_promotion_signal,
            diagnostics.interaction_candidate_promotion_signal,
        ),
        (
            mechanism.lifecycle_candidate_promotion_raw_evidence,
            diagnostics.interaction_candidate_promotion_raw_evidence,
        ),
        (
            mechanism.lifecycle_candidate_promotion_confirmed,
            diagnostics.interaction_candidate_promotion_confirmed,
        ),
        (
            mechanism.lifecycle_candidate_promotion_evidence_streak_pre,
            diagnostics.interaction_candidate_promotion_evidence_streak_pre,
        ),
        (
            mechanism.lifecycle_candidate_promotion_evidence_streak_updated,
            diagnostics.interaction_candidate_promotion_evidence_streak_updated,
        ),
        (
            mechanism.lifecycle_candidate_promotion_evidence_streak_proposal_post,
            diagnostics.interaction_candidate_promotion_evidence_streak_proposal_post,
        ),
        (
            mechanism.lifecycle_candidate_promotion_evidence_streak_post,
            diagnostics.interaction_candidate_promotion_evidence_streak_post,
        ),
        (
            mechanism.lifecycle_candidate_reacquisition_required_pre,
            diagnostics.interaction_candidate_reacquisition_required_pre,
        ),
        (
            mechanism.lifecycle_candidate_reacquisition_required_proposal_post,
            diagnostics.interaction_candidate_reacquisition_required_proposal_post,
        ),
        (
            mechanism.lifecycle_candidate_reacquisition_required_post,
            diagnostics.interaction_candidate_reacquisition_required_post,
        ),
        (
            mechanism.lifecycle_candidate_reacquisition_confirmed,
            diagnostics.interaction_candidate_reacquisition_confirmed,
        ),
        (
            mechanism.lifecycle_candidate_reset_mask,
            diagnostics.interaction_matching_candidate_reset_mask,
        ),
        (
            mechanism.lifecycle_applied_candidate_reset_mask,
            diagnostics.interaction_applied_matching_candidate_reset_mask,
        ),
        (mechanism.consumer_read_acquire_pre, diagnostics.consumer_read_acquire_pre),
        (mechanism.consumer_read_acquire_post, diagnostics.consumer_read_acquire_post),
        (mechanism.consumer_confirmed_write_pre, diagnostics.consumer_confirmed_write_pre),
        (mechanism.consumer_confirmed_write_post, diagnostics.consumer_confirmed_write_post),
        (mechanism.consumer_read_mask_pre, diagnostics.consumer_read_mask_pre),
        (mechanism.consumer_read_mask_post, diagnostics.consumer_read_mask_post),
    ):
        chex.assert_trees_all_equal(projected, source)
    route = diagnostics.route
    chex.assert_trees_all_equal(mechanism.router_source_slots, route.source_slots)
    chex.assert_trees_all_equal(mechanism.router_survivor_mask, route.survivor_mask)
    chex.assert_trees_all_equal(mechanism.router_new_mask, route.new_mask)
    chex.assert_trees_all_equal(mechanism.router_evicted_mask, route.evicted_mask)
    for field in (
        "consumer_route_source_slots_exact",
        "consumer_route_identity_masks_exact",
        "consumer_route_stable_prefix_exact",
        "consumer_route_survivor_values_exact",
        "consumer_route_reset_values_exact",
        "consumer_route_no_carry_reset_exact",
        "consumer_route_behavior_values_exact",
        "consumer_route_q_values_exact",
        "consumer_route_trace_values_exact",
        "consumer_route_last_observation_exact",
        "consumer_route_grounded_values_exact",
        "consumer_route_values_exact",
        "consumer_lifecycle_destination_reset_exact",
    ):
        projected = getattr(mechanism, field)
        source = getattr(diagnostics, field)
        chex.assert_trees_all_equal(projected, source)
        assert bool(projected)
    chex.assert_trees_all_equal(
        mechanism.behavior_credit_gradient_chi,
        diagnostics.behavior_gradient_chi,
    )
    chex.assert_trees_all_equal(
        mechanism.behavior_credit_gradient_phi,
        diagnostics.behavior_gradient_phi,
    )
    assert bool(mechanism.behavior_prediction_matches_decision) is bool(
        diagnostics.behavior_prediction_matches_decision
    )
    assert bool(mechanism.behavior_credit_valid)
    assert not bool(mechanism.mixed_credit_valid)
    assert bool(mechanism.state_learning_valid)
    assert bool(mechanism.state_learning_committed)
    assert int(mechanism.state_builder_step_delta) == 1
    assert int(mechanism.state_builder_learning_delta) == 1
    assert int(mechanism.behavior_step_delta) == 1
    assert int(mechanism.interaction_step_delta) == 1
    assert int(mechanism.table_world_step_delta) == 1
    assert int(mechanism.grounded_world_step_delta) == 0
    assert int(mechanism.control_step_delta) == 1
    assert int(mechanism.router_route_delta) == 1
    assert int(mechanism.integrated_step_delta) == 1
    budget_after = bridge.resource_budget(step.state)
    assert budget_after.total_state_nbytes == budget_before.total_state_nbytes
    assert _tree_signature(step.state) == _tree_signature(state)


def test_grounded_mechanism_trace_binds_affine_heads_and_isolated_row_update() -> None:
    bridge = _grounded_bridge()
    state = bridge.initialize(jr.key(302), jr.key(303))
    expected = _agent_result_for_state(bridge, state)
    step = bridge.step(state)
    mechanism = step.trace.mechanism
    grounded = expected.diagnostics.grounded_world_update
    cached_grounded = state.agent.current_evaluation.grounded_world

    assert bool(step.trace.accepted)
    assert grounded is not None
    assert cached_grounded is not None
    assert bool(mechanism.valid)
    assert bool(mechanism.grounded_enabled)
    joint_index = 2 * int(step.trace.focal_action) + int(step.trace.partner_action)
    assert int(mechanism.grounded_executed_joint_index) == joint_index
    for projected, source in (
        (mechanism.grounded_feature_contribution, grounded.prediction.feature_contribution),
        (mechanism.grounded_row_bias, grounded.prediction.row_bias),
        (mechanism.grounded_raw_predictions, grounded.prediction.raw_predictions),
        (mechanism.grounded_targets, grounded.targets),
        (mechanism.grounded_errors, grounded.errors),
        (mechanism.grounded_fit_loss_by_head, grounded.fit_loss_by_head),
        (
            mechanism.grounded_representation_loss_by_head,
            grounded.representation_loss_by_head,
        ),
        (mechanism.grounded_representation_gradient, grounded.representation_gradient),
        (
            mechanism.grounded_representation_gradient_by_head,
            grounded.representation_gradient_by_head,
        ),
        (
            mechanism.grounded_representation_gradient_norm_by_head,
            grounded.representation_gradient_norm_by_head,
        ),
        (
            mechanism.grounded_proposed_weight_row_bit_change_mask,
            grounded.proposed_weight_row_bit_change_mask,
        ),
        (
            mechanism.grounded_proposed_bias_row_bit_change_mask,
            grounded.proposed_bias_row_bit_change_mask,
        ),
        (
            mechanism.grounded_executed_weight_row_delta_norm_by_head,
            grounded.executed_weight_row_delta_norm_by_head,
        ),
        (
            mechanism.grounded_executed_bias_row_delta_by_head,
            grounded.executed_bias_row_delta_by_head,
        ),
    ):
        chex.assert_trees_all_equal(projected, source)
    chex.assert_trees_all_equal(
        mechanism.grounded_raw_predictions,
        mechanism.grounded_feature_contribution + mechanism.grounded_row_bias,
    )
    chex.assert_trees_all_equal(
        mechanism.grounded_raw_predictions,
        cached_grounded.grounded_raw_predictions[joint_index],
    )
    outside = np.ones((4,), dtype=bool)
    outside[joint_index] = False
    assert not bool(
        jnp.any(mechanism.grounded_proposed_weight_row_bit_change_mask[outside])
    )
    assert not bool(
        jnp.any(mechanism.grounded_proposed_bias_row_bit_change_mask[outside])
    )
    assert bool(mechanism.grounded_prediction_valid)
    assert bool(mechanism.grounded_target_valid)
    assert bool(mechanism.grounded_gradient_valid)
    assert bool(mechanism.grounded_row_update_isolated)
    assert bool(mechanism.grounded_update_applied)
    assert bool(mechanism.grounded_credit_valid)
    assert bool(mechanism.mixed_credit_valid)


def test_mechanism_trace_has_one_static_contract_across_default_grounded_and_scan() -> None:
    ordinary = _shared_bridge()
    grounded = _grounded_bridge()
    ordinary_state = ordinary.initialize(jr.key(304), jr.key(305))
    grounded_state = grounded.initialize(jr.key(306), jr.key(307))
    ordinary_trace = ordinary.step(ordinary_state).trace.mechanism
    grounded_trace = grounded.step(grounded_state).trace.mechanism

    assert _tree_signature(ordinary_trace) == _tree_signature(grounded_trace)
    expected_contract = {
        "grounded_raw_predictions": ((10,), "float32"),
        "grounded_representation_gradient": ((24,), "float32"),
        "grounded_representation_gradient_by_head": ((10, 24), "float32"),
        "lifecycle_pre_descriptors": ((12, 2), "int32"),
        "lifecycle_relevance_probe_scores": ((12,), "float32"),
        "lifecycle_relevance_probe_errors": ((1, 12), "float32"),
        "lifecycle_candidate_promotion_signal": ((66,), "float32"),
        "lifecycle_candidate_promotion_raw_evidence": ((66,), "bool"),
        "lifecycle_candidate_promotion_evidence_streak_pre": ((66,), "int32"),
        "lifecycle_candidate_promotion_evidence_streak_updated": ((66,), "int32"),
        "lifecycle_candidate_promotion_evidence_streak_proposal_post": (
            (66,),
            "int32",
        ),
        "lifecycle_candidate_promotion_evidence_streak_post": ((66,), "int32"),
        "random_curation_active_priorities": ((12,), "float32"),
        "random_curation_candidate_priorities": ((66,), "float32"),
        "consumer_read_mask_post": ((12,), "bool"),
        "router_source_slots": ((12,), "int32"),
        "behavior_credit_gradient_phi": ((12,), "float32"),
        "mixed_credit_gradient_chi": ((24,), "float32"),
        "integrated_step_delta": ((), "int32"),
    }
    for name, (shape, dtype) in expected_contract.items():
        value = getattr(grounded_trace, name)
        assert value.shape == shape
        assert str(value.dtype) == dtype

    def body(state: HiddenPartnerWorldOnlineState, _: object):
        result = grounded.step(state)
        return result.state, result.trace.mechanism

    final, traces = jax.jit(
        lambda state: jax.lax.scan(body, state, xs=None, length=2)
    )(grounded_state)
    assert int(final.step_count) == 2
    assert traces.grounded_raw_predictions.shape == (2, 10)
    assert traces.lifecycle_pre_descriptors.shape == (2, 12, 2)
    assert traces.grounded_representation_gradient_by_head.shape == (2, 10, 24)
    np.testing.assert_array_equal(traces.valid, np.ones((2,), dtype=bool))


@pytest.mark.parametrize("grounded", [False, True])
def test_rejected_and_blocked_mechanism_traces_use_exact_neutral_sentinels(
    grounded: bool,
) -> None:
    bridge = _grounded_bridge() if grounded else _shared_bridge()
    state = bridge.initialize(jr.key(308), jr.key(309))
    corrupt = state.replace(
        agent=state.agent.replace(
            behavior=state.agent.behavior.replace(
                weights=state.agent.behavior.weights.at[0, 0].add(0.25)
            )
        )
    )

    rejected = bridge.step(corrupt)
    blocked = bridge.step(rejected.state)

    assert bool(rejected.trace.active)
    assert not bool(blocked.trace.active)
    _assert_mechanism_fully_neutral(rejected.trace.mechanism)
    _assert_mechanism_fully_neutral(blocked.trace.mechanism)


@pytest.mark.parametrize("pair", ((0, 2), (4, 5)), ids=("C", "D"))
def test_mechanism_trace_projects_native_lifecycle_transaction_and_router_identity(
    pair: tuple[int, int],
) -> None:
    bridge = _shared_bridge()
    initial = bridge.initialize(jr.key(310), jr.key(311))
    prepared = _force_next_interaction_promotion(bridge, initial, pair)
    expected = _agent_result_for_state(bridge, prepared)
    step = bridge.step(prepared)
    mechanism = step.trace.mechanism
    diagnostics = expected.diagnostics

    assert bool(step.trace.accepted)
    assert bool(mechanism.lifecycle_proposed)
    assert bool(mechanism.lifecycle_applied)
    assert bool(mechanism.router_descriptors_changed)
    assert int(mechanism.lifecycle_proposal_replaced_slot) == 0
    assert int(mechanism.lifecycle_applied_replaced_slot) == 0
    assert int(mechanism.lifecycle_proposal_promoted_candidate) >= 0
    assert (
        int(mechanism.lifecycle_applied_promoted_candidate)
        == int(mechanism.lifecycle_proposal_promoted_candidate)
    )
    chex.assert_trees_all_equal(
        mechanism.lifecycle_proposal_descriptors,
        diagnostics.interaction_proposal_descriptors,
    )
    chex.assert_trees_all_equal(
        mechanism.lifecycle_applied_descriptors,
        step.state.agent.router.descriptors,
    )
    chex.assert_trees_all_equal(
        mechanism.router_source_slots,
        diagnostics.route.source_slots,
    )
    chex.assert_trees_all_equal(
        mechanism.router_survivor_mask,
        diagnostics.route.survivor_mask,
    )
    chex.assert_trees_all_equal(mechanism.router_new_mask, diagnostics.route.new_mask)
    chex.assert_trees_all_equal(
        mechanism.router_evicted_mask,
        diagnostics.route.evicted_mask,
    )
    assert int(jnp.sum(mechanism.router_new_mask)) == 1
    assert int(jnp.sum(mechanism.router_evicted_mask)) == 1
    assert int(mechanism.router_generation_count_after) == 1
    assert int(mechanism.router_generation_delta) == 1


def test_oracle_schedule_and_noise_projection_is_exactly_bound_to_world_transition() -> None:
    bridge = _shared_bridge()
    state = bridge.initialize(jr.key(312), jr.key(313))
    transition, _ = bridge.world.step(state.world, state.action)
    oracle = transition.oracle
    step = bridge.step(state)

    assert bool(step.trace.accepted)
    assert bool(step.trace.oracle_trace_valid)
    for projected, source in (
        (step.trace.oracle_step_count, oracle.step_count),
        (step.trace.oracle_cycle_index, oracle.cycle_index),
        (step.trace.oracle_cycle_step, oracle.cycle_step),
        (step.trace.oracle_cycle_length, oracle.cycle_length),
        (step.trace.oracle_segment_index, oracle.segment_index),
        (step.trace.oracle_segment_step, oracle.segment_step),
        (step.trace.oracle_segment_length, oracle.segment_length),
        (step.trace.oracle_regime_id, oracle.regime_id),
        (step.trace.oracle_next_cycle_index, oracle.next_cycle_index),
        (step.trace.oracle_next_segment_index, oracle.next_segment_index),
        (step.trace.oracle_next_regime_id, oracle.next_regime_id),
        (step.trace.oracle_schedule_switched, oracle.schedule_switched),
        (step.trace.oracle_partner_intended_action, oracle.partner_intended_action),
        (step.trace.oracle_partner_intended_sign, oracle.partner_intended_sign),
        (step.trace.oracle_partner_flipped, oracle.partner_flipped),
        (step.trace.oracle_focal_action_sign, oracle.focal_action_sign),
        (step.trace.oracle_partner_action_sign, oracle.partner_action_sign),
        (step.trace.oracle_world_sign, oracle.world_sign),
        (step.trace.oracle_next_world_sign, oracle.next_world_sign),
        (step.trace.oracle_world_cue_flipped, oracle.world_cue_flipped),
        (step.trace.oracle_next_world_cue_flipped, oracle.next_world_cue_flipped),
        (
            step.trace.oracle_full_information_action,
            oracle.full_information_optimal_focal_action,
        ),
        (
            step.trace.oracle_full_information_action_margin,
            oracle.full_information_action_margin,
        ),
        (
            step.trace.oracle_full_information_action_tied,
            oracle.full_information_action_tied,
        ),
        (
            step.trace.oracle_realized_counterfactual_rewards,
            oracle.counterfactual_rewards,
        ),
    ):
        chex.assert_trees_all_equal(projected, source)
    chex.assert_trees_all_equal(
        step.trace.oracle_world_cue_flipped,
        state.world.current_cues != state.world.world_sign,
    )
    chex.assert_trees_all_equal(
        step.trace.oracle_next_world_cue_flipped,
        step.state.world.current_cues != step.state.world.world_sign,
    )


@pytest.mark.parametrize(
    "mutation",
    ("noisy_greedy", "random_action", "explored", "rng_before"),
)
def test_balanced_current_policy_primitive_tamper_rejects_before_work(
    mutation: str,
) -> None:
    bridge = _balanced_bridge()
    state = bridge.initialize(jr.key(314), jr.key(315))
    selection = state.agent.current_selection
    if mutation == "noisy_greedy":
        corrupt_selection = selection.replace(
            noisy_greedy_action=1 - selection.noisy_greedy_action
        )
    elif mutation == "random_action":
        corrupt_selection = selection.replace(random_action=1 - selection.random_action)
    elif mutation == "explored":
        corrupt_selection = selection.replace(explored=~selection.explored)
    else:
        corrupt_selection = selection.replace(rng_key_before=jr.key(999))
    corrupt = state.replace(
        agent=state.agent.replace(current_selection=corrupt_selection)
    )

    rejected = bridge.step(corrupt)

    assert bool(rejected.trace.selection_binding_valid)
    assert bool(rejected.trace.action_policy_valid)
    assert not bool(rejected.trace.policy_replay_valid)
    assert int(rejected.trace.proposed_world_step_delta) == 0
    assert int(rejected.trace.proposed_agent_step_delta) == 0
    _assert_atomic_rejection(corrupt, rejected)
    _assert_mechanism_fully_neutral(rejected.trace.mechanism)


class _NextSelectionTamperingAgent(IntegratedHiddenPartnerAgent):
    def __init__(self, mutation: str):
        super().__init__(
            IntegratedHiddenPartnerConfig(action_selection_mode="externally_forced")
        )
        self._mutation = mutation

    def update_with_forced_next_action(
        self,
        state,
        transition,
        next_action,
    ) -> IntegratedUpdateResult:
        result = super().update_with_forced_next_action(
            state,
            transition,
            next_action,
        )
        selection = result.state.current_selection
        if self._mutation == "binding":
            corrupt_selection = selection.replace(action=1 - selection.action)
            corrupt_control = result.state.control
        elif self._mutation == "replay":
            corrupt_selection = selection.replace(
                noisy_greedy_action=1 - selection.noisy_greedy_action
            )
            corrupt_control = result.state.control
        else:
            hostile_before = jr.key(999)
            replayed = self.select_planner_action(
                result.state.control.replace(rng_key=hostile_before),
                result.state.current_evaluation.planner_scores,
            )
            corrupt_selection = replayed.replace(
                action=next_action,
                externally_forced=jnp.asarray(True, dtype=jnp.bool_),
            )
            corrupt_control = result.state.control.replace(
                last_action=next_action,
                rng_key=corrupt_selection.rng_key_after,
            )
        return result.replace(
            state=result.state.replace(
                control=corrupt_control,
                current_selection=corrupt_selection,
            ),
            diagnostics=result.diagnostics.replace(next_selection=corrupt_selection),
        )


@pytest.mark.parametrize("mutation", ("binding", "replay", "continuity"))
def test_next_selection_binding_and_replay_are_checked_before_commit(
    mutation: str,
) -> None:
    bridge = HiddenPartnerWorldOnlineBridge(
        agent=_NextSelectionTamperingAgent(mutation),
        focal_action_policy="balanced_external",
    )
    state = bridge.initialize(jr.key(316), jr.key(317))

    rejected = bridge.step(state)

    assert int(rejected.trace.proposed_world_step_delta) == 1
    assert int(rejected.trace.proposed_agent_step_delta) == 1
    assert bool(rejected.trace.next_action_policy_valid) is (mutation != "binding")
    assert bool(rejected.trace.next_selection_binding_valid) is (mutation == "replay")
    assert bool(rejected.trace.next_policy_replay_valid) is (mutation != "replay")
    assert bool(rejected.trace.next_selection_diagnostics_valid)
    _assert_atomic_rejection(state, rejected)
    _assert_mechanism_fully_neutral(rejected.trace.mechanism)


class _NextSelectionDiagnosticsTamperingAgent(IntegratedHiddenPartnerAgent):
    def __init__(self) -> None:
        super().__init__(
            IntegratedHiddenPartnerConfig(action_selection_mode="externally_forced")
        )

    def update_with_forced_next_action(
        self,
        state,
        transition,
        next_action,
    ) -> IntegratedUpdateResult:
        result = super().update_with_forced_next_action(
            state,
            transition,
            next_action,
        )
        diagnostic_selection = result.diagnostics.next_selection.replace(
            random_action=1 - result.diagnostics.next_selection.random_action
        )
        return result.replace(
            diagnostics=result.diagnostics.replace(
                next_selection=diagnostic_selection
            )
        )


def test_diagnostics_only_next_selection_tamper_is_noncausal_and_trace_invalid() -> None:
    world_key = jr.key(318)
    agent_key = jr.key(319)
    ordinary = _balanced_bridge()
    hostile = HiddenPartnerWorldOnlineBridge(
        agent=_NextSelectionDiagnosticsTamperingAgent(),
        focal_action_policy="balanced_external",
    )

    ordinary_step = ordinary.step(ordinary.initialize(world_key, agent_key))
    hostile_step = hostile.step(hostile.initialize(world_key, agent_key))

    assert bool(hostile_step.trace.accepted)
    assert not bool(hostile_step.trace.next_selection_diagnostics_valid)
    assert not bool(hostile_step.trace.learner_trace_valid)
    assert not bool(hostile_step.trace.all_finite)
    assert int(hostile_step.trace.next_action) == int(hostile_step.state.action)
    chex.assert_trees_all_equal(
        hostile_step.trace.next_action_random_action,
        hostile_step.state.agent.current_selection.random_action,
    )
    chex.assert_trees_all_equal(
        hostile_step.trace.next_action_policy_rng_before,
        jr.key_data(hostile_step.state.agent.current_selection.rng_key_before),
    )
    chex.assert_trees_all_equal(
        _unwrap_prng_keys(hostile_step.state.world),
        _unwrap_prng_keys(ordinary_step.state.world),
    )
    chex.assert_trees_all_equal(
        _unwrap_prng_keys(hostile_step.state.agent),
        _unwrap_prng_keys(ordinary_step.state.agent),
    )


def test_cross_policy_config_tokens_reject_same_shape_states_in_both_directions() -> None:
    ordinary = _shared_bridge()
    balanced = _balanced_bridge()
    ordinary_state = ordinary.initialize(jr.key(318), jr.key(319))
    balanced_state = balanced.initialize(jr.key(318), jr.key(319))
    assert _tree_signature(ordinary_state) == _tree_signature(balanced_state)

    ordinary_rejects_balanced = ordinary.step(balanced_state)
    balanced_rejects_ordinary = balanced.step(ordinary_state)

    for before, rejected in (
        (balanced_state, ordinary_rejects_balanced),
        (ordinary_state, balanced_rejects_ordinary),
    ):
        assert not bool(rejected.trace.config_token_valid)
        assert not bool(rejected.trace.entry_state_contract_valid)
        _assert_atomic_rejection(before, rejected)
        _assert_mechanism_fully_neutral(rejected.trace.mechanism)


class _OracleSemanticCorruptingWorld(HiddenPartnerWorldFeedbackWorld):
    def __init__(self, mutation: str):
        super().__init__()
        self._mutation = mutation

    def step(self, state, focal_action):
        transition, next_state = super().step(state, focal_action)
        oracle = transition.oracle
        if self._mutation == "schedule":
            oracle = oracle.replace(segment_step=oracle.segment_step + 1)
        elif self._mutation == "next_schedule":
            oracle = oracle.replace(next_segment_index=oracle.next_segment_index + 1)
        elif self._mutation == "intended":
            oracle = oracle.replace(
                partner_intended_action=1 - oracle.partner_intended_action
            )
        elif self._mutation == "cue_bits":
            oracle = oracle.replace(world_cue_flipped=~oracle.world_cue_flipped)
        elif self._mutation == "counterfactual":
            oracle = oracle.replace(
                counterfactual_rewards=1.0 - oracle.counterfactual_rewards
            )
        elif self._mutation == "optimal_action":
            oracle = oracle.replace(
                full_information_optimal_focal_action=(
                    1 - oracle.full_information_optimal_focal_action
                )
            )
        elif self._mutation == "action_margin":
            oracle = oracle.replace(
                full_information_action_margin=(
                    oracle.full_information_action_margin / 2.0
                )
            )
        elif self._mutation == "action_tied":
            oracle = oracle.replace(
                full_information_action_tied=~oracle.full_information_action_tied
            )
        else:
            oracle = oracle.replace(next_world_sign=-oracle.next_world_sign)
        return transition.replace(oracle=oracle), next_state


@pytest.mark.parametrize(
    "mutation",
    (
        "schedule",
        "next_schedule",
        "intended",
        "cue_bits",
        "counterfactual",
        "optimal_action",
        "action_margin",
        "action_tied",
        "world_relation",
    ),
)
def test_oracle_provenance_corruption_is_neutral_and_evaluator_only(
    mutation: str,
) -> None:
    world_key = jr.key(320)
    agent_key = jr.key(321)
    ordinary = _shared_bridge()
    hostile = HiddenPartnerWorldOnlineBridge(
        world=_OracleSemanticCorruptingWorld(mutation)
    )
    ordinary_state = ordinary.initialize(world_key, agent_key)
    hostile_state = hostile.initialize(world_key, agent_key)

    ordinary_step = ordinary.step(ordinary_state)
    hostile_step = hostile.step(hostile_state)

    assert bool(hostile_step.trace.accepted)
    assert not bool(hostile_step.trace.oracle_trace_valid)
    assert not bool(hostile_step.trace.all_finite)
    _assert_oracle_schedule_neutral(hostile_step.trace)
    chex.assert_trees_all_equal(
        _unwrap_prng_keys(hostile_step.state.world),
        _unwrap_prng_keys(ordinary_step.state.world),
    )
    chex.assert_trees_all_equal(
        _unwrap_prng_keys(hostile_step.state.agent),
        _unwrap_prng_keys(ordinary_step.state.agent),
    )
    chex.assert_trees_all_equal(
        hostile_step.trace.mechanism,
        ordinary_step.trace.mechanism,
    )


def test_balanced_filter_and_oracle_failures_cannot_change_actions_or_mechanisms() -> None:
    world_key = jr.key(322)
    agent_key = jr.key(323)
    ordinary = _balanced_bridge(initial_action=1)
    hostile_agent = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(action_selection_mode="externally_forced")
    )
    hostile = HiddenPartnerWorldOnlineBridge(
        world=_OracleCorruptingWorld(),
        agent=hostile_agent,
        focal_action_policy="balanced_external",
        initial_external_action=1,
    )
    ordinary_state = ordinary.initialize(world_key, agent_key)
    hostile_initial = hostile.initialize(world_key, agent_key)
    hostile_state = hostile_initial.replace(
        world_filter=hostile_initial.world_filter.replace(
            valid=jnp.asarray(False, dtype=jnp.bool_)
        )
    )

    ordinary_step = ordinary.step(ordinary_state)
    hostile_step = hostile.step(hostile_state)

    assert bool(hostile_step.trace.accepted)
    assert not bool(hostile_step.trace.filter_trace_valid)
    assert not bool(hostile_step.trace.oracle_trace_valid)
    assert not bool(hostile_step.trace.all_finite)
    assert int(hostile_step.trace.focal_action) == int(ordinary_step.trace.focal_action) == 1
    assert int(hostile_step.trace.next_action) == int(ordinary_step.trace.next_action) == 0
    chex.assert_trees_all_equal(
        _unwrap_prng_keys(hostile_step.state.world),
        _unwrap_prng_keys(ordinary_step.state.world),
    )
    chex.assert_trees_all_equal(
        _unwrap_prng_keys(hostile_step.state.agent),
        _unwrap_prng_keys(ordinary_step.state.agent),
    )
    chex.assert_trees_all_equal(
        hostile_step.trace.mechanism,
        ordinary_step.trace.mechanism,
    )


def test_random_curation_trace_proves_selection_only_matched_compute() -> None:
    common = IntegratedHiddenPartnerConfig(
        feature_lifecycle_enabled=False,
        replacement_interval=64,
    )
    random_bridge = HiddenPartnerWorldOnlineBridge(
        agent=IntegratedHiddenPartnerAgent(
            dataclasses.replace(common, random_feature_curation=True)
        )
    )
    learned_bridge = HiddenPartnerWorldOnlineBridge(
        agent=IntegratedHiddenPartnerAgent(
            dataclasses.replace(common, random_feature_curation=False)
        )
    )
    world_key = jr.key(324)
    agent_key = jr.key(325)
    random_state = _force_next_interaction_promotion(
        random_bridge,
        random_bridge.initialize(world_key, agent_key)
    )
    learned_state = _force_next_interaction_promotion(
        learned_bridge,
        learned_bridge.initialize(world_key, agent_key)
    )
    random_expected = _agent_result_for_state(random_bridge, random_state)
    random_step = random_bridge.step(random_state)
    learned_step = learned_bridge.step(learned_state)
    mechanism = random_step.trace.mechanism

    assert bool(random_step.trace.accepted)
    assert bool(learned_step.trace.accepted)
    assert bool(mechanism.random_curation_enabled)
    assert bool(mechanism.random_curation_attempted)
    assert bool(mechanism.random_curation_applied)
    assert not bool(mechanism.lifecycle_applied)
    chex.assert_shape(mechanism.random_curation_active_priorities, (12,))
    chex.assert_shape(mechanism.random_curation_candidate_priorities, (66,))
    assert mechanism.random_curation_active_priorities.dtype == jnp.float32
    assert mechanism.random_curation_candidate_priorities.dtype == jnp.float32
    for projected, source in (
        (
            mechanism.random_curation_active_priorities,
            random_expected.diagnostics.random_active_priorities,
        ),
        (
            mechanism.random_curation_candidate_priorities,
            random_expected.diagnostics.random_candidate_priorities,
        ),
        (
            mechanism.random_curation_selected_active_worst_slot,
            random_expected.diagnostics.curation_selected_active_worst_slot,
        ),
        (
            mechanism.random_curation_selected_promotion_candidate,
            random_expected.diagnostics.curation_selected_promotion_candidate,
        ),
        (
            mechanism.random_curation_selected_refresh_candidate,
            random_expected.diagnostics.curation_selected_refresh_candidate,
        ),
    ):
        chex.assert_trees_all_equal(projected, source)
    chex.assert_trees_all_equal(
        _unwrap_prng_keys(random_step.state.agent),
        _unwrap_prng_keys(learned_step.state.agent),
    )
    chex.assert_trees_all_equal(
        random_step.trace.mechanism.behavior_credit_gradient_chi,
        learned_step.trace.mechanism.behavior_credit_gradient_chi,
    )
    for field in (
        "lifecycle_relevance_probe_scores",
        "lifecycle_relevance_probe_errors",
        "lifecycle_candidate_promotion_signal",
        "lifecycle_candidate_promotion_evidence_streak_pre",
        "lifecycle_candidate_promotion_evidence_streak_updated",
        "lifecycle_candidate_promotion_evidence_streak_proposal_post",
        "lifecycle_candidate_promotion_evidence_streak_post",
    ):
        chex.assert_trees_all_equal(
            getattr(random_step.trace.mechanism, field),
            getattr(learned_step.trace.mechanism, field),
        )
