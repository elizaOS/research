# mypy: disable-error-code="attr-defined,call-arg,no-any-return,arg-type,type-var,union-attr"
"""Red-first B/M/P contract for the versioned live-memory action stack."""

from __future__ import annotations

import copy
import dataclasses

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from test_external_learned_state_live_memory_adapter import (
    _adapter as _v1_adapter,
)
from test_external_learned_state_live_memory_adapter import (
    _event_input,
    _transition,
    _tree_exact,
)

from alberta_framework.core.external_learned_state_live_memory_action_stack_adapter import (
    EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ACTION_STACK_CONFIG_SCHEMA,
    ExternalLearnedStateLiveMemoryActionStackAdapter,
    ExternalLearnedStateLiveMemoryActionStackConfig,
    ExternalLearnedStateLiveMemoryActionStackFeedback,
    ExternalLearnedStateLiveMemoryActionStackFinalizedTransition,
    ExternalLearnedStateLiveMemoryActionStackMemoryPreparation,
    ExternalLearnedStateLiveMemoryActionStackStartedFinalization,
    ExternalLearnedStateLiveMemoryActionStackStartedIntegrityReceipt,
    ExternalLearnedStateLiveMemoryActionStackStartedResult,
)
from alberta_framework.core.external_learned_state_live_memory_action_stack_adapter import (
    _tree_digest as _action_stack_tree_digest,
)
from alberta_framework.core.external_learned_state_live_memory_adapter import (
    EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ADAPTER_CONFIG_SCHEMA,
    ExternalLearnedStateLiveMemoryPendingBinding,
)
from alberta_framework.core.prototype_factorized_partner_planner import (
    FactorizedPartnerPlannerAgentState,
    PrototypeFactorizedPartnerPlanner,
    PrototypeFactorizedPartnerPlannerConfig,
    PrototypeFactorizedPartnerPlannerState,
)
from alberta_framework.core.recurrent_latent_world_model_ensemble import (
    RecurrentLatentWorldModelEnsembleConfig,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

N_ACTIONS = 2
MASK = jnp.ones((N_ACTIONS,), dtype=jnp.bool_)
OWNER = (
    0xA101A101,
    0xA202A202,
    0xA303A303,
    0xA404A404,
    0xA505A505,
    0xA606A606,
    0xA707A707,
    0xA808A808,
)
PLANNER_WORDS = jnp.asarray(
    (
        0xB101B101,
        0xB202B202,
        0xB303B303,
        0xB404B404,
        0xB505B505,
        0xB606B606,
        0xB707B707,
        0xB808B808,
    ),
    dtype=jnp.uint32,
)


@pytest.fixture(autouse=True)
def _bounded_jax_execution() -> object:
    with jax.disable_jit():
        yield


def _adapter() -> ExternalLearnedStateLiveMemoryActionStackAdapter:
    donor = _v1_adapter()
    return ExternalLearnedStateLiveMemoryActionStackAdapter(
        ExternalLearnedStateLiveMemoryActionStackConfig(
            coordinator=donor.config.coordinator,
            learned_memory=donor.config.learned_memory,
            final_action_owner_digest=OWNER,
        )
    )


def _start(
    adapter: ExternalLearnedStateLiveMemoryActionStackAdapter,
    seed: int,
) -> object:
    return adapter.start(
        adapter.init(jr.key(seed)),
        jnp.asarray((0.2, -0.1), dtype=jnp.float32),
        hard_action_mask=MASK,
    )


def _identity_finalize(
    adapter: ExternalLearnedStateLiveMemoryActionStackAdapter,
    prepared: ExternalLearnedStateLiveMemoryActionStackMemoryPreparation,
) -> ExternalLearnedStateLiveMemoryActionStackFinalizedTransition:
    prototype = prepared.memory_candidate_state.coordinator_state.inner_state.prototype_state
    action = prototype.current_action
    return adapter.bind_final_action(
        prepared,
        prototype,
        planner_action_before_mask=action,
        planner_candidate_words=PLANNER_WORDS,
        planner_consumed=jnp.asarray(False, dtype=jnp.bool_),
    )


def _identity_started_finalize(
    adapter: ExternalLearnedStateLiveMemoryActionStackAdapter,
    state: object,
) -> ExternalLearnedStateLiveMemoryActionStackStartedFinalization:
    prototype = state.coordinator_state.inner_state.prototype_state
    return adapter.prepare_started_final_action(
        state,
        prototype,
        planner_action_before_mask=prototype.current_action,
        planner_candidate_words=PLANNER_WORDS,
        planner_consumed=jnp.asarray(False, dtype=jnp.bool_),
    )


def _adopt(
    adapter: ExternalLearnedStateLiveMemoryActionStackAdapter,
    state: object,
    finalized: ExternalLearnedStateLiveMemoryActionStackFinalizedTransition,
) -> object:
    receipt = adapter.integrity_receipt(finalized)
    return adapter.adopt_finalized_transition(state, finalized, receipt)


def _replace_stored_action(state: object, slot: int, action: int) -> object:
    learned = state.learned_memory_state
    memory = learned.memory
    entries = dataclasses.replace(
        memory.entries,
        actions=memory.entries.actions.at[slot].set(
            jax.nn.one_hot(action, N_ACTIONS, dtype=jnp.float32)
        ),
    )
    return state.replace(
        learned_memory_state=learned.replace(memory=dataclasses.replace(memory, entries=entries))
    )


def _mutate_non_action_learner_leaf(prototype: object) -> object:
    """Change one finite learner parameter without touching decision identity."""

    slot = prototype.oak_state
    oak = slot.oak_state
    stomp = oak.stomp_state
    learner = stomp.base_learner_state
    biases = list(learner.head_params.biases)
    biases[0] = biases[0].at[0].add(jnp.asarray(0.125, dtype=jnp.float32))
    changed_learner = learner.replace(head_params=learner.head_params.replace(biases=tuple(biases)))
    return prototype.replace(
        oak_state=slot.replace(
            oak_state=oak.replace(stomp_state=stomp.replace(base_learner_state=changed_learner))
        )
    )


def _with_pair_reward_cells(
    state: PrototypeFactorizedPartnerPlannerState,
    reward_cells: jax.Array,
    *,
    reward_index: int,
) -> PrototypeFactorizedPartnerPlannerState:
    cells = jnp.asarray(reward_cells, dtype=jnp.float32)

    def replace_agent(
        agent: FactorizedPartnerPlannerAgentState,
    ) -> FactorizedPartnerPlannerAgentState:
        bias = jnp.zeros_like(agent.grounded.bias).at[:, reward_index].set(cells.reshape((-1,)))
        return agent.replace(
            grounded=agent.grounded.replace(
                weights=jnp.zeros_like(agent.grounded.weights),
                bias=bias,
            )
        )

    return state.replace(
        agent_0=replace_agent(state.agent_0),
        agent_1=replace_agent(state.agent_1),
    )


def _mutate_external_ensemble_learner(state: object) -> object:
    """Change one finite non-Prototype coordinator sidecar parameter."""

    coordinator = state.coordinator_state
    inner = coordinator.inner_state
    ensemble = inner.ensemble_state
    members = list(ensemble.member_states)
    learner = members[0].learner_state
    biases = list(learner.head_params.biases)
    biases[0] = biases[0].at[0].add(jnp.asarray(0.0625, dtype=jnp.float32))
    changed_learner = learner.replace(head_params=learner.head_params.replace(biases=tuple(biases)))
    members[0] = members[0].replace(learner_state=changed_learner)
    changed_inner = inner.replace(ensemble_state=ensemble.replace(member_states=tuple(members)))
    return state.replace(coordinator_state=coordinator.replace(inner_state=changed_inner))


def _retag_standard_finalized(
    adapter: ExternalLearnedStateLiveMemoryActionStackAdapter,
    finalized: ExternalLearnedStateLiveMemoryActionStackFinalizedTransition,
) -> ExternalLearnedStateLiveMemoryActionStackFinalizedTransition:
    binding = finalized.final_action_binding.replace(
        content_tag_words=adapter._final_binding_tag(finalized.final_action_binding)
    )
    bare = finalized.replace(
        final_action_binding=binding,
        content_tag_words=jnp.zeros_like(finalized.content_tag_words),
    )
    return bare.replace(content_tag_words=adapter._finalized_tag(bare))


def _retag_started_finalized(
    adapter: ExternalLearnedStateLiveMemoryActionStackAdapter,
    finalized: ExternalLearnedStateLiveMemoryActionStackStartedFinalization,
) -> ExternalLearnedStateLiveMemoryActionStackStartedFinalization:
    binding = finalized.final_action_binding.replace(
        content_tag_words=adapter._started_binding_tag(finalized.final_action_binding)
    )
    bare = finalized.replace(
        final_action_binding=binding,
        content_tag_words=jnp.zeros_like(finalized.content_tag_words),
    )
    return bare.replace(content_tag_words=adapter._started_finalized_tag(bare))


def _feedback(
    state: object,
    *,
    learn: bool = True,
) -> ExternalLearnedStateLiveMemoryActionStackFeedback:
    binding = state.action_binding
    assert bool(binding.memory_feedback_required)
    used = bool(binding.retrieval_used_expected)
    return ExternalLearnedStateLiveMemoryActionStackFeedback(
        action_binding_words=binding.content_tag_words,
        memory_transaction_words=binding.memory_transaction_words,
        prototype_decision_id=binding.prototype_decision_id,
        base_action=binding.base_action,
        memory_action=binding.memory_action,
        final_action=binding.final_action,
        hard_action_mask=binding.hard_action_mask,
        retrieval_used=binding.retrieval_used_expected,
        counterfactual_available=jnp.asarray(used and learn, dtype=jnp.bool_),
        counterfactual_delta=jnp.asarray(
            0.5 if used and learn else 0.0,
            dtype=jnp.float32,
        ),
    )


def _one_entry(
    adapter: ExternalLearnedStateLiveMemoryActionStackAdapter,
    *,
    seed: int,
) -> tuple[object, int, jax.Array]:
    state = _start(adapter, seed)
    prepared = adapter.prepare_memory_transition(
        state,
        _transition(state.coordinator_state),
        _event_input(provenance=31),
        MASK,
    )
    finalized = _identity_finalize(adapter, prepared)
    adopted = _adopt(adapter, state, finalized)
    assert bool(adopted.diagnostics.transaction_applied)
    memory_result = prepared.donor_prepared.learned_memory_result
    assert memory_result is not None
    return (
        adopted.state,
        int(memory_result.slot),
        state.coordinator_state.current_raw_observation,
    )


def _memory_then_planner_divergence(
    adapter: ExternalLearnedStateLiveMemoryActionStackAdapter,
    *,
    seed: int,
) -> tuple[object, ExternalLearnedStateLiveMemoryActionStackFinalizedTransition]:
    state, slot, stored_key = _one_entry(adapter, seed=seed)
    transition = _transition(
        state.coordinator_state,
        next_observation=tuple(float(value) for value in np.asarray(stored_key)),
    )
    preview = adapter.coordinator.step(state.coordinator_state, transition)
    memory_action = 1 - int(preview.state.current_action)
    state = _replace_stored_action(state, slot, memory_action)
    assert bool(adapter.state_valid(state))
    prepared = adapter.prepare_memory_transition(
        state,
        transition,
        _event_input(provenance=32),
        MASK,
    )
    assert bool(prepared.preparation_valid)
    assert int(prepared.memory_candidate_state.action_binding.memory_action) == memory_action

    memory_prototype = prepared.memory_candidate_state.coordinator_state.inner_state.prototype_state
    final_action = 1 - memory_action
    selected = adapter.coordinator.inner.prototype.replace_cached_primitive_action(
        memory_prototype,
        decision_id=memory_prototype.current_decision_id,
        decision_observation=memory_prototype.current_representation,
        proposed_action=jnp.asarray(final_action, dtype=jnp.int32),
        safety_action_mask=MASK,
    )
    assert bool(selected.committed)
    finalized = adapter.bind_final_action(
        prepared,
        selected.state,
        planner_action_before_mask=jnp.asarray(final_action, dtype=jnp.int32),
        planner_candidate_words=PLANNER_WORDS,
        planner_consumed=jnp.asarray(True, dtype=jnp.bool_),
    )
    assert bool(finalized.finalization_valid)
    adopted = _adopt(adapter, state, finalized)
    assert bool(adopted.diagnostics.transaction_applied)
    return adopted.state, finalized


def test_v2_genesis_is_explicit_b_equals_m_equals_p_and_v1_schema_is_unchanged() -> None:
    adapter = _adapter()
    state = _start(adapter, 401)
    binding = state.action_binding

    assert adapter.to_config()["schema"] == (
        EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ACTION_STACK_CONFIG_SCHEMA
    )
    assert _v1_adapter().to_config()["schema"] == (
        EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ADAPTER_CONFIG_SCHEMA
    )
    assert "final_action" not in {
        field.name for field in dataclasses.fields(ExternalLearnedStateLiveMemoryPendingBinding)
    }
    assert bool(binding.available)
    assert not bool(binding.memory_feedback_required)
    assert int(binding.base_action) == int(binding.memory_action)
    assert int(binding.memory_action) == int(binding.final_action)
    assert int(binding.memory_action_before_mask) == int(binding.memory_action)
    assert int(binding.planner_action_before_mask) == int(binding.final_action)
    assert not bool(binding.planner_bound)
    assert not bool(binding.planner_consumed)
    assert bool(jnp.any(binding.content_tag_words != 0))
    assert bool(adapter.state_valid(state))


def test_config_rejects_recurrent_cached_action_projection_lane() -> None:
    donor = _v1_adapter()
    coordinator = donor.config.coordinator
    prototype = coordinator.inner.prototype
    recurrent = RecurrentLatentWorldModelEnsembleConfig(
        observation_dim=prototype.oak.observation_dim,
        n_actions=N_ACTIONS,
    )
    # The coordinator's feature lifecycle independently rejects this lane at
    # canonical construction.  Mutate an isolated exact-type copy to exercise
    # the action-stack's own defensive config boundary.
    recurrent_coordinator = copy.deepcopy(coordinator)
    object.__setattr__(
        recurrent_coordinator.inner.prototype,
        "recurrent_latent_world_model_ensemble",
        recurrent,
    )
    with pytest.raises(ValueError, match="recurrent latent world-model"):
        ExternalLearnedStateLiveMemoryActionStackConfig(
            coordinator=recurrent_coordinator,
            learned_memory=donor.config.learned_memory,
            final_action_owner_digest=OWNER,
        )


def test_receipt_static_checks_reject_duck_metadata_impostors() -> None:
    adapter = _adapter()

    class DuckReceipt:
        source_state_words = jnp.zeros((8,), dtype=jnp.uint32)
        finalized_content_tag_words = jnp.zeros((8,), dtype=jnp.uint32)
        final_action_owner_words = jnp.asarray(OWNER, dtype=jnp.uint32)
        integrity_bound = jnp.asarray(True, dtype=jnp.bool_)
        content_tag_words = jnp.zeros((8,), dtype=jnp.uint32)

        def replace(self, **changes: object) -> DuckReceipt:
            del changes
            return self

    duck = DuckReceipt()
    assert not bool(adapter._receipt_static_contract_valid(duck))
    assert not bool(adapter._receipt_content_tag_valid(duck))
    assert not bool(adapter._started_receipt_static_contract_valid(duck))
    assert not bool(adapter._started_receipt_content_tag_valid(duck))


def test_valid_m_not_equal_p_keeps_feedback_m_bound_and_stores_executed_p() -> None:
    adapter = _adapter()
    state, finalized = _memory_then_planner_divergence(adapter, seed=403)
    binding = state.action_binding
    assert bool(binding.memory_feedback_required)
    assert bool(binding.retrieval_used_expected)
    assert bool(binding.planner_bound)
    assert bool(binding.planner_consumed)
    assert int(binding.memory_action) != int(binding.final_action)
    assert int(state.coordinator_state.current_action) == int(binding.final_action)
    assert bool(adapter.state_valid(state))
    assert int(finalized.bind_work.prototype_replacement_evaluations) == 0
    assert int(finalized.bind_work.coordinator_update_evaluations) == 0
    assert int(finalized.bind_work.planner_model_evaluations) == 0

    transition = _transition(state.coordinator_state, next_observation=(0.7, -0.3))
    prepared = adapter.prepare_memory_transition(
        state,
        transition,
        _event_input(provenance=33, query_uncertainty=2.0),
        MASK,
        _feedback(state),
    )
    assert bool(prepared.preparation_valid)
    assert bool(prepared.feedback_identity_valid)
    assert int(prepared.feedback.memory_action) == int(binding.memory_action)
    assert int(prepared.feedback.final_action) == int(binding.final_action)
    assert prepared.settlement_result is not None
    assert bool(prepared.settlement_result.diagnostics.transaction_applied)
    assert bool(prepared.settlement_result.diagnostics.learning_eligible)
    assert int(prepared.memory_candidate_state.learned_memory_state.learned_feedback_count) == 1
    assert int(prepared.prepare_work.feedback_settlement_evaluations) == 1
    assert int(prepared.prepare_work.coordinator_update_evaluations) == 1
    np.testing.assert_array_equal(
        prepared.donor_prepared.completed_entry.action,
        jax.nn.one_hot(binding.final_action, N_ACTIONS, dtype=jnp.float32),
    )


def test_transition_m_is_rejected_before_work_but_transition_p_is_accepted() -> None:
    adapter = _adapter()
    state, _ = _memory_then_planner_divergence(adapter, seed=409)
    binding = state.action_binding
    transition_p = _transition(state.coordinator_state, next_observation=(0.6, -0.4))
    transition_m = dataclasses.replace(transition_p, action=binding.memory_action)

    rejected = adapter.prepare_memory_transition(
        state,
        transition_m,
        _event_input(provenance=34),
        MASK,
        _feedback(state, learn=False),
    )
    assert not bool(rejected.preparation_valid)
    assert not bool(rejected.transition_final_action_exact)
    assert int(rejected.prepare_work.feedback_settlement_evaluations) == 0
    assert int(rejected.prepare_work.coordinator_update_evaluations) == 0
    assert int(rejected.prepare_work.learned_memory_query_evaluations) == 0
    _tree_exact(rejected.memory_candidate_state, state)

    wrong_memory_feedback = dataclasses.replace(
        _feedback(state, learn=False),
        memory_action=binding.final_action,
    )
    refused_feedback = adapter.prepare_memory_transition(
        state,
        transition_p,
        _event_input(provenance=34),
        MASK,
        wrong_memory_feedback,
    )
    assert not bool(refused_feedback.feedback_identity_valid)
    assert int(refused_feedback.prepare_work.feedback_settlement_evaluations) == 0
    assert int(refused_feedback.prepare_work.coordinator_update_evaluations) == 0
    _tree_exact(refused_feedback.memory_candidate_state, state)

    accepted = adapter.prepare_memory_transition(
        state,
        transition_p,
        _event_input(provenance=34),
        MASK,
        _feedback(state, learn=False),
    )
    assert bool(accepted.preparation_valid)
    assert bool(accepted.transition_final_action_exact)
    assert int(accepted.prepare_work.feedback_settlement_evaluations) == 1
    assert int(accepted.prepare_work.coordinator_update_evaluations) == 1
    assert int(accepted.prepare_work.learned_memory_query_evaluations) == 1


def test_only_finalized_content_can_be_bound_and_tamper_or_replay_rolls_back() -> None:
    adapter = _adapter()
    state = _start(adapter, 419)
    prepared = adapter.prepare_memory_transition(
        state,
        _transition(state.coordinator_state),
        _event_input(provenance=35),
        MASK,
    )
    with pytest.raises(TypeError, match="finalized"):
        adapter.integrity_receipt(prepared)

    finalized = _identity_finalize(adapter, prepared)
    receipt = adapter.integrity_receipt(finalized)
    tampered = finalized.replace(
        candidate_state=finalized.candidate_state.replace(
            action_binding=finalized.candidate_state.action_binding.replace(
                final_action=(1 - finalized.candidate_state.action_binding.final_action).astype(
                    jnp.int32
                )
            )
        )
    )
    refused = adapter.adopt_finalized_transition(state, tampered, receipt)
    assert not bool(refused.diagnostics.finalized_content_matches)
    assert not bool(refused.diagnostics.transaction_applied)
    _tree_exact(refused.state, state)

    accepted = adapter.adopt_finalized_transition(state, finalized, receipt)
    assert bool(accepted.diagnostics.transaction_applied)

    bool_int_alias = receipt.replace(
        integrity_bound=jnp.asarray(1, dtype=jnp.int32),
    )
    alias_refused = adapter.adopt_finalized_transition(state, finalized, bool_int_alias)
    assert not bool(alias_refused.diagnostics.receipt_static_contract_valid)
    assert not bool(alias_refused.diagnostics.receipt_matches)
    assert not bool(alias_refused.diagnostics.transaction_applied)
    _tree_exact(alias_refused.state, state)

    wrong_shape = receipt.replace(
        source_state_words=receipt.source_state_words.reshape((2, 4)),
    )
    shape_refused = adapter.adopt_finalized_transition(state, finalized, wrong_shape)
    assert not bool(shape_refused.diagnostics.receipt_static_contract_valid)
    assert not bool(shape_refused.diagnostics.transaction_applied)
    _tree_exact(shape_refused.state, state)

    wrong_dtype = receipt.replace(
        final_action_owner_words=receipt.final_action_owner_words.astype(jnp.int32),
    )
    dtype_refused = adapter.adopt_finalized_transition(state, finalized, wrong_dtype)
    assert not bool(dtype_refused.diagnostics.receipt_static_contract_valid)
    assert not bool(dtype_refused.diagnostics.transaction_applied)
    _tree_exact(dtype_refused.state, state)

    retagged_wrong_static = bool_int_alias.replace(
        content_tag_words=adapter._receipt_tag(bool_int_alias),
    )
    retagged_refused = adapter.adopt_finalized_transition(
        state,
        finalized,
        retagged_wrong_static,
    )
    assert not bool(retagged_refused.diagnostics.receipt_static_contract_valid)
    assert bool(retagged_refused.diagnostics.receipt_content_tag_valid)
    assert not bool(retagged_refused.diagnostics.transaction_applied)
    _tree_exact(retagged_refused.state, state)

    replay = adapter.adopt_finalized_transition(accepted.state, finalized, receipt)
    assert not bool(replay.diagnostics.source_state_matches)
    assert not bool(replay.diagnostics.transaction_applied)
    _tree_exact(replay.state, accepted.state)
    assert int(replay.adoption_work.donor_evaluations) == 0


def test_final_bind_rejects_valid_non_action_learner_injection_without_work() -> None:
    adapter = _adapter()
    state = _start(adapter, 421)
    prepared = adapter.prepare_memory_transition(
        state,
        _transition(state.coordinator_state),
        _event_input(provenance=351),
        MASK,
    )
    assert bool(prepared.preparation_valid)
    source = prepared.memory_candidate_state.coordinator_state.inner_state.prototype_state
    injected = _mutate_non_action_learner_leaf(source)
    assert bool(adapter.coordinator.inner.prototype.validate_state(injected))
    _tree_exact(injected.current_decision_id, source.current_decision_id)
    _tree_exact(injected.current_raw_observation, source.current_raw_observation)
    _tree_exact(injected.current_representation, source.current_representation)
    _tree_exact(injected.step_words, source.step_words)

    refused = adapter.bind_final_action(
        prepared,
        injected,
        planner_action_before_mask=injected.current_action,
        planner_candidate_words=PLANNER_WORDS,
        planner_consumed=jnp.asarray(False, dtype=jnp.bool_),
    )
    assert not bool(refused.finalization_valid)
    _tree_exact(refused.candidate_state, prepared.source_state)
    assert int(refused.bind_work.final_action_binding_evaluations) == 1
    assert int(refused.bind_work.prototype_replacement_evaluations) == 0
    assert int(refused.bind_work.coordinator_update_evaluations) == 0
    assert int(refused.bind_work.planner_model_evaluations) == 0
    assert int(refused.bind_work.learned_memory_evaluations) == 0


def test_standard_receipt_reconstructs_and_rejects_external_semantic_forgery() -> None:
    adapter = _adapter()
    state = _start(adapter, 422)
    prepared = adapter.prepare_memory_transition(
        state,
        _transition(state.coordinator_state),
        _event_input(provenance=352),
        MASK,
    )
    finalized = _identity_finalize(adapter, prepared)
    assert bool(finalized.finalization_valid)
    binding = finalized.final_action_binding

    forged_bindings = (
        binding.replace(hard_action_mask=binding.hard_action_mask.reshape((1, N_ACTIONS))),
        binding.replace(planner_consumed=jnp.asarray(0, dtype=jnp.int32)),
        binding.replace(planner_consumed=~binding.planner_consumed),
        binding.replace(
            planner_action_before_mask=(1 - binding.planner_action_before_mask).astype(jnp.int32)
        ),
        binding.replace(hard_action_mask=jnp.zeros_like(binding.hard_action_mask)),
    )
    for forged_binding in forged_bindings:
        forged = _retag_standard_finalized(
            adapter,
            finalized.replace(final_action_binding=forged_binding),
        )
        with pytest.raises(ValueError, match="recomputed finalized contract"):
            adapter.integrity_receipt(forged)

    unrelated_candidate = _mutate_external_ensemble_learner(finalized.candidate_state)
    assert bool(adapter.state_valid(unrelated_candidate))
    unrelated = _retag_standard_finalized(
        adapter,
        finalized.replace(candidate_state=unrelated_candidate),
    )
    with pytest.raises(ValueError, match="recomputed finalized contract"):
        adapter.integrity_receipt(unrelated)

    forged_work = _retag_standard_finalized(
        adapter,
        finalized.replace(
            bind_work=finalized.bind_work.replace(
                final_action_binding_evaluations=jnp.asarray(0, dtype=jnp.int32)
            )
        ),
    )
    with pytest.raises(ValueError, match="recomputed finalized contract"):
        adapter.integrity_receipt(forged_work)


def test_standard_bind_rejects_retagged_memory_preparation_semantic_forgery() -> None:
    adapter = _adapter()
    state, _ = _memory_then_planner_divergence(adapter, seed=433)
    binding = state.action_binding
    assert int(binding.memory_action) != int(binding.final_action)
    transition = _transition(
        state.coordinator_state,
        next_observation=(0.6, -0.4),
    )
    event_input = _event_input(provenance=356)
    feedback = _feedback(state)
    for unsupported in (
        {"partner_policy_fusion_input": object()},
        {"partner_policy_fusion_feedback": object()},
        {"extended_action_mask": jnp.ones((N_ACTIONS,), dtype=jnp.bool_)},
    ):
        with pytest.raises(ValueError, match="lack an exact stored result-semantic binding"):
            adapter.prepare_memory_transition(
                state,
                transition,
                event_input,
                MASK,
                feedback,
                **unsupported,
            )
    prepared = adapter.prepare_memory_transition(
        state,
        transition,
        event_input,
        MASK,
        feedback,
    )
    assert bool(prepared.preparation_valid)
    donor = prepared.donor_prepared
    assert donor is not None
    settlement = prepared.settlement_result
    assert settlement is not None

    changed_candidate = _mutate_external_ensemble_learner(
        prepared.memory_candidate_state
    )
    assert bool(adapter.state_valid(changed_candidate))
    changed_v1_source = _mutate_external_ensemble_learner(donor.source_state)
    changed_v1_candidate = _mutate_external_ensemble_learner(donor.candidate_state)
    changed_coordinator = donor.coordinator_result
    changed_memory = donor.learned_memory_result
    assert changed_coordinator is not None
    assert changed_memory is not None

    forged_preparations = (
        prepared.replace(
            transition=prepared.transition.replace(
                action=binding.memory_action,
            )
        ),
        prepared.replace(
            transition_final_action_exact=jnp.asarray(False, dtype=jnp.bool_),
        ),
        prepared.replace(
            feedback_identity_valid=jnp.asarray(False, dtype=jnp.bool_),
        ),
        prepared.replace(
            prepare_work=prepared.prepare_work.replace(
                coordinator_update_evaluations=jnp.asarray(0, dtype=jnp.int32),
            )
        ),
        prepared.replace(
            settlement_result=settlement.replace(
                diagnostics=settlement.diagnostics.replace(
                    transaction_applied=jnp.asarray(False, dtype=jnp.bool_),
                )
            )
        ),
        prepared.replace(memory_candidate_state=changed_candidate),
        prepared.replace(
            donor_prepared=donor.replace(
                transition=donor.transition.replace(
                    action=binding.memory_action,
                )
            )
        ),
        prepared.replace(
            donor_prepared=donor.replace(
                event_input=donor.event_input.replace(
                    provenance_id=(donor.event_input.provenance_id + 1).astype(jnp.int32),
                )
            )
        ),
        prepared.replace(
            donor_prepared=donor.replace(
                hard_action_mask=donor.hard_action_mask.at[0].set(
                    ~donor.hard_action_mask[0]
                )
            )
        ),
        prepared.replace(
            donor_prepared=donor.replace(
                extended_action_mask=jnp.ones((N_ACTIONS,), dtype=jnp.bool_),
            )
        ),
        prepared.replace(
            donor_prepared=donor.replace(
                completed_entry=donor.completed_entry.replace(
                    reward=(donor.completed_entry.reward + 0.25).astype(jnp.float32),
                )
            )
        ),
        prepared.replace(
            donor_prepared=donor.replace(
                query_key=donor.query_key.at[0].add(jnp.asarray(0.125, dtype=jnp.float32)),
            )
        ),
        prepared.replace(
            donor_prepared=donor.replace(source_state=changed_v1_source)
        ),
        prepared.replace(
            donor_prepared=donor.replace(candidate_state=changed_v1_candidate)
        ),
        prepared.replace(
            donor_prepared=donor.replace(
                coordinator_result=changed_coordinator.replace(
                    diagnostics=changed_coordinator.diagnostics.replace(
                        transaction_applied=jnp.asarray(False, dtype=jnp.bool_),
                    )
                )
            )
        ),
        prepared.replace(
            donor_prepared=donor.replace(
                learned_memory_result=changed_memory.replace(
                    wrote=jnp.asarray(False, dtype=jnp.bool_),
                )
            )
        ),
    )
    forged_work = tuple(
        prepared.replace(
            prepare_work=prepared.prepare_work.replace(
                **{
                    field.name: (
                        getattr(prepared.prepare_work, field.name)
                        + jnp.asarray(1, dtype=jnp.int32)
                    )
                }
            )
        )
        for field in dataclasses.fields(prepared.prepare_work)
    )
    forged_donor_work = tuple(
        prepared.replace(
            donor_prepared=donor.replace(
                **{
                    name: getattr(donor, name) + jnp.asarray(1, dtype=jnp.int32)
                }
            )
        )
        for name in (
            "settlement_evaluations",
            "coordinator_evaluations",
            "learned_memory_query_evaluations",
            "learned_memory_write_evaluations",
            "cached_action_replacement_evaluations",
        )
    )
    for forged in forged_preparations + forged_work + forged_donor_work:
        forged = forged.replace(
            content_tag_words=adapter._memory_preparation_tag(forged)
        )
        finalized = _identity_finalize(adapter, forged)
        assert not bool(finalized.finalization_valid)
        _tree_exact(finalized.candidate_state, forged.source_state)
        with pytest.raises(ValueError, match="recomputed finalized contract"):
            adapter.integrity_receipt(finalized)


def test_standard_bind_rejects_malformed_nested_donor_preparation() -> None:
    adapter = _adapter()
    state = _start(adapter, 434)
    prepared = adapter.prepare_memory_transition(
        state,
        _transition(state.coordinator_state),
        _event_input(provenance=357),
        MASK,
    )
    donor = prepared.donor_prepared
    assert donor is not None
    malformed = prepared.replace(
        donor_prepared=donor.replace(coordinator_result=object())
    )
    malformed = malformed.replace(
        content_tag_words=adapter._memory_preparation_tag(malformed)
    )
    with pytest.raises(ValueError, match="malformed static contract"):
        _identity_finalize(adapter, malformed)


def test_standard_malformed_nested_records_fail_closed_to_exact_caller_state() -> None:
    adapter = _adapter()
    state = _start(adapter, 427)
    prepared = adapter.prepare_memory_transition(
        state,
        _transition(state.coordinator_state),
        _event_input(provenance=353),
        MASK,
    )
    finalized = _identity_finalize(adapter, prepared)
    receipt = adapter.integrity_receipt(finalized)

    malformed_action_binding = finalized.candidate_state.action_binding.replace(
        final_action=jnp.asarray(
            (int(finalized.candidate_state.action_binding.final_action),), dtype=jnp.int32
        )
    )
    malformed_action_binding = malformed_action_binding.replace(
        content_tag_words=adapter._binding_tag(malformed_action_binding)
    )
    malformed_candidate = finalized.candidate_state.replace(action_binding=malformed_action_binding)
    malformed_selected = _retag_standard_finalized(
        adapter,
        finalized.replace(
            final_action_binding=finalized.final_action_binding.replace(
                selected_prototype_state=object()
            )
        ),
    )
    malformed_preparation = prepared.replace(memory_candidate_state=object())
    malformed_preparation = malformed_preparation.replace(
        content_tag_words=adapter._memory_preparation_tag(malformed_preparation)
    )
    records = (
        _retag_standard_finalized(
            adapter,
            finalized.replace(candidate_state=malformed_candidate),
        ),
        _retag_standard_finalized(
            adapter,
            finalized.replace(
                finalization_valid=jnp.asarray((True,), dtype=jnp.bool_),
            ),
        ),
        malformed_selected,
        _retag_standard_finalized(
            adapter,
            finalized.replace(memory_preparation=malformed_preparation),
        ),
    )
    for malformed in records:
        with pytest.raises(ValueError, match="recomputed finalized contract"):
            adapter.integrity_receipt(malformed)
        refused = adapter.adopt_finalized_transition(state, malformed, receipt)
        assert refused.diagnostics.transaction_applied.shape == ()
        assert refused.diagnostics.transaction_applied.dtype == jnp.bool_
        assert not bool(refused.diagnostics.transaction_applied)
        assert int(refused.adoption_work.final_action_binding_reconstructions) == 0
        _tree_exact(refused.state, state)


def test_standard_adoption_reconstructs_once_and_malformed_receipts_roll_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter()
    state = _start(adapter, 428)
    prepared = adapter.prepare_memory_transition(
        state,
        _transition(state.coordinator_state),
        _event_input(provenance=355),
        MASK,
    )
    finalized = _identity_finalize(adapter, prepared)
    receipt = adapter.integrity_receipt(finalized)
    original = adapter.bind_final_action
    calls = 0

    def counted(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(adapter, "bind_final_action", counted)
    adopted = adapter.adopt_finalized_transition(state, finalized, receipt)
    assert bool(adopted.diagnostics.transaction_applied)
    assert calls == 1
    assert int(adopted.adoption_work.final_action_binding_reconstructions) == 1
    assert int(adopted.adoption_work.donor_evaluations) == 0
    assert int(adopted.adoption_work.prototype_replacement_evaluations) == 0
    assert int(adopted.adoption_work.planner_model_evaluations) == 0

    receipt_fields = (
        "source_state_words",
        "finalized_content_tag_words",
        "final_action_owner_words",
        "integrity_bound",
        "content_tag_words",
    )
    for name in receipt_fields:
        malformed_receipt = receipt.replace(**{name: object()})
        refused = adapter.adopt_finalized_transition(
            state,
            finalized,
            malformed_receipt,
        )
        assert refused.diagnostics.transaction_applied.shape == ()
        assert refused.diagnostics.transaction_applied.dtype == jnp.bool_
        assert not bool(refused.diagnostics.transaction_applied)
        assert int(refused.adoption_work.final_action_binding_reconstructions) == 1
        _tree_exact(refused.state, state)


def test_real_pair_planner_started_bind_and_adopt_is_exact_zero_work_transaction() -> None:
    adapter = _adapter()
    source_0 = _start(adapter, 423)
    source_1 = _start(adapter, 424)
    prototype = adapter.coordinator.inner.prototype
    prototype_0 = source_0.coordinator_state.inner_state.prototype_state
    prototype_1 = source_1.coordinator_state.inner_state.prototype_state
    raw_dim = int(prototype_0.current_raw_observation.shape[0])
    representation_dim = int(prototype_0.current_representation.shape[0])
    planner = PrototypeFactorizedPartnerPlanner(
        prototype,
        PrototypeFactorizedPartnerPlannerConfig(
            observation_dim=raw_dim,
            prototype_representation_dim=representation_dim,
            n_actions=N_ACTIONS,
            planning_enabled=True,
        ),
    )
    target = 1 - int(prototype_0.current_action)
    reward_cells = jnp.zeros((N_ACTIONS, N_ACTIONS), dtype=jnp.float32).at[target, :].set(5.0)
    planner_source = _with_pair_reward_cells(
        planner.init(jr.key(425)),
        reward_cells,
        reward_index=raw_dim,
    )
    prepared_pair = planner.prepare_pair(
        planner_source,
        prototype_0,
        prototype_1,
        jnp.ones((2, N_ACTIONS), dtype=jnp.bool_),
    )
    assert bool(prepared_pair.diagnostics.pair_committed)
    assert int(prepared_pair.diagnostics.proposed_actions[0]) == target
    assert int(prepared_pair.prototype_agent_0.current_action) == target
    assert int(prototype_0.current_action) != target
    _tree_exact(
        planner_source.agent_0.behavior.step_words, prepared_pair.state.agent_0.behavior.step_words
    )
    _tree_exact(
        planner_source.agent_0.grounded.update_words,
        prepared_pair.state.agent_0.grounded.update_words,
    )
    pair_words = _action_stack_tree_digest("exact-started-pair-candidate-v2", prepared_pair)
    assert bool(jnp.any(pair_words != 0))

    finalized = adapter.prepare_started_final_action(
        source_0,
        prepared_pair.prototype_agent_0,
        planner_action_before_mask=prepared_pair.diagnostics.proposed_actions[0],
        planner_candidate_words=pair_words,
        planner_consumed=prepared_pair.state.agent_0.cache.planner_consumed,
    )
    assert type(finalized) is ExternalLearnedStateLiveMemoryActionStackStartedFinalization
    assert bool(finalized.finalization_valid)
    candidate = finalized.candidate_state
    before = source_0.action_binding
    after = candidate.action_binding
    assert int(before.base_action) == int(after.base_action)
    assert int(before.memory_action) == int(after.memory_action)
    assert int(after.final_action) == target
    assert int(after.memory_action) != int(after.final_action)
    assert bool(after.planner_bound)
    assert bool(after.planner_consumed)
    _tree_exact(after.planner_candidate_words, pair_words)
    _tree_exact(candidate.learned_memory_state, source_0.learned_memory_state)
    _tree_exact(candidate.coordinator_state.event_words, source_0.coordinator_state.event_words)
    _tree_exact(
        candidate.coordinator_state.inner_state.prototype_state.step_words,
        prototype_0.step_words,
    )
    assert int(finalized.bind_work.final_action_binding_evaluations) == 1
    assert int(finalized.bind_work.prototype_replacement_evaluations) == 0
    assert int(finalized.bind_work.coordinator_update_evaluations) == 0
    assert int(finalized.bind_work.planner_model_evaluations) == 0
    assert int(finalized.bind_work.learned_memory_evaluations) == 0

    receipt = adapter.started_final_action_integrity_receipt(finalized)
    assert type(receipt) is ExternalLearnedStateLiveMemoryActionStackStartedIntegrityReceipt
    adopted = adapter.adopt_started_final_action(source_0, finalized, receipt)
    assert type(adopted) is ExternalLearnedStateLiveMemoryActionStackStartedResult
    assert bool(adopted.diagnostics.transaction_applied)
    _tree_exact(adopted.state, candidate)
    assert int(adopted.adoption_work.donor_evaluations) == 0
    assert int(adopted.adoption_work.coordinator_update_evaluations) == 0
    assert int(adopted.adoption_work.prototype_replacement_evaluations) == 0
    assert int(adopted.adoption_work.planner_model_evaluations) == 0
    assert int(adopted.adoption_work.learned_memory_evaluations) == 0

    receipt_alias = receipt.replace(
        integrity_bound=jnp.asarray(1, dtype=jnp.int32),
    )
    receipt_alias_refused = adapter.adopt_started_final_action(
        source_0,
        finalized,
        receipt_alias,
    )
    assert not bool(receipt_alias_refused.diagnostics.receipt_static_contract_valid)
    assert not bool(receipt_alias_refused.diagnostics.transaction_applied)
    _tree_exact(receipt_alias_refused.state, source_0)

    receipt_shape = receipt.replace(
        finalized_content_tag_words=receipt.finalized_content_tag_words.reshape((2, 4)),
    )
    receipt_shape_refused = adapter.adopt_started_final_action(
        source_0,
        finalized,
        receipt_shape,
    )
    assert not bool(receipt_shape_refused.diagnostics.receipt_static_contract_valid)
    assert not bool(receipt_shape_refused.diagnostics.transaction_applied)
    _tree_exact(receipt_shape_refused.state, source_0)

    stale = adapter.adopt_started_final_action(source_1, finalized, receipt)
    assert not bool(stale.diagnostics.source_state_matches)
    assert not bool(stale.diagnostics.transaction_applied)
    _tree_exact(stale.state, source_1)

    tampered = finalized.replace(
        candidate_state=finalized.candidate_state.replace(
            action_binding=finalized.candidate_state.action_binding.replace(
                final_action=finalized.source_state.action_binding.final_action,
            )
        )
    )
    refused = adapter.adopt_started_final_action(source_0, tampered, receipt)
    assert not bool(refused.diagnostics.finalized_content_matches)
    assert not bool(refused.diagnostics.transaction_applied)
    _tree_exact(refused.state, source_0)

    replay = adapter.adopt_started_final_action(adopted.state, finalized, receipt)
    assert not bool(replay.diagnostics.source_state_matches)
    assert not bool(replay.diagnostics.transaction_applied)
    _tree_exact(replay.state, adopted.state)

    already_bound = adapter.prepare_started_final_action(
        adopted.state,
        prepared_pair.prototype_agent_0,
        planner_action_before_mask=prepared_pair.diagnostics.proposed_actions[0],
        planner_candidate_words=pair_words,
        planner_consumed=prepared_pair.state.agent_0.cache.planner_consumed,
    )
    assert not bool(already_bound.finalization_valid)
    with pytest.raises(ValueError, match="recomputed started contract"):
        adapter.started_final_action_integrity_receipt(already_bound)

    non_genesis, _, _ = _one_entry(adapter, seed=426)
    non_genesis_refused = adapter.prepare_started_final_action(
        non_genesis,
        non_genesis.coordinator_state.inner_state.prototype_state,
        planner_action_before_mask=non_genesis.action_binding.final_action,
        planner_candidate_words=pair_words,
        planner_consumed=jnp.asarray(True, dtype=jnp.bool_),
    )
    assert not bool(non_genesis_refused.finalization_valid)
    _tree_exact(non_genesis_refused.candidate_state, non_genesis)


def test_started_receipt_reconstructs_and_rejects_external_semantic_forgery() -> None:
    adapter = _adapter()
    state = _start(adapter, 429)
    finalized = _identity_started_finalize(adapter, state)
    assert bool(finalized.finalization_valid)
    binding = finalized.final_action_binding
    forged_bindings = (
        binding.replace(hard_action_mask=binding.hard_action_mask.reshape((1, N_ACTIONS))),
        binding.replace(planner_consumed=jnp.asarray(0, dtype=jnp.int32)),
        binding.replace(planner_consumed=~binding.planner_consumed),
        binding.replace(
            planner_action_before_mask=(1 - binding.planner_action_before_mask).astype(jnp.int32)
        ),
        binding.replace(hard_action_mask=jnp.zeros_like(binding.hard_action_mask)),
    )
    for forged_binding in forged_bindings:
        forged = _retag_started_finalized(
            adapter,
            finalized.replace(final_action_binding=forged_binding),
        )
        with pytest.raises(ValueError, match="recomputed started contract"):
            adapter.started_final_action_integrity_receipt(forged)

    unrelated_candidate = _mutate_external_ensemble_learner(finalized.candidate_state)
    assert bool(adapter.state_valid(unrelated_candidate))
    unrelated = _retag_started_finalized(
        adapter,
        finalized.replace(candidate_state=unrelated_candidate),
    )
    with pytest.raises(ValueError, match="recomputed started contract"):
        adapter.started_final_action_integrity_receipt(unrelated)

    forged_work = _retag_started_finalized(
        adapter,
        finalized.replace(
            bind_work=finalized.bind_work.replace(
                final_action_binding_evaluations=jnp.asarray(0, dtype=jnp.int32)
            )
        ),
    )
    forged_genesis_flag = _retag_started_finalized(
        adapter,
        finalized.replace(source_genesis_valid=jnp.asarray(False, dtype=jnp.bool_)),
    )
    for forged in (forged_work, forged_genesis_flag):
        with pytest.raises(ValueError, match="recomputed started contract"):
            adapter.started_final_action_integrity_receipt(forged)


def test_started_malformed_nested_records_fail_closed_to_exact_caller_state() -> None:
    adapter = _adapter()
    state = _start(adapter, 430)
    finalized = _identity_started_finalize(adapter, state)
    receipt = adapter.started_final_action_integrity_receipt(finalized)

    malformed_action_binding = finalized.candidate_state.action_binding.replace(
        final_action=jnp.asarray(
            (int(finalized.candidate_state.action_binding.final_action),), dtype=jnp.int32
        )
    )
    malformed_action_binding = malformed_action_binding.replace(
        content_tag_words=adapter._binding_tag(malformed_action_binding)
    )
    malformed_candidate = finalized.candidate_state.replace(action_binding=malformed_action_binding)
    records = (
        _retag_started_finalized(
            adapter,
            finalized.replace(candidate_state=malformed_candidate),
        ),
        _retag_started_finalized(
            adapter,
            finalized.replace(
                finalization_valid=jnp.asarray((True,), dtype=jnp.bool_),
            ),
        ),
        _retag_started_finalized(
            adapter,
            finalized.replace(
                final_action_binding=finalized.final_action_binding.replace(
                    selected_prototype_state=object()
                )
            ),
        ),
        _retag_started_finalized(
            adapter,
            finalized.replace(source_state=object()),
        ),
    )
    for malformed in records:
        with pytest.raises(ValueError, match="recomputed started contract"):
            adapter.started_final_action_integrity_receipt(malformed)
        refused = adapter.adopt_started_final_action(state, malformed, receipt)
        assert refused.diagnostics.transaction_applied.shape == ()
        assert refused.diagnostics.transaction_applied.dtype == jnp.bool_
        assert not bool(refused.diagnostics.transaction_applied)
        assert int(refused.adoption_work.final_action_binding_reconstructions) == 0
        _tree_exact(refused.state, state)


def test_started_adoption_reconstructs_once_and_malformed_receipts_roll_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter()
    state = _start(adapter, 431)
    finalized = _identity_started_finalize(adapter, state)
    receipt = adapter.started_final_action_integrity_receipt(finalized)
    original = adapter.prepare_started_final_action
    calls = 0

    def counted(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(adapter, "prepare_started_final_action", counted)
    adopted = adapter.adopt_started_final_action(state, finalized, receipt)
    assert bool(adopted.diagnostics.transaction_applied)
    assert calls == 1
    assert int(adopted.adoption_work.final_action_binding_reconstructions) == 1
    assert int(adopted.adoption_work.donor_evaluations) == 0
    assert int(adopted.adoption_work.prototype_replacement_evaluations) == 0
    assert int(adopted.adoption_work.planner_model_evaluations) == 0

    receipt_fields = (
        "source_state_words",
        "finalized_content_tag_words",
        "final_action_owner_words",
        "integrity_bound",
        "content_tag_words",
    )
    for name in receipt_fields:
        malformed_receipt = receipt.replace(**{name: object()})
        refused = adapter.adopt_started_final_action(
            state,
            finalized,
            malformed_receipt,
        )
        assert refused.diagnostics.transaction_applied.shape == ()
        assert refused.diagnostics.transaction_applied.dtype == jnp.bool_
        assert not bool(refused.diagnostics.transaction_applied)
        assert int(refused.adoption_work.final_action_binding_reconstructions) == 1
        _tree_exact(refused.state, state)


def test_action_stack_public_host_only_surfaces_reject_tracers() -> None:
    adapter = _adapter()
    state = _start(adapter, 432)
    prepared = adapter.prepare_memory_transition(
        state,
        _transition(state.coordinator_state),
        _event_input(provenance=354),
        MASK,
    )
    finalized = _identity_finalize(adapter, prepared)
    started_finalized = _identity_started_finalize(adapter, state)
    with pytest.raises(RuntimeError, match="state validation is host-only"):
        jax.make_jaxpr(adapter.state_valid)(state)
    with pytest.raises(RuntimeError, match="receipt creation is host-only"):
        jax.make_jaxpr(adapter.integrity_receipt)(finalized)
    with pytest.raises(RuntimeError, match="receipt creation is host-only"):
        jax.make_jaxpr(adapter.started_final_action_integrity_receipt)(started_finalized)

    v1 = _v1_adapter()
    v1_state = v1.start(
        v1.init(jr.key(433)),
        jnp.asarray((0.2, -0.1), dtype=jnp.float32),
    )

    def traced_upgrade(mask: jax.Array) -> object:
        return adapter.upgrade_v1_state(v1, v1_state, hard_action_mask=mask)

    with pytest.raises(RuntimeError, match="v1 upgrade is host-only"):
        jax.make_jaxpr(traced_upgrade)(MASK)


def test_explicit_v1_upgrade_is_exactly_p_equals_m_and_preserves_pending_feedback() -> None:
    v1 = _v1_adapter()
    v1_state = v1.start(
        v1.init(jr.key(431)),
        jnp.asarray((0.2, -0.1), dtype=jnp.float32),
    )
    first = v1.step(
        v1_state,
        _transition(v1_state.coordinator_state),
        _event_input(provenance=36),
        MASK,
    )
    assert bool(v1.state_valid(first.state))

    adapter = _adapter()
    upgraded = adapter.upgrade_v1_state(v1, first.state, hard_action_mask=MASK)
    assert bool(adapter.state_valid(upgraded))
    assert int(upgraded.action_binding.memory_action) == int(upgraded.action_binding.final_action)
    assert bool(upgraded.action_binding.memory_feedback_required) == bool(
        first.state.pending_binding.available
    )
    _tree_exact(upgraded.coordinator_state, first.state.coordinator_state)
    _tree_exact(upgraded.learned_memory_state, first.state.learned_memory_state)
