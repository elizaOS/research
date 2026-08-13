# mypy: disable-error-code="attr-defined,call-arg"
"""Focused contracts for default-off calibrated search -> live dispatch v2."""

from __future__ import annotations

from typing import cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework as alberta
from alberta_framework.core.calibrated_extended_search_control import (
    CANDIDATE_KIND_OPTION,
    CANDIDATE_KIND_PRIMITIVE,
    SEARCH_MODE_COMBINED,
    CalibratedExtendedSearchControlConfig,
)
from alberta_framework.core.oak import OaKConfig, OaKState
from alberta_framework.core.options import (
    DISPATCH_OWNER_BASE_PRIMITIVE,
    DISPATCH_OWNER_OPTION,
    STOMPConfig,
    SubtaskSpec,
)
from alberta_framework.core.prototype_agent import (
    PrototypeAgentState,
    PrototypeTransition,
)
from alberta_framework.core.prototype_stomp_calibrated_dispatch import (
    PROTOTYPE_STOMP_CALIBRATED_DISPATCH_ASSESSMENT,
    PROTOTYPE_STOMP_CALIBRATED_DISPATCH_ERROR_CLOCK_EXHAUSTED,
    PROTOTYPE_STOMP_CALIBRATED_DISPATCH_EVIDENCE_LEVEL,
    PROTOTYPE_STOMP_CALIBRATED_DISPATCH_SCIENTIFIC_PROMOTION_ALLOWED,
    PrototypeSTOMPCalibratedDispatchAgent,
    PrototypeSTOMPCalibratedDispatchConfig,
    PrototypeSTOMPCalibratedDispatchState,
)
from alberta_framework.core.prototype_stomp_calibrated_search import (
    PrototypeSTOMPCalibratedSearchAgent,
    PrototypeSTOMPCalibratedSearchConfig,
)
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig

pytestmark = pytest.mark.unit

ANCHORS = jnp.asarray(
    ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)),
    dtype=jnp.float32,
)
ACTIVE = jnp.ones((4,), dtype=jnp.bool_)
SOURCE = jnp.asarray((0xC0DE, 0xD15A), dtype=jnp.uint32)
ALL_SAFE = jnp.ones((2,), dtype=jnp.bool_)
NONE_SAFE = jnp.zeros((2,), dtype=jnp.bool_)
_MAX_WORDS = jnp.asarray((0xFFFFFFFF, 0xFFFFFFFF), dtype=jnp.uint32)


def _sidecar_config(*, enabled: bool = True) -> PrototypeSTOMPCalibratedSearchConfig:
    from alberta_framework.core.prototype_agent import PrototypeAgentConfig

    prototype = PrototypeAgentConfig(
        oak=OaKConfig(
            stomp=STOMPConfig(
                subtask_specs=(
                    SubtaskSpec(
                        feature_index=0,
                        threshold=100.0,
                        pseudo_reward_scale=1.0,
                        max_option_steps=3,
                    ),
                ),
                observation_dim=2,
                n_primitive_actions=2,
                base_step_size=0.01,
                base_avg_reward_step_size=0.01,
                option_step_size=0.01,
                option_avg_reward_step_size=0.01,
                option_model_decay=0.0,
                option_model_step_size=0.2,
                option_planning_backups_per_step=0,
                epsilon_base=0.0,
                epsilon_option=0.0,
            )
        ),
        world_model=ActionConditionedWorldModelConfig(
            observation_dim=2,
            n_actions=2,
            hidden_sizes=(),
            step_size=0.02,
            sparsity=0.0,
            use_layer_norm=False,
        ),
        n_dreams_per_step=0,
        auto_curate_every=0,
    )
    search = CalibratedExtendedSearchControlConfig(
        mode=SEARCH_MODE_COMBINED,
        observation_dim=2,
        anchor_capacity=4,
        n_primitive_actions=2,
        n_options=1,
        backup_budget=1,
        calibration_evidence_floor=2,
        model_support_floor=1,
        confidence_scale=1.0,
        support_prior=1.0,
        model_error_scale=10.0,
        backup_step_size=0.1,
        max_observations=32,
    )
    return PrototypeSTOMPCalibratedSearchConfig(
        prototype=prototype,
        search=search,
        enabled=enabled,
    )


def _agent(*, enabled: bool = True) -> PrototypeSTOMPCalibratedDispatchAgent:
    return PrototypeSTOMPCalibratedDispatchAgent(
        PrototypeSTOMPCalibratedDispatchConfig(
            sidecar=_sidecar_config(),
            enabled=enabled,
        )
    )


def _state(
    agent: PrototypeSTOMPCalibratedDispatchAgent,
    *,
    selected_extended_action: int = 0,
) -> PrototypeSTOMPCalibratedDispatchState:
    state = agent.init(
        jr.key(7),
        anchor_bank=ANCHORS,
        anchor_active=ACTIVE,
        source_digest=SOURCE,
        representation_generation=3,
        lifecycle_id=jnp.asarray((17, 29), dtype=jnp.uint32),
    )
    prototype = state.sidecar.prototype
    oak = cast(OaKState, prototype.oak_state)
    stomp = oak.stomp_state
    learner = stomp.base_learner_state
    biases = tuple(
        jnp.full_like(
            bias,
            20.0 if index == selected_extended_action else -20.0,
        )
        for index, bias in enumerate(learner.head_params.biases)
    )
    weights = tuple(jnp.zeros_like(weight) for weight in learner.head_params.weights)
    learner = learner.replace(
        head_params=learner.head_params.replace(weights=weights, biases=biases)
    )
    prototype = cast(
        PrototypeAgentState,
        prototype.replace(
            oak_state=oak.replace(stomp_state=stomp.replace(base_learner_state=learner))
        ),
    )
    rebound = agent.sidecar.rebind(
        state.sidecar,
        prototype_state=prototype,
        source_digest=SOURCE,
        representation_generation=3,
    )
    assert bool(rebound.transaction_applied)
    wrapped = agent._with_checksum(state.replace(sidecar=rebound.state))
    assert bool(agent.validate_state(wrapped))
    return wrapped


def _transition(
    state: PrototypeSTOMPCalibratedDispatchState,
    next_observation: jax.Array,
) -> PrototypeTransition:
    prototype = state.sidecar.prototype
    return PrototypeTransition(
        observation=prototype.current_raw_observation,
        action=prototype.current_action,
        decision_id=prototype.current_decision_id,
        reward=jnp.asarray(1.0, dtype=jnp.float32),
        discount=jnp.asarray(1.0, dtype=jnp.float32),
        terminated=jnp.asarray(False, dtype=jnp.bool_),
        truncated=jnp.asarray(False, dtype=jnp.bool_),
        next_observation=next_observation,
        next_decision_observation=next_observation,
    )


def _start_withheld(
    agent: PrototypeSTOMPCalibratedDispatchAgent,
    *,
    anchor_index: int = 0,
) -> PrototypeSTOMPCalibratedDispatchState:
    result = agent.start(
        _state(agent),
        ANCHORS[anchor_index],
        safety_action_mask=NONE_SAFE,
    )
    assert bool(result.diagnostics.transaction_committed)
    assert int(result.decision.action) == -1
    assert not bool(result.state.sidecar.adapter_pending)
    assert not bool(result.state.sidecar.search.pending)
    return result.state


def _seed_evidence(
    agent: PrototypeSTOMPCalibratedDispatchAgent,
    state: PrototypeSTOMPCalibratedDispatchState,
    *,
    anchor_index: int,
    eligible_extended_actions: tuple[int, ...],
    q_row: tuple[float, float, float],
) -> PrototypeSTOMPCalibratedDispatchState:
    cfg = agent.config.sidecar.search
    search = state.sidecar.search
    targets = search.last_target_available
    value_counts = search.value_change_counts
    error_counts = search.model_error_counts
    support_counts = search.support_counts
    value_means = search.value_change_means
    error_means = search.model_error_means
    for extended_action in eligible_extended_actions:
        flat = extended_action * cfg.anchor_capacity + anchor_index
        targets = targets.at[flat].set(True)
        value_counts = value_counts.at[flat].set(cfg.calibration_evidence_floor)
        error_counts = error_counts.at[flat].set(cfg.calibration_evidence_floor)
        support_counts = support_counts.at[flat].set(cfg.model_support_floor)
        value_means = value_means.at[flat].set(jnp.float32(1.0))
        error_means = error_means.at[flat].set(jnp.float32(0.0))
    search = search.replace(
        q_values=search.q_values.at[anchor_index].set(jnp.asarray(q_row, dtype=jnp.float32)),
        has_last_decision=jnp.asarray(True, dtype=jnp.bool_),
        last_target_available=targets,
        value_change_counts=value_counts,
        model_error_counts=error_counts,
        support_counts=support_counts,
        value_change_means=value_means,
        model_error_means=error_means,
        anchor_revisit_trials=search.anchor_revisit_trials.at[anchor_index].set(
            cfg.calibration_evidence_floor
        ),
        anchor_revisit_successes=search.anchor_revisit_successes.at[anchor_index].set(
            cfg.calibration_evidence_floor
        ),
    )
    search = search.replace(pending_cache_digest=agent.sidecar.controller._pending_checksum(search))
    sidecar = agent.sidecar._with_checksum(state.sidecar.replace(search=search))
    result = agent._with_checksum(state.replace(sidecar=sidecar))
    assert bool(agent.validate_state(result))
    return result


def _set_option_keyboard_action_one(
    agent: PrototypeSTOMPCalibratedDispatchAgent,
    state: PrototypeSTOMPCalibratedDispatchState,
) -> PrototypeSTOMPCalibratedDispatchState:
    prototype = state.sidecar.prototype
    oak = cast(OaKState, prototype.oak_state)
    stomp = oak.stomp_state
    weights = stomp.option_policies.q_weights.at[0].set(
        jnp.asarray(((0.0, 0.0), (5.0, 0.0)), dtype=jnp.float32)
    )
    prototype = cast(
        PrototypeAgentState,
        prototype.replace(
            oak_state=oak.replace(
                stomp_state=stomp.replace(
                    option_policies=stomp.option_policies.replace(q_weights=weights)
                )
            )
        ),
    )
    sidecar = agent.sidecar._with_checksum(state.sidecar.replace(prototype=prototype))
    result = agent._with_checksum(state.replace(sidecar=sidecar))
    assert bool(agent.validate_state(result))
    return result


def _tree_nbytes(tree: object) -> int:
    return sum(
        int(jnp.asarray(leaf).size) * int(jnp.asarray(leaf).dtype.itemsize)
        for leaf in jax.tree_util.tree_leaves(tree)
    )


def test_config_is_separate_default_off_l0_and_strictly_round_trips() -> None:
    config = PrototypeSTOMPCalibratedDispatchConfig(sidecar=_sidecar_config())
    assert not config.enabled
    assert PrototypeSTOMPCalibratedDispatchConfig.from_config(config.to_config()) == config
    assert config.to_config()["proposal_unavailable_fallback"] == (
        "independently_safe_current_owner_counterfactual_only"
    )
    assert config.to_config()["proposal_available_distinct_from_dispatch_authorized"]
    assert PROTOTYPE_STOMP_CALIBRATED_DISPATCH_EVIDENCE_LEVEL == "L0"
    assert PROTOTYPE_STOMP_CALIBRATED_DISPATCH_ASSESSMENT == "not_assessed"
    assert not PROTOTYPE_STOMP_CALIBRATED_DISPATCH_SCIENTIFIC_PROMOTION_ALLOWED
    assert alberta.PrototypeSTOMPCalibratedDispatchAgent is (PrototypeSTOMPCalibratedDispatchAgent)
    malformed = config.to_config()
    malformed["enabled"] = 1
    with pytest.raises(ValueError, match="exact bool"):
        PrototypeSTOMPCalibratedDispatchConfig.from_config(malformed)


def test_disabled_v2_is_exact_v1_start_update_parity_and_ignores_mask() -> None:
    sidecar_config = _sidecar_config()
    v1 = PrototypeSTOMPCalibratedSearchAgent(sidecar_config)
    v2 = PrototypeSTOMPCalibratedDispatchAgent(
        PrototypeSTOMPCalibratedDispatchConfig(sidecar=sidecar_config, enabled=False)
    )
    lifecycle_id = jnp.asarray((17, 29), dtype=jnp.uint32)
    v1_state = v1.init(
        jr.key(13),
        anchor_bank=ANCHORS,
        anchor_active=ACTIVE,
        source_digest=SOURCE,
        representation_generation=3,
        lifecycle_id=lifecycle_id,
    )
    v2_state = v2.init(
        jr.key(13),
        anchor_bank=ANCHORS,
        anchor_active=ACTIVE,
        source_digest=SOURCE,
        representation_generation=3,
        lifecycle_id=lifecycle_id,
    )
    chex.assert_trees_all_equal(v1_state, v2_state.sidecar)
    v1_start = v1.start(v1_state, ANCHORS[0])
    v2_start = v2.start(v2_state, ANCHORS[0], safety_action_mask=NONE_SAFE)
    chex.assert_trees_all_equal(v1_start.state, v2_start.state.sidecar)
    chex.assert_trees_all_equal(v1_start.decision, v2_start.decision)
    transition = _transition(v2_start.state, ANCHORS[1])
    v1_update = v1.update_transition(v1_start.state, transition)
    v2_update = v2.update_transition(
        v2_start.state,
        transition,
        safety_action_mask=NONE_SAFE,
    )
    chex.assert_trees_all_equal(v1_update.state, v2_update.state.sidecar)
    chex.assert_trees_all_equal(v1_update.decision, v2_update.decision)
    np.testing.assert_array_equal(v2_update.state.policy_call_count_words, 0)


def test_no_history_safe_start_dispatches_current_owner_but_all_false_withholds() -> None:
    agent = _agent()
    safe = agent.start(_state(agent), ANCHORS[0], safety_action_mask=ALL_SAFE)
    assert not bool(safe.proposal.available)
    assert bool(safe.diagnostics.dispatch_authorized)
    assert int(safe.decision.action) == 0
    assert bool(safe.state.sidecar.adapter_pending)

    withheld = agent.start(_state(agent), ANCHORS[0], safety_action_mask=NONE_SAFE)
    assert not bool(withheld.proposal.available)
    assert not bool(withheld.diagnostics.dispatch_authorized)
    assert int(withheld.decision.action) == -1
    assert int(agent.decision(withheld.state).action) == -1
    assert not bool(withheld.state.sidecar.adapter_pending)


def test_misbound_candidate_arm_rolls_back_start_and_applied_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _agent()
    initial = _state(agent)
    monkeypatch.setattr(
        agent,
        "_arm_binding_valid",
        lambda *_args, **_kwargs: jnp.asarray(False, dtype=jnp.bool_),
    )
    rejected = agent.start(initial, ANCHORS[0], safety_action_mask=ALL_SAFE)
    chex.assert_trees_all_equal(rejected.state, initial)
    assert not bool(rejected.diagnostics.transaction_committed)
    assert not bool(rejected.diagnostics.prototype_learning_applied)
    assert not bool(rejected.diagnostics.next_arm_bound_after_dispatch)
    assert int(rejected.decision.action) == -1
    assert not bool(rejected.decision.armed)


def test_candidate_gate_is_anchor_and_head_specific_and_skips_higher_unready_q() -> None:
    agent = _agent()
    at_other_anchor = _seed_evidence(
        agent,
        _start_withheld(agent, anchor_index=1),
        anchor_index=0,
        eligible_extended_actions=(1,),
        q_row=(0.0, 10.0, 100.0),
    )
    other = agent.retry_dispatch(at_other_anchor, safety_action_mask=ALL_SAFE)
    assert not bool(other.proposal.available)
    assert not bool(jnp.any(other.proposal.candidate_eligible))
    assert bool(other.diagnostics.dispatch_authorized)

    only_zero_ready = _seed_evidence(
        agent,
        _start_withheld(agent),
        anchor_index=0,
        eligible_extended_actions=(0,),
        q_row=(1.0, 100.0, 200.0),
    )
    selected = agent.retry_dispatch(only_zero_ready, safety_action_mask=ALL_SAFE)
    assert bool(selected.proposal.available)
    np.testing.assert_array_equal(
        selected.proposal.candidate_eligible,
        jnp.asarray((True, False, False), dtype=jnp.bool_),
    )
    assert int(selected.proposal.planned_extended_action) == 0
    assert int(selected.diagnostics.selected_candidate_flat_index) == 0
    assert int(selected.diagnostics.selected_candidate_value_change_count) == 2
    assert int(selected.diagnostics.selected_candidate_support_count) == 1


def test_primitive_proposal_replaces_one_owner_cache_then_arms_effective_action() -> None:
    agent = _agent()
    state = _seed_evidence(
        agent,
        _start_withheld(agent),
        anchor_index=0,
        eligible_extended_actions=(1,),
        q_row=(-1.0, 8.0, 100.0),
    )
    before = cast(OaKState, state.sidecar.prototype.oak_state)
    result = agent.retry_dispatch(state, safety_action_mask=ALL_SAFE)
    after = cast(OaKState, result.state.sidecar.prototype.oak_state)

    assert bool(result.proposal.available)
    assert int(result.proposal.planned_kind) == CANDIDATE_KIND_PRIMITIVE
    assert int(result.proposal.proposed_primitive_action) == 1
    assert int(result.decision.action) == 1
    assert bool(result.diagnostics.action_changed)
    assert int(result.diagnostics.actual_credit_owner) == DISPATCH_OWNER_BASE_PRIMITIVE
    assert int(result.state.sidecar.search.pending_executed_kind) == CANDIDATE_KIND_PRIMITIVE
    assert int(result.state.sidecar.search.pending_executed_index) == 1
    np.testing.assert_array_equal(before.step_words, after.step_words)
    np.testing.assert_array_equal(before.stomp_state.step_words, after.stomp_state.step_words)
    np.testing.assert_array_equal(
        state.sidecar.prototype.step_words,
        result.state.sidecar.prototype.step_words,
    )


def test_option_head_is_pure_keyboard_proposal_not_option_start_or_double_update() -> None:
    agent = _agent()
    state = _set_option_keyboard_action_one(
        agent,
        _seed_evidence(
            agent,
            _start_withheld(agent),
            anchor_index=0,
            eligible_extended_actions=(2,),
            q_row=(-5.0, -4.0, 9.0),
        ),
    )
    before = cast(OaKState, state.sidecar.prototype.oak_state)
    result = agent.retry_dispatch(state, safety_action_mask=ALL_SAFE)
    after = cast(OaKState, result.state.sidecar.prototype.oak_state)

    assert int(result.proposal.planned_kind) == CANDIDATE_KIND_OPTION
    assert int(result.proposal.planned_option_index) == 0
    assert bool(result.proposal.keyboard_used)
    np.testing.assert_array_equal(
        result.proposal.keyboard_vector,
        jnp.asarray((1.0,), dtype=jnp.float32),
    )
    assert int(result.proposal.proposed_primitive_action) == 1
    assert int(result.decision.action) == 1
    assert not bool(result.diagnostics.planned_option_started_by_dispatch)
    assert int(result.diagnostics.actual_credit_owner) == DISPATCH_OWNER_BASE_PRIMITIVE
    assert int(result.diagnostics.actual_executing_option) == -1
    assert int(after.stomp_state.executing_option) == -1
    assert int(result.state.sidecar.search.pending_executed_kind) == CANDIDATE_KIND_PRIMITIVE
    assert int(result.state.sidecar.search.pending_executed_index) == 1
    np.testing.assert_array_equal(before.step_words, after.step_words)
    np.testing.assert_array_equal(before.execution_counts, after.execution_counts)
    np.testing.assert_array_equal(before.stomp_state.step_words, after.stomp_state.step_words)
    assert int(before.stomp_state.option_steps) == int(after.stomp_state.option_steps)


def test_active_option_replacement_preserves_option_owner_and_arms_option_credit() -> None:
    agent = _agent()
    withheld = agent.start(
        _state(agent, selected_extended_action=2),
        ANCHORS[0],
        safety_action_mask=NONE_SAFE,
    )
    assert int(withheld.decision.action) == -1
    before = cast(OaKState, withheld.state.sidecar.prototype.oak_state)
    assert int(before.stomp_state.executing_option) == 0
    previous_action = int(before.stomp_state.option_last_intra_action)
    proposed_action = 1 - previous_action
    q_row = [-10.0, -10.0, -10.0]
    q_row[proposed_action] = 8.0
    state = _seed_evidence(
        agent,
        withheld.state,
        anchor_index=0,
        eligible_extended_actions=(proposed_action,),
        q_row=cast(tuple[float, float, float], tuple(q_row)),
    )
    result = agent.retry_dispatch(state, safety_action_mask=ALL_SAFE)
    after = cast(OaKState, result.state.sidecar.prototype.oak_state)

    assert int(result.proposal.planned_kind) == CANDIDATE_KIND_PRIMITIVE
    assert int(result.proposal.proposed_primitive_action) == proposed_action
    assert int(result.diagnostics.actual_credit_owner) == DISPATCH_OWNER_OPTION
    assert int(result.diagnostics.actual_executing_option) == 0
    assert int(after.stomp_state.executing_option) == 0
    assert int(after.stomp_state.option_last_intra_action) == proposed_action
    assert int(after.stomp_state.last_primitive_action) == proposed_action
    assert int(after.stomp_state.base_last_action) == int(before.stomp_state.base_last_action)
    assert int(after.stomp_state.option_steps) == int(before.stomp_state.option_steps)
    chex.assert_trees_all_equal(after.stomp_state.option_models, before.stomp_state.option_models)
    np.testing.assert_array_equal(after.step_words, before.step_words)
    np.testing.assert_array_equal(after.stomp_state.step_words, before.stomp_state.step_words)
    assert int(result.state.sidecar.search.pending_executed_kind) == CANDIDATE_KIND_OPTION
    assert int(result.state.sidecar.search.pending_executed_index) == 0


def test_hard_mask_uses_safe_current_owner_fallback_and_never_claims_proposal_dispatch() -> None:
    agent = _agent()
    state = _seed_evidence(
        agent,
        _start_withheld(agent),
        anchor_index=0,
        eligible_extended_actions=(1,),
        q_row=(-1.0, 8.0, -2.0),
    )
    result = agent.retry_dispatch(
        state,
        safety_action_mask=jnp.asarray((True, False), dtype=jnp.bool_),
    )
    assert bool(result.proposal.available)
    assert int(result.proposal.proposed_primitive_action) == 1
    assert bool(result.diagnostics.used_safe_current_owner_fallback)
    assert bool(result.diagnostics.dispatch_authorized)
    assert not bool(result.diagnostics.action_changed)
    assert int(result.decision.action) == 0
    assert int(result.state.sidecar.search.pending_executed_index) == 0


def test_no_action_rejects_forged_learning_then_safe_retry_has_no_learning() -> None:
    agent = _agent()
    state = _start_withheld(agent)
    step_words = np.asarray(state.sidecar.prototype.step_words)
    forged = _transition(state, ANCHORS[1])
    rejected = agent.update_transition(
        state,
        forged,
        safety_action_mask=ALL_SAFE,
    )
    chex.assert_trees_all_equal(rejected.state, state)
    assert not bool(rejected.diagnostics.transaction_committed)
    assert not bool(rejected.diagnostics.prototype_learning_applied)
    assert int(rejected.decision.action) == -1

    retry = agent.retry_dispatch(state, safety_action_mask=ALL_SAFE)
    assert bool(retry.diagnostics.transaction_committed)
    assert bool(retry.diagnostics.dispatch_authorized)
    assert not bool(retry.diagnostics.prototype_learning_applied)
    assert int(retry.decision.action) == 0
    np.testing.assert_array_equal(retry.state.sidecar.prototype.step_words, step_words)
    accepted = agent.update_transition(
        retry.state,
        _transition(retry.state, ANCHORS[1]),
        safety_action_mask=ALL_SAFE,
    )
    assert bool(accepted.diagnostics.transaction_committed)
    assert bool(accepted.diagnostics.prototype_learning_applied)


def test_current_decision_and_checkpoint_keep_no_action_withheld_under_jit() -> None:
    agent = _agent()
    initial = _state(agent)
    eager = agent.start(initial, ANCHORS[0], safety_action_mask=NONE_SAFE)
    compiled = jax.jit(
        lambda value, mask: agent.start(
            value,
            ANCHORS[0],
            safety_action_mask=mask,
        )
    )(initial, NONE_SAFE)
    chex.assert_trees_all_equal(eager.state, compiled.state)
    assert int(jax.jit(agent.decision)(compiled.state).action) == -1

    payload = agent.checkpoint_payload(compiled.state)
    restored = agent.restore_checkpoint(
        payload,
        source_digest=SOURCE,
        representation_generation=3,
    )
    assert int(agent.decision(restored).action) == -1
    tampered = dict(payload)
    digest = cast(jax.Array, payload["state_sha256"])
    tampered["state_sha256"] = digest.at[0].set(digest[0] ^ jnp.uint8(1))
    with pytest.raises(ValueError, match="SHA"):
        agent.restore_checkpoint(
            tampered,
            source_digest=SOURCE,
            representation_generation=3,
        )
    with pytest.raises(ValueError, match="stale"):
        agent.restore_checkpoint(
            payload,
            source_digest=SOURCE + jnp.uint32(1),
            representation_generation=3,
        )


def test_reachable_clock_boundaries_block_unrecorded_fallback_and_dispatch() -> None:
    # Commit-only exhaustion is unreachable: the state invariant requires
    # dispatch commits <= policy calls and both clocks have equal uint64 capacity.
    agent = _agent()
    no_history = _start_withheld(agent)
    no_history = agent._with_checksum(no_history.replace(policy_call_count_words=_MAX_WORDS))
    blocked_fallback = agent.retry_dispatch(no_history, safety_action_mask=ALL_SAFE)
    assert not bool(blocked_fallback.proposal.available)
    assert not bool(blocked_fallback.diagnostics.dispatch_authorized)
    assert int(blocked_fallback.decision.action) == -1
    assert bool(blocked_fallback.state.policy_unavailable)
    assert int(blocked_fallback.state.policy_error) == (
        PROTOTYPE_STOMP_CALIBRATED_DISPATCH_ERROR_CLOCK_EXHAUSTED
    )

    eligible = _seed_evidence(
        agent,
        _start_withheld(agent),
        anchor_index=0,
        eligible_extended_actions=(1,),
        q_row=(0.0, 5.0, 0.0),
    )
    eligible = agent._with_checksum(
        eligible.replace(
            policy_call_count_words=_MAX_WORDS,
            dispatch_commit_count_words=_MAX_WORDS,
        )
    )
    blocked_proposal = agent.retry_dispatch(eligible, safety_action_mask=ALL_SAFE)
    assert bool(blocked_proposal.proposal.available)
    assert not bool(blocked_proposal.diagnostics.replacement_committed)
    assert not bool(blocked_proposal.diagnostics.next_arm_applied)
    assert int(blocked_proposal.decision.action) == -1
    assert bool(blocked_proposal.state.policy_unavailable)


def test_stale_record_and_corrupt_wrapper_cannot_authorize_transition() -> None:
    agent = _agent()
    safe = agent.start(_state(agent), ANCHORS[0], safety_action_mask=ALL_SAFE)
    stale = agent._with_checksum(
        safe.state.replace(last_decision_id=safe.state.last_decision_id.at[3].add(jnp.uint32(1)))
    )
    assert bool(agent.validate_state(stale))
    assert int(agent.decision(stale).action) == -1
    rejected = agent.update_transition(
        stale,
        _transition(stale, ANCHORS[1]),
        safety_action_mask=ALL_SAFE,
    )
    chex.assert_trees_all_equal(rejected.state, stale)

    corrupt = safe.state.replace(last_effective_primitive_action=jnp.int32(1))
    assert not bool(agent.validate_state(corrupt))
    rejected_corrupt = agent.update_transition(
        corrupt,
        _transition(corrupt, ANCHORS[1]),
        safety_action_mask=ALL_SAFE,
    )
    chex.assert_trees_all_equal(rejected_corrupt.state, corrupt)


def test_eager_jit_update_and_one_step_scan_are_identical() -> None:
    agent = _agent()
    started = agent.start(_state(agent), ANCHORS[0], safety_action_mask=ALL_SAFE)
    transition = _transition(started.state, ANCHORS[1])
    eager = agent.update_transition(
        started.state,
        transition,
        safety_action_mask=ALL_SAFE,
    )
    compiled = jax.jit(
        lambda state, item, mask: agent.update_transition(
            state,
            item,
            safety_action_mask=mask,
        )
    )(started.state, transition, ALL_SAFE)
    chex.assert_trees_all_equal(eager.state, compiled.state)
    chex.assert_trees_all_equal(eager.decision, compiled.decision)

    stacked_transition = jax.tree_util.tree_map(lambda value: value[None], transition)
    scan = agent.scan_transitions(
        started.state,
        stacked_transition,
        ALL_SAFE[None, :],
    )
    chex.assert_trees_all_equal(scan.state, eager.state)
    assert int(scan.actions[0]) == int(eager.decision.action)


def test_resource_budget_is_exact_bounded_and_nonpromoting() -> None:
    agent = _agent()
    state = _state(agent)
    budget = agent.resource_budget(state)
    assert budget.sidecar_persistent_state_nbytes == _tree_nbytes(state.sidecar)
    assert budget.total_persistent_state_nbytes == _tree_nbytes(state)
    assert budget.dispatch_binding_nbytes == (
        budget.total_persistent_state_nbytes - budget.sidecar_persistent_state_nbytes
    )
    assert budget.max_keyboard_proposals_per_policy_call == 1
    assert budget.max_cached_action_replacements_per_policy_call == 1
    assert budget.max_next_arm_calls_per_decision == 1
    assert budget.additional_rng_draws_per_policy_call == 0
    assert budget.additional_backward_passes_per_policy_call == 0
    assert budget.additional_model_updates_per_policy_call == 0
    assert budget.persistent_state_growth_per_transition_bytes == 0
    assert budget.proposal_unavailability_can_dispatch_safe_current_owner
    assert budget.proposal_available_distinct_from_dispatch_authorized
    assert not budget.planned_option_starts_option
    assert not budget.scientific_promotion_allowed
