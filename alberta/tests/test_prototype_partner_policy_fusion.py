# mypy: disable-error-code="call-arg"
"""Transactional PrototypeAgent integration for bounded partner policy fusion."""

# mypy: disable-error-code="attr-defined"

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework as alberta
import alberta_framework.core as core
from alberta_framework.core.intelligence_amplification import (
    ExoCerebellumConfig,
    IAConfig,
)
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.partner_policy_fusion import (
    ROUTE_ACCEPT,
    ROUTE_BLEND,
    SOURCE_OPTION_KEYBOARD,
    PartnerMessageBatch,
    PartnerPolicyFusionConfig,
    PartnerPolicyFusionFeedback,
)
from alberta_framework.core.prototype_agent import (
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeInteractionState,
    PrototypePartnerPolicyFusionFeedback,
    PrototypePartnerPolicyFusionInput,
    PrototypeRecurrentLatentWorldModelState,
    PrototypeTransition,
    load_prototype_checkpoint,
    save_prototype_checkpoint,
)
from alberta_framework.core.recurrent_latent_world_model_ensemble import (
    RecurrentLatentWorldModelEnsembleConfig,
)

pytestmark = pytest.mark.unit


OBSERVATION_DIM = 2
N_ACTIONS = 2


def _oak() -> OaKConfig:
    return OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=(
                SubtaskSpec(
                    feature_index=0,
                    threshold=1.0e6,
                    max_option_steps=8,
                ),
            ),
            observation_dim=OBSERVATION_DIM,
            n_primitive_actions=N_ACTIONS,
            base_hidden_sizes=(),
            epsilon_base=0.0,
            epsilon_option=0.0,
        )
    )


def _fusion_config(**overrides: Any) -> PartnerPolicyFusionConfig:
    values: dict[str, Any] = {
        "max_partners": 2,
        "context_dim": OBSERVATION_DIM,
        "n_actions": N_ACTIONS,
        "max_abs_context": 10.0,
        "assistance_value_bound": 10.0,
    }
    values.update(overrides)
    return PartnerPolicyFusionConfig(**values)


def _agent(
    *,
    ia: bool = False,
    counter_cap: int = 100,
    fusion_overrides: dict[str, Any] | None = None,
    recurrent: bool = False,
) -> PrototypeAgent:
    ia_config = None
    if ia:
        ia_config = IAConfig(
            cerebellum=ExoCerebellumConfig(
                n_demons=1,
                obs_dim=OBSERVATION_DIM,
                step_size=0.05,
            ),
            cortex=_oak(),
        )
    fusion_values = {} if fusion_overrides is None else fusion_overrides
    return PrototypeAgent(
        PrototypeAgentConfig(
            oak=_oak(),
            ia=ia_config,
            partner_policy_fusion=_fusion_config(
                counter_cap=counter_cap,
                **fusion_values,
            ),
            recurrent_latent_world_model_ensemble=(
                RecurrentLatentWorldModelEnsembleConfig(
                    observation_dim=OBSERVATION_DIM,
                    n_actions=N_ACTIONS,
                    latent_dim=2,
                    ensemble_size=2,
                    learning_rate=0.01,
                    bootstrap_probability=0.7,
                    uncertainty_warmup_steps=1,
                    initialization_scale=0.1,
                    max_updates=100,
                )
                if recurrent
                else None
            ),
        )
    )


def _transition(
    state: Any,
    next_observation: tuple[float, float] = (0.25, -0.5),
    *,
    reward: float = 0.5,
) -> PrototypeTransition:
    observation = jnp.asarray(next_observation, dtype=jnp.float32)
    return PrototypeTransition(
        observation=state.current_raw_observation,
        action=state.current_action,
        decision_id=state.current_decision_id,
        reward=jnp.asarray(reward, dtype=jnp.float32),
        discount=jnp.asarray(1.0, dtype=jnp.float32),
        terminated=jnp.asarray(False, dtype=jnp.bool_),
        truncated=jnp.asarray(False, dtype=jnp.bool_),
        next_observation=observation,
        next_decision_observation=observation,
    )


def _messages(
    agent: PrototypeAgent,
    state: Any,
    *,
    suggested_action: int,
    safe_horizon: int = 1,
    event_increment: int = 1,
    declared_confidence: float = 1.0,
    communication_cost: float = 0.0,
) -> PartnerMessageBatch:
    fusion = agent.partner_policy_fusion
    assert fusion is not None
    batch = fusion.empty_messages()
    telemetry_maximum = int(np.iinfo(np.int32).max)
    next_decision_id = min(int(state.step_count) + 1, telemetry_maximum)
    next_event_id = min(
        int(state.observation_event_count) + event_increment,
        telemetry_maximum,
    )
    next_decision_words = _add_identity_words(state.step_words, 1)
    next_event_words = _add_identity_words(
        state.observation_event_words,
        event_increment,
    )
    valid_through_event_words = _add_identity_words(
        next_event_words,
        safe_horizon,
    )
    return cast(
        PartnerMessageBatch,
        batch.replace(
            available=batch.available.at[0].set(True),
            partner_id=batch.partner_id.at[0].set(0),
            observation_id=batch.observation_id.at[0].set(101),
            context_id=batch.context_id.at[0].set(201),
            suggested_action=batch.suggested_action.at[0].set(suggested_action),
            declared_confidence=batch.declared_confidence.at[0].set(
                declared_confidence
            ),
            rationale_reference=batch.rationale_reference.at[0].set(301),
            provenance_reference=batch.provenance_reference.at[0].set(401),
            communication_cost=batch.communication_cost.at[0].set(
                communication_cost
            ),
            issued_decision_id=batch.issued_decision_id.at[0].set(next_decision_id),
            issued_event_id=batch.issued_event_id.at[0].set(next_event_id),
            valid_through_event_id=batch.valid_through_event_id.at[0].set(
                min(next_event_id + safe_horizon, telemetry_maximum)
            ),
            issued_decision_words=batch.issued_decision_words.at[0].set(
                next_decision_words
            ),
            issued_event_words=batch.issued_event_words.at[0].set(
                next_event_words
            ),
            valid_through_event_words=(
                batch.valid_through_event_words.at[0].set(
                    valid_through_event_words
                )
            ),
        ),
    )


def _add_identity_words(words: Any, delta: int) -> Any:
    """Advance one concrete big-endian uint64 test identity."""

    materialized = np.asarray(words, dtype=np.uint32)
    value = (int(materialized[0]) << 32) | int(materialized[1])
    advanced = value + delta
    assert 0 <= advanced <= np.iinfo(np.uint64).max
    return jnp.asarray(
        (advanced >> 32, advanced & np.iinfo(np.uint32).max),
        dtype=jnp.uint32,
    )


def _sidecar(
    agent: PrototypeAgent,
    state: Any,
    *,
    suggested_action: int,
    mask: tuple[bool, bool] = (True, True),
    available: bool = True,
    context: tuple[float, float] = (0.25, -0.5),
    event_increment: int = 1,
    declared_confidence: float = 1.0,
    communication_cost: float = 0.0,
    keyboard_available: bool = False,
    keyboard_vector: tuple[float, ...] = (0.0,),
    prototype_decision_id: Any = None,
) -> PrototypePartnerPolicyFusionInput:
    expected_prototype_decision_id = state.current_decision_id.at[3].set(
        state.current_decision_id[3] + jnp.asarray(1, dtype=jnp.uint32)
    )
    bound_prototype_decision_id = (
        expected_prototype_decision_id
        if prototype_decision_id is None
        else prototype_decision_id
    )
    return PrototypePartnerPolicyFusionInput(
        available=jnp.asarray(available, dtype=jnp.bool_),
        prototype_decision_id=bound_prototype_decision_id,
        observation_id=jnp.asarray(101, dtype=jnp.int32),
        context_id=jnp.asarray(201, dtype=jnp.int32),
        context_features=jnp.asarray(context, dtype=jnp.float32),
        safety_action_mask=jnp.asarray(mask, dtype=jnp.bool_),
        keyboard_available=jnp.asarray(keyboard_available, dtype=jnp.bool_),
        keyboard_vector=jnp.asarray(keyboard_vector, dtype=jnp.float32),
        messages=_messages(
            agent,
            state,
            suggested_action=suggested_action,
            event_increment=event_increment,
            declared_confidence=declared_confidence,
            communication_cost=communication_cost,
        ),
    )


def _feedback(
    decision: Any,
    prototype_decision_id: Any,
    *,
    available: bool = True,
    assistance: float = 10.0,
    safe: bool = True,
) -> PrototypePartnerPolicyFusionFeedback:
    return PrototypePartnerPolicyFusionFeedback(
        prototype_decision_id=prototype_decision_id,
        feedback=PartnerPolicyFusionFeedback(
            available=jnp.asarray(available, dtype=jnp.bool_),
            decision_id=decision.decision_id,
            executed_event_id=decision.event_id,
            decision_words=decision.decision_words,
            executed_event_words=decision.event_words,
            executed_action=decision.effective_action,
            partner_id=decision.selected_partner_id,
            assistance_value_available=jnp.asarray(True, dtype=jnp.bool_),
            realized_assistance_value=jnp.asarray(assistance, dtype=jnp.float32),
            safety_outcome_available=jnp.asarray(True, dtype=jnp.bool_),
            safety_outcome_ok=jnp.asarray(safe, dtype=jnp.bool_),
        ),
    )


def _assert_tree_equal(left: object, right: object) -> None:
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    assert str(left_tree) == str(right_tree)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def _materialize_keys(tree: object) -> object:
    def convert(value: object) -> object:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jnp.issubdtype(dtype, jax.dtypes.prng_key):
            return jr.key_data(value)  # type: ignore[arg-type]
        return value

    return jax.tree.map(convert, tree)


def test_config_roundtrip_dimensions_and_default_shapes_are_compatible() -> None:
    default = PrototypeAgentConfig()
    default_payload = default.to_config()
    assert "partner_policy_fusion" not in default_payload
    assert PrototypeAgentConfig.from_config(default_payload).to_config() == default_payload
    assert PrototypeAgent(default).init(jr.key(0)).ia_state is None

    config = _agent().config
    restored = PrototypeAgentConfig.from_config(config.to_config())
    assert restored.to_config() == config.to_config()
    assert restored.partner_policy_fusion == config.partner_policy_fusion
    with pytest.raises(ValueError, match="context_dim"):
        PrototypeAgentConfig(
            oak=_oak(),
            partner_policy_fusion=_fusion_config(context_dim=3),
        )
    with pytest.raises(ValueError, match="n_actions"):
        PrototypeAgentConfig(
            oak=_oak(),
            partner_policy_fusion=_fusion_config(n_actions=3),
        )


def test_integration_types_are_identity_exported() -> None:
    assert alberta.PrototypeInteractionState is PrototypeInteractionState
    assert core.PrototypeInteractionState is PrototypeInteractionState
    assert (
        alberta.PrototypePartnerPolicyFusionInput
        is PrototypePartnerPolicyFusionInput
    )
    assert (
        core.PrototypePartnerPolicyFusionInput
        is PrototypePartnerPolicyFusionInput
    )
    assert (
        alberta.PrototypePartnerPolicyFusionFeedback
        is PrototypePartnerPolicyFusionFeedback
    )
    assert (
        core.PrototypePartnerPolicyFusionFeedback
        is PrototypePartnerPolicyFusionFeedback
    )
    assert (
        alberta.PrototypePartnerPolicyFusionDiagnostics
        is core.PrototypePartnerPolicyFusionDiagnostics
    )


def test_first_partner_opportunity_changes_the_next_executed_action() -> None:
    agent = _agent()
    initial = agent.start(agent.init(jr.key(7)), jnp.zeros((OBSERVATION_DIM,)))
    first_action = int(initial.current_action)
    proposed = 1 - first_action

    first = agent.update_transition(
        initial,
        _transition(initial),
        partner_policy_fusion_input=_sidecar(
            agent,
            initial,
            suggested_action=proposed,
        ),
    )
    fusion = first.partner_policy_fusion_diagnostics
    assert fusion is not None
    assert bool(first.transition_diagnostics.valid)
    assert int(fusion.decision.route) == ROUTE_ACCEPT
    assert bool(fusion.decision.partner_influenced)
    assert int(fusion.counterfactual_base_action) != int(fusion.effective_action)
    assert int(first.action) == proposed
    assert int(first.state.current_action) == proposed
    assert int(first.state.oak_state.stomp_state.last_primitive_action) == proposed
    assert int(initial.current_action) == first_action  # start itself remains base-only

    # The authoritative next transition must accept the exact fused dispatch.
    second_transition = _transition(first.state, (0.5, 0.1))
    second = agent.update_transition(
        first.state,
        second_transition,
        partner_policy_fusion_feedback=_feedback(
            fusion.decision,
            first.state.current_decision_id,
        ),
    )
    assert bool(second.transition_diagnostics.valid)
    second_fusion = second.partner_policy_fusion_diagnostics
    assert second_fusion is not None
    assert bool(second_fusion.feedback.applied)


def test_missing_invalid_and_unsafe_sidecars_fall_back_without_learning() -> None:
    agent = _agent()
    state = agent.start(agent.init(jr.key(3)), jnp.zeros((OBSERVATION_DIM,)))
    missing = agent.update_transition(state, _transition(state))
    missing_diag = missing.partner_policy_fusion_diagnostics
    assert missing_diag is not None
    assert int(missing.action) == int(missing_diag.counterfactual_base_action)
    assert not bool(missing_diag.decision_input_supplied)
    wrapper_before = cast(PrototypeInteractionState, state.ia_state)
    wrapper_missing = cast(PrototypeInteractionState, missing.state.ia_state)
    _assert_tree_equal(
        wrapper_missing.partner_policy_fusion_state,
        wrapper_before.partner_policy_fusion_state,
    )

    invalid = agent.update_transition(
        state,
        _transition(state),
        partner_policy_fusion_input=_sidecar(
            agent,
            state,
            suggested_action=1 - int(state.current_action),
            context=(float("nan"), 0.0),
        ),
    )
    invalid_diag = invalid.partner_policy_fusion_diagnostics
    assert invalid_diag is not None
    assert int(invalid.action) == int(invalid_diag.counterfactual_base_action)
    assert not bool(invalid_diag.decision.input_valid)

    base = int(state.current_action)
    unsafe_proposal = 1 - base
    mask = (True, False) if base == 0 else (False, True)
    shielded = agent.update_transition(
        state,
        _transition(state),
        partner_policy_fusion_input=_sidecar(
            agent,
            state,
            suggested_action=unsafe_proposal,
            mask=mask,
        ),
    )
    shielded_diag = shielded.partner_policy_fusion_diagnostics
    assert shielded_diag is not None
    assert int(shielded.action) == base
    assert not bool(shielded_diag.decision.partner_influenced)
    assert not bool(shielded_diag.decision.feedback_armed)


def test_feedback_is_exact_and_invalid_transition_rolls_back_partner_effects() -> None:
    agent = _agent()
    initial = agent.start(agent.init(jr.key(11)), jnp.zeros((OBSERVATION_DIM,)))
    first = agent.update_transition(
        initial,
        _transition(initial),
        partner_policy_fusion_input=_sidecar(
            agent,
            initial,
            suggested_action=1 - int(initial.current_action),
        ),
    )
    first_diag = first.partner_policy_fusion_diagnostics
    assert first_diag is not None
    armed_wrapper = cast(PrototypeInteractionState, first.state.ia_state)

    stale = cast(
        PrototypePartnerPolicyFusionFeedback,
        _feedback(
            first_diag.decision,
            first.state.current_decision_id,
        ).replace(
            feedback=_feedback(
                first_diag.decision,
                first.state.current_decision_id,
            ).feedback.replace(
                decision_id=first_diag.decision.decision_id + jnp.int32(1)
            )
        ),
    )
    stale_result = agent.update_transition(
        first.state,
        _transition(first.state, (0.4, -0.2)),
        partner_policy_fusion_feedback=stale,
    )
    stale_diag = stale_result.partner_policy_fusion_diagnostics
    assert stale_diag is not None
    assert not bool(stale_diag.feedback.applied)
    stale_wrapper = cast(PrototypeInteractionState, stale_result.state.ia_state)
    _assert_tree_equal(
        stale_wrapper.partner_policy_fusion_state,
        armed_wrapper.partner_policy_fusion_state,
    )

    invalid_transition = cast(
        PrototypeTransition,
        _transition(first.state, (0.4, -0.2)).replace(
            action=jnp.asarray(1 - int(first.state.current_action), dtype=jnp.int32)
        ),
    )
    rolled_back = agent.update_transition(
        first.state,
        invalid_transition,
        partner_policy_fusion_feedback=_feedback(
            first_diag.decision,
            first.state.current_decision_id,
        ),
    )
    assert not bool(rolled_back.transition_diagnostics.valid)
    _assert_tree_equal(
        _materialize_keys(rolled_back.state),
        _materialize_keys(first.state),
    )
    rollback_diag = rolled_back.partner_policy_fusion_diagnostics
    assert rollback_diag is not None
    assert not bool(rollback_diag.feedback.applied)


def test_duplicate_and_misattributed_feedback_are_exact_partner_noops() -> None:
    agent = _agent()
    initial = agent.start(
        agent.init(jr.key(17)),
        jnp.zeros((OBSERVATION_DIM,), dtype=jnp.float32),
    )
    transition = _transition(initial)
    base = agent.update_transition(initial, transition)
    armed = agent.update_transition(
        initial,
        transition,
        partner_policy_fusion_input=_sidecar(
            agent,
            initial,
            suggested_action=1 - int(base.action),
        ),
    )
    armed_diag = armed.partner_policy_fusion_diagnostics
    assert armed_diag is not None
    valid_feedback = _feedback(
        armed_diag.decision,
        armed.state.current_decision_id,
    )
    armed_interaction = cast(PrototypeInteractionState, armed.state.ia_state)

    wrong_partner = valid_feedback.replace(
        feedback=valid_feedback.feedback.replace(
            partner_id=jnp.asarray(1, dtype=jnp.int32)
        )
    )
    misattributed = agent.update_transition(
        armed.state,
        _transition(armed.state, (0.45, -0.1)),
        partner_policy_fusion_feedback=wrong_partner,
    )
    misattributed_diag = misattributed.partner_policy_fusion_diagnostics
    assert misattributed_diag is not None
    assert not bool(misattributed_diag.feedback.applied)
    misattributed_interaction = cast(
        PrototypeInteractionState,
        misattributed.state.ia_state,
    )
    _assert_tree_equal(
        misattributed_interaction.partner_policy_fusion_state,
        armed_interaction.partner_policy_fusion_state,
    )

    applied = agent.update_transition(
        armed.state,
        _transition(armed.state, (0.45, -0.1)),
        partner_policy_fusion_feedback=valid_feedback,
    )
    applied_diag = applied.partner_policy_fusion_diagnostics
    assert applied_diag is not None
    assert bool(applied_diag.feedback.applied)
    applied_interaction = cast(PrototypeInteractionState, applied.state.ia_state)
    duplicate = agent.update_transition(
        applied.state,
        _transition(applied.state, (0.2, 0.3)),
        partner_policy_fusion_feedback=valid_feedback,
    )
    duplicate_diag = duplicate.partner_policy_fusion_diagnostics
    assert duplicate_diag is not None
    assert not bool(duplicate_diag.feedback.applied)
    duplicate_interaction = cast(PrototypeInteractionState, duplicate.state.ia_state)
    _assert_tree_equal(
        duplicate_interaction.partner_policy_fusion_state,
        applied_interaction.partner_policy_fusion_state,
    )


def test_ia_coexists_with_partner_wrapper() -> None:
    agent = _agent(ia=True)
    state = agent.init(jr.key(19))
    assert isinstance(state.ia_state, PrototypeInteractionState)
    assert state.ia_state.ia_state is not None
    primed = agent.start(state, jnp.zeros((OBSERVATION_DIM,)))
    assert isinstance(primed.ia_state, PrototypeInteractionState)
    chex.assert_trees_all_equal(
        primed.ia_state.ia_state.cortex_state.stomp_state.base_last_obs,
        jnp.zeros((OBSERVATION_DIM,)),
    )
    result = agent.update_transition(
        primed,
        _transition(primed),
        partner_policy_fusion_input=_sidecar(
            agent,
            primed,
            suggested_action=1 - int(primed.current_action),
        ),
    )
    assert result.ia_augmented_obs is not None
    assert result.ia_recommendation is not None


def test_eager_jit_scan_checkpoint_and_fixed_resource_budget(tmp_path: Path) -> None:
    agent = _agent()
    initial = agent.start(agent.init(jr.key(23)), jnp.zeros((OBSERVATION_DIM,)))
    sidecar = _sidecar(
        agent,
        initial,
        suggested_action=1 - int(initial.current_action),
    )
    transition = _transition(initial)
    eager = agent.update_transition(
        initial,
        transition,
        partner_policy_fusion_input=sidecar,
    )
    compiled = jax.jit(agent.update_transition)(
        initial,
        transition,
        partner_policy_fusion_input=sidecar,
    )
    _assert_tree_equal(_materialize_keys(eager), _materialize_keys(compiled))

    transitions = jax.tree.map(lambda value: value[None], transition)
    sidecars = jax.tree.map(lambda value: value[None], sidecar)
    scanned = jax.jit(agent.scan_transitions)(
        initial,
        transitions,
        partner_policy_fusion_input=sidecars,
    )
    _assert_tree_equal(_materialize_keys(scanned.state), _materialize_keys(eager.state))

    checkpoint = tmp_path / "prototype-partner"
    save_prototype_checkpoint(agent, eager.state, checkpoint)
    restored_agent, restored_state = load_prototype_checkpoint(checkpoint)
    assert restored_agent.to_config() == agent.to_config()
    _assert_tree_equal(_materialize_keys(restored_state), _materialize_keys(eager.state))

    fusion = agent.partner_policy_fusion
    assert fusion is not None
    wrapper = cast(PrototypeInteractionState, eager.state.ia_state)
    budget = fusion.resource_budget
    actual_bytes = sum(
        int(np.asarray(leaf).nbytes)
        for leaf in jax.tree_util.tree_leaves(wrapper.partner_policy_fusion_state)
    )
    assert actual_bytes == budget.persistent_state_bytes
    wrapper_bytes = sum(
        int(np.asarray(leaf).nbytes)
        for leaf in jax.tree_util.tree_leaves(wrapper)
    )
    # Full Prototype lifecycle binding adds four uint32 words and one boolean.
    assert wrapper_bytes == budget.persistent_state_bytes + 17
    assert budget.replay_capacity == 0
    assert budget.dynamic_partner_capacity == 0


def test_exact_counter_capacity_disarms_without_advancing_partner_state() -> None:
    agent = _agent()
    state = agent.start(agent.init(jr.key(29)), jnp.zeros((OBSERVATION_DIM,)))
    maximum = jnp.asarray(np.iinfo(np.int32).max, dtype=jnp.int32)
    near_maximum_words = jnp.asarray(
        (np.iinfo(np.uint32).max, np.iinfo(np.uint32).max - 3),
        dtype=jnp.uint32,
    )
    near_maximum_observation_words = jnp.asarray(
        (np.iinfo(np.uint32).max, np.iinfo(np.uint32).max - 2),
        dtype=jnp.uint32,
    )
    exhausted = cast(
        Any,
        state.replace(
            step_count=maximum,
            step_words=near_maximum_words,
            observation_event_count=maximum,
            observation_event_words=near_maximum_observation_words,
            oak_state=state.oak_state.replace(
                step_count=maximum,
                step_words=near_maximum_words,
                stomp_state=state.oak_state.stomp_state.replace(
                    step_count=maximum,
                    step_words=near_maximum_words,
                ),
            ),
        ),
    )
    before = cast(PrototypeInteractionState, exhausted.ia_state)
    result = agent.update_transition(
        exhausted,
        _transition(exhausted),
        partner_policy_fusion_input=_sidecar(
            agent,
            exhausted,
            suggested_action=1 - int(exhausted.current_action),
        ),
    )
    assert bool(result.transition_diagnostics.valid)
    assert not bool(result.state.started)
    assert int(result.action) == -1
    after = cast(PrototypeInteractionState, result.state.ia_state)
    _assert_tree_equal(
        after.partner_policy_fusion_state,
        before.partner_policy_fusion_state,
    )


def test_partner_fusion_uses_exact_outer_identity_after_int32_saturation() -> None:
    agent = _agent()
    state = agent.start(agent.init(jr.key(290)), jnp.zeros((OBSERVATION_DIM,)))
    telemetry = jnp.asarray(np.iinfo(np.int32).max, dtype=jnp.int32)
    step_words = jnp.asarray(
        (0, np.iinfo(np.int32).max),
        dtype=jnp.uint32,
    )
    event_words = jnp.asarray(
        (0, np.iinfo(np.int32).max + 1),
        dtype=jnp.uint32,
    )
    base = state.oak_state.stomp_state.base_learner_state.replace(
        step_count=telemetry,
        step_words=step_words,
    )
    stomp = state.oak_state.stomp_state.replace(
        base_learner_state=base,
        step_count=telemetry,
        step_words=step_words,
    )
    oak = state.oak_state.replace(
        stomp_state=stomp,
        step_count=telemetry,
        step_words=step_words,
    )
    high = cast(
        Any,
        state.replace(
            oak_state=oak,
            step_count=telemetry,
            step_words=step_words,
            observation_event_count=telemetry,
            observation_event_words=event_words,
        ),
    )

    result = agent.update_transition(
        high,
        _transition(high),
        partner_policy_fusion_input=_sidecar(
            agent,
            high,
            suggested_action=1 - int(high.current_action),
        ),
    )

    assert bool(result.transition_diagnostics.valid)
    diagnostics = result.partner_policy_fusion_diagnostics
    assert diagnostics is not None
    assert bool(diagnostics.decision.availability.decision_identity_valid)
    wrapper = cast(PrototypeInteractionState, result.state.ia_state)
    fusion_state = wrapper.partner_policy_fusion_state
    np.testing.assert_array_equal(
        fusion_state.last_decision_words,
        result.state.step_words,
    )
    np.testing.assert_array_equal(
        fusion_state.last_event_words,
        result.state.observation_event_words,
    )
    assert int(fusion_state.last_decision_id) == np.iinfo(np.int32).max
    assert int(fusion_state.last_event_id) == np.iinfo(np.int32).max


def test_full_prototype_lifecycle_binding_rejects_old_decision_and_feedback() -> None:
    agent = _agent()
    observation = jnp.zeros((OBSERVATION_DIM,), dtype=jnp.float32)
    old = agent.start(
        agent.init(
            jr.key(37),
            lifecycle_id=jnp.asarray([1, 2], dtype=jnp.uint32),
        ),
        observation,
    )
    fresh = agent.start(
        agent.init(
            jr.key(37),
            lifecycle_id=jnp.asarray([9, 10], dtype=jnp.uint32),
        ),
        observation,
    )
    assert int(old.current_action) == int(fresh.current_action)
    old_sidecar = _sidecar(
        agent,
        old,
        suggested_action=1 - int(old.current_action),
    )
    replayed_decision = agent.update_transition(
        fresh,
        _transition(fresh),
        partner_policy_fusion_input=old_sidecar,
    )
    replay_diag = replayed_decision.partner_policy_fusion_diagnostics
    assert replay_diag is not None
    assert not bool(replay_diag.decision_prototype_decision_id_matches)
    assert int(replayed_decision.action) == int(replay_diag.counterfactual_base_action)
    fresh_before = cast(PrototypeInteractionState, fresh.ia_state)
    fresh_after = cast(PrototypeInteractionState, replayed_decision.state.ia_state)
    _assert_tree_equal(
        fresh_after.partner_policy_fusion_state,
        fresh_before.partner_policy_fusion_state,
    )

    old_armed = agent.update_transition(
        old,
        _transition(old),
        partner_policy_fusion_input=old_sidecar,
    )
    fresh_armed = agent.update_transition(
        fresh,
        _transition(fresh),
        partner_policy_fusion_input=_sidecar(
            agent,
            fresh,
            suggested_action=1 - int(fresh.current_action),
        ),
    )
    old_diag = old_armed.partner_policy_fusion_diagnostics
    assert old_diag is not None
    old_feedback = _feedback(
        old_diag.decision,
        old_armed.state.current_decision_id,
    )
    fresh_partner_before = cast(
        PrototypeInteractionState,
        fresh_armed.state.ia_state,
    )
    feedback_replay = agent.update_transition(
        fresh_armed.state,
        _transition(fresh_armed.state, (0.6, -0.1)),
        partner_policy_fusion_feedback=old_feedback,
    )
    feedback_diag = feedback_replay.partner_policy_fusion_diagnostics
    assert feedback_diag is not None
    assert not bool(feedback_diag.feedback_prototype_decision_id_matches)
    assert not bool(feedback_diag.feedback.applied)
    fresh_partner_after = cast(
        PrototypeInteractionState,
        feedback_replay.state.ia_state,
    )
    _assert_tree_equal(
        fresh_partner_after.partner_policy_fusion_state,
        fresh_partner_before.partner_policy_fusion_state,
    )


def test_keyboard_proposal_is_a_real_blend_source_and_dispatch_owner() -> None:
    agent = _agent(
        fusion_overrides={
            "min_feedback_for_learned_routing": 1,
            "base_blend_weight": 0.0,
        }
    )
    initial = agent.start(
        agent.init(jr.key(41)),
        jnp.zeros((OBSERVATION_DIM,), dtype=jnp.float32),
    )
    first_transition = _transition(initial)
    first_base = agent.update_transition(initial, first_transition)
    first = agent.update_transition(
        initial,
        first_transition,
        partner_policy_fusion_input=_sidecar(
            agent,
            initial,
            suggested_action=1 - int(first_base.action),
        ),
    )
    first_diag = first.partner_policy_fusion_diagnostics
    assert first_diag is not None

    # Force an idle OaK owner whose exact base Q selects primitive 0. The
    # option keyboard independently and strongly selects primitive 1.
    stomp = first.state.oak_state.stomp_state
    learner = stomp.base_learner_state
    base_params = learner.head_params.replace(
        weights=(
            jnp.asarray([[10.0, 0.0]], dtype=jnp.float32),
            jnp.asarray([[-10.0, 0.0]], dtype=jnp.float32),
            jnp.asarray([[-20.0, 0.0]], dtype=jnp.float32),
        )
    )
    option_weights = stomp.option_policies.q_weights.at[0].set(
        jnp.asarray(
            [[-5.0, 0.0], [5.0, 0.0]],
            dtype=jnp.float32,
        )
    )
    prepared_stomp = stomp.replace(
        base_learner_state=learner.replace(head_params=base_params),
        option_policies=stomp.option_policies.replace(q_weights=option_weights),
        executing_option=jnp.asarray(-1, dtype=jnp.int32),
        base_last_action=first.state.current_action,
        last_primitive_action=first.state.current_action,
    )
    prepared = first.state.replace(
        oak_state=first.state.oak_state.replace(stomp_state=prepared_stomp)
    )
    assert bool(agent._checkpoint_state_valid(prepared))
    transition = _transition(prepared, (1.0, 0.0))
    second = agent.update_transition(
        prepared,
        transition,
        partner_policy_fusion_input=_sidecar(
            agent,
            prepared,
            suggested_action=0,
            declared_confidence=0.4,
            keyboard_available=True,
            keyboard_vector=(1.0,),
        ),
        partner_policy_fusion_feedback=_feedback(
            first_diag.decision,
            first.state.current_decision_id,
        ),
    )
    diagnostics = second.partner_policy_fusion_diagnostics
    assert diagnostics is not None
    assert bool(second.transition_diagnostics.valid)
    assert bool(diagnostics.feedback.applied)
    assert int(diagnostics.decision.route) == ROUTE_BLEND
    assert bool(diagnostics.keyboard_proposal.available)
    assert int(diagnostics.keyboard_proposal.action) == 1
    assert int(diagnostics.decision.option_keyboard_action) == 1
    assert int(diagnostics.decision.scores.blend_selected_source) == (
        SOURCE_OPTION_KEYBOARD
    )
    assert bool(diagnostics.decision.option_keyboard_influenced)
    assert int(diagnostics.counterfactual_base_action) == 0
    assert int(diagnostics.effective_action) == 1
    assert int(second.state.oak_state.stomp_state.base_last_action) == 1

    authoritative = agent.update_transition(
        second.state,
        _transition(second.state, (0.3, 0.2)),
    )
    assert bool(authoritative.transition_diagnostics.valid)


def test_partner_replacement_rebuilds_recurrent_cache_for_effective_action() -> None:
    agent = _agent(recurrent=True)
    initial = agent.start(
        agent.init(jr.key(43)),
        jnp.asarray([0.2, -0.1], dtype=jnp.float32),
    )
    sidecar = _sidecar(
        agent,
        initial,
        suggested_action=1 - int(initial.current_action),
    )
    transition = _transition(initial, (0.4, -0.2))
    eager = agent.update_transition(
        initial,
        transition,
        partner_policy_fusion_input=sidecar,
    )
    compiled = jax.jit(agent.update_transition)(
        initial,
        transition,
        partner_policy_fusion_input=sidecar,
    )
    _assert_tree_equal(_materialize_keys(eager), _materialize_keys(compiled))
    assert bool(eager.transition_diagnostics.valid)
    recurrent = cast(
        PrototypeRecurrentLatentWorldModelState,
        eager.state.world_model_state,
    )
    assert int(recurrent.decision_cache.action) == int(eager.action)
    assert int(eager.state.current_action) == int(eager.action)
    assert int(eager.state.oak_state.stomp_state.last_primitive_action) == int(
        eager.action
    )
    chex.assert_trees_all_equal(
        recurrent.decision_cache.observation,
        eager.state.current_representation,
    )
    assert bool(agent._checkpoint_state_valid(eager.state))


def test_unsafe_base_fails_closed_by_rolling_back_eager_and_jit() -> None:
    agent = _agent()
    state = agent.start(
        agent.init(jr.key(47)),
        jnp.zeros((OBSERVATION_DIM,), dtype=jnp.float32),
    )
    transition = _transition(state)
    baseline = agent.update_transition(state, transition)
    unsafe_base = int(baseline.action)
    safe_partner = 1 - unsafe_base
    mask = (False, True) if unsafe_base == 0 else (True, False)
    sidecar = _sidecar(
        agent,
        state,
        suggested_action=safe_partner,
        mask=mask,
    )
    eager = agent.update_transition(
        state,
        transition,
        partner_policy_fusion_input=sidecar,
    )
    compiled = jax.jit(agent.update_transition)(
        state,
        transition,
        partner_policy_fusion_input=sidecar,
    )
    _assert_tree_equal(_materialize_keys(eager), _materialize_keys(compiled))
    assert not bool(eager.transition_diagnostics.valid)
    assert bool(eager.transition_diagnostics.post_update_checked)
    assert int(eager.action) == int(state.current_action)
    assert bool(eager.state.started)
    assert int(eager.state.current_action) >= 0
    _assert_tree_equal(_materialize_keys(eager.state), _materialize_keys(state))
    diagnostics = eager.partner_policy_fusion_diagnostics
    assert diagnostics is not None
    assert bool(diagnostics.dispatch_replacement.failed_closed)
    assert not bool(diagnostics.transaction_applied)


def test_boundary_sidecar_binds_the_autoreset_decision_event() -> None:
    agent = _agent()
    state = agent.start(
        agent.init(jr.key(53)),
        jnp.zeros((OBSERVATION_DIM,), dtype=jnp.float32),
    )
    final_observation = jnp.asarray([0.8, -0.4], dtype=jnp.float32)
    reset_observation = jnp.asarray([-0.2, 0.6], dtype=jnp.float32)
    transition = PrototypeTransition(
        observation=state.current_raw_observation,
        action=state.current_action,
        decision_id=state.current_decision_id,
        reward=jnp.asarray(0.3, dtype=jnp.float32),
        discount=jnp.asarray(0.0, dtype=jnp.float32),
        terminated=jnp.asarray(True, dtype=jnp.bool_),
        truncated=jnp.asarray(False, dtype=jnp.bool_),
        next_observation=final_observation,
        next_decision_observation=reset_observation,
    )
    result = agent.update_transition(
        state,
        transition,
        partner_policy_fusion_input=_sidecar(
            agent,
            state,
            suggested_action=1 - int(state.current_action),
            event_increment=2,
        ),
    )
    diagnostics = result.partner_policy_fusion_diagnostics
    assert diagnostics is not None
    assert bool(result.transition_diagnostics.valid)
    assert int(diagnostics.decision.event_id) == int(
        state.observation_event_count
    ) + 2
    assert int(result.state.observation_event_count) == int(
        state.observation_event_count
    ) + 2
    chex.assert_trees_all_equal(
        result.state.current_raw_observation,
        reset_observation,
    )
