# mypy: disable-error-code="attr-defined,call-arg"
"""Focused contracts for the live Prototype/STOMP calibrated-search adapter."""

from __future__ import annotations

import dataclasses
from typing import cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.calibrated_extended_search_control import (
    CANDIDATE_KIND_OPTION,
    CANDIDATE_KIND_PRIMITIVE,
    SEARCH_MODE_COMBINED,
    CalibratedExtendedSearchControlConfig,
)
from alberta_framework.core.oak import OaKConfig, OaKState
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import (
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeAgentState,
    PrototypeTransition,
)
from alberta_framework.core.prototype_stomp_calibrated_search import (
    PROTOTYPE_STOMP_CALIBRATED_SEARCH_ERROR_CAPACITY,
    PrototypeSTOMPCalibratedSearchAgent,
    PrototypeSTOMPCalibratedSearchConfig,
    PrototypeSTOMPCalibratedSearchState,
)
from alberta_framework.core.world_model import (
    ActionConditionedWorldModel,
    ActionConditionedWorldModelConfig,
    ActionConditionedWorldModelState,
)

pytestmark = pytest.mark.unit

ANCHORS = jnp.asarray(
    ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)),
    dtype=jnp.float32,
)
ACTIVE = jnp.ones((4,), dtype=jnp.bool_)
SOURCE = jnp.asarray((0xC0DE, 0x51DE), dtype=jnp.uint32)


def _config(
    *,
    enabled: bool = True,
    option_threshold: float = 100.0,
    max_option_steps: int = 3,
    max_observations: int = 32,
) -> PrototypeSTOMPCalibratedSearchConfig:
    prototype = PrototypeAgentConfig(
        oak=OaKConfig(
            stomp=STOMPConfig(
                subtask_specs=(
                    SubtaskSpec(
                        feature_index=0,
                        threshold=option_threshold,
                        pseudo_reward_scale=1.0,
                        max_option_steps=max_option_steps,
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
        max_observations=max_observations,
    )
    return PrototypeSTOMPCalibratedSearchConfig(
        prototype=prototype,
        search=search,
        enabled=enabled,
    )


def _adapter_state(
    adapter: PrototypeSTOMPCalibratedSearchAgent,
    *,
    selected_extended_action: int,
) -> PrototypeSTOMPCalibratedSearchState:
    state = adapter.init(
        jr.key(7),
        anchor_bank=ANCHORS,
        anchor_active=ACTIVE,
        source_digest=SOURCE,
        representation_generation=3,
        lifecycle_id=jnp.asarray((17, 29), dtype=jnp.uint32),
    )
    prototype = state.prototype
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
    stomp = stomp.replace(base_learner_state=learner)
    oak = cast(OaKState, oak.replace(stomp_state=stomp))
    prototype = cast(PrototypeAgentState, prototype.replace(oak_state=oak))
    rebound = adapter.rebind(
        state,
        prototype_state=prototype,
        source_digest=SOURCE,
        representation_generation=3,
    )
    assert bool(rebound.transaction_applied)
    assert bool(adapter.validate_state(rebound.state))
    return rebound.state


def _transition(
    state: PrototypeSTOMPCalibratedSearchState,
    next_observation: jax.Array,
    *,
    reward: float = 1.0,
    discount: float = 1.0,
    terminated: bool = False,
    truncated: bool = False,
    next_decision_observation: jax.Array | None = None,
) -> PrototypeTransition:
    decision = state.prototype
    return PrototypeTransition(
        observation=decision.current_raw_observation,
        action=decision.current_action,
        decision_id=decision.current_decision_id,
        reward=jnp.asarray(reward, dtype=jnp.float32),
        discount=jnp.asarray(discount, dtype=jnp.float32),
        terminated=jnp.asarray(terminated, dtype=jnp.bool_),
        truncated=jnp.asarray(truncated, dtype=jnp.bool_),
        next_observation=next_observation,
        next_decision_observation=(
            next_observation
            if next_decision_observation is None
            else next_decision_observation
        ),
    )


def test_config_rejects_any_second_planning_budget_or_nonlegacy_encoding() -> None:
    config = _config()
    assert PrototypeSTOMPCalibratedSearchConfig.from_config(config.to_config()) == config
    with pytest.raises(ValueError, match="legacy STOMP planning"):
        PrototypeSTOMPCalibratedSearchConfig(
            prototype=PrototypeAgentConfig(
                oak=OaKConfig(
                    stomp=dataclasses.replace(
                        config.prototype.oak.stomp,
                        option_planning_backups_per_step=1,
                    )
                ),
                world_model=config.prototype.world_model,
            ),
            search=config.search,
        )


def test_primitive_arm_resolves_after_exactly_one_real_transition() -> None:
    adapter = PrototypeSTOMPCalibratedSearchAgent(_config())
    state = _adapter_state(adapter, selected_extended_action=0)
    started = adapter.start(state, ANCHORS[0])
    assert int(started.state.search.pending_executed_kind) == CANDIDATE_KIND_PRIMITIVE
    owner = np.asarray(started.state.search.pending_decision_id)

    result = adapter.update_transition(started.state, _transition(started.state, ANCHORS[1]))

    assert bool(result.diagnostics.prototype_transition_applied)
    assert bool(result.diagnostics.resolution_attempted)
    assert bool(result.diagnostics.natural_resolution)
    assert not bool(result.diagnostics.censored_resolution)
    assert bool(result.search_observe.diagnostics.transaction_applied)
    assert int(result.search_observe.state.last_decision_id[-1]) == int(owner[-1])
    assert bool(adapter.validate_state(result.state))


def test_option_arms_once_persists_across_steps_and_resolves_naturally() -> None:
    adapter = PrototypeSTOMPCalibratedSearchAgent(_config(max_option_steps=3))
    option_head = adapter.config.prototype.oak.n_primitive_actions
    state = _adapter_state(adapter, selected_extended_action=option_head)
    started = adapter.start(state, ANCHORS[0])
    assert int(started.state.search.pending_executed_kind) == CANDIDATE_KIND_OPTION
    assert bool(started.diagnostics.arm_applied)
    owner = np.asarray(started.state.search.pending_decision_id)
    frozen_revision = int(started.state.search.state_revision)

    first = adapter.update_transition(started.state, _transition(started.state, ANCHORS[1]))
    assert not bool(first.diagnostics.resolution_attempted)
    assert not bool(first.diagnostics.arm_attempted)
    assert int(first.state.pending_elapsed_primitive_steps) == 1
    assert int(first.state.search.state_revision) == frozen_revision
    np.testing.assert_array_equal(first.state.search.pending_decision_id, owner)
    # These frozen words/checksums identify the model snapshot that armed the
    # option. They intentionally do not chase the live world model as it learns
    # during the option's intermediate primitive transitions.
    np.testing.assert_array_equal(
        first.state.pending_primitive_model_words,
        started.state.pending_primitive_model_words,
    )
    np.testing.assert_array_equal(
        first.state.pending_primitive_model_checksum,
        started.state.pending_primitive_model_checksum,
    )
    live_world = cast(
        ActionConditionedWorldModelState,
        first.state.prototype.world_model_state,
    )
    assert not np.array_equal(
        live_world.step_words,
        first.state.pending_primitive_model_words,
    )
    assert bool(adapter.validate_state(first.state))

    second = adapter.update_transition(first.state, _transition(first.state, ANCHORS[2]))
    assert not bool(second.diagnostics.resolution_attempted)
    assert not bool(second.diagnostics.arm_attempted)
    assert int(second.state.pending_elapsed_primitive_steps) == 2
    np.testing.assert_array_equal(second.state.search.pending_decision_id, owner)

    third = adapter.update_transition(second.state, _transition(second.state, ANCHORS[3]))
    assert bool(third.diagnostics.natural_resolution)
    assert not bool(third.diagnostics.censored_resolution)
    assert bool(third.search_observe.diagnostics.transaction_applied)
    np.testing.assert_array_equal(third.search_observe.state.last_decision_id, owner)
    assert bool(adapter.validate_state(third.state))


@pytest.mark.parametrize(
    ("terminated", "truncated", "discount"),
    ((False, True, 1.0), (True, False, 0.0)),
)
def test_option_truncation_and_environment_ending_are_censored(
    terminated: bool,
    truncated: bool,
    discount: float,
) -> None:
    adapter = PrototypeSTOMPCalibratedSearchAgent(_config(max_option_steps=20))
    option_head = adapter.config.prototype.oak.n_primitive_actions
    state = _adapter_state(adapter, selected_extended_action=option_head)
    started = adapter.start(state, ANCHORS[0])
    support_before = np.asarray(started.state.search.support_counts)
    transition = _transition(
        started.state,
        ANCHORS[1],
        discount=discount,
        terminated=terminated,
        truncated=truncated,
        next_decision_observation=ANCHORS[2],
    )

    result = adapter.update_transition(started.state, transition)

    assert bool(result.diagnostics.censored_resolution)
    assert not bool(result.diagnostics.natural_resolution)
    assert bool(result.search_observe.diagnostics.transaction_applied)
    np.testing.assert_array_equal(result.search_observe.state.support_counts, support_before)
    # Outcome/censoring uses the final observation, but the next arm belongs to
    # the autoreset decision observation selected by Prototype.
    np.testing.assert_array_equal(
        result.state.search.pending_anchor_observation,
        ANCHORS[2],
    )
    assert bool(adapter.validate_state(result.state))


def test_environment_end_that_independently_reaches_goal_settles_then_arms_reset() -> None:
    adapter = PrototypeSTOMPCalibratedSearchAgent(
        _config(option_threshold=0.5, max_option_steps=20)
    )
    option_head = adapter.config.prototype.oak.n_primitive_actions
    state = _adapter_state(adapter, selected_extended_action=option_head)
    started = adapter.start(state, ANCHORS[1])
    transition = _transition(
        started.state,
        ANCHORS[0],
        discount=0.0,
        terminated=True,
        next_decision_observation=ANCHORS[3],
    )

    result = adapter.update_transition(started.state, transition)

    assert bool(result.diagnostics.natural_resolution)
    assert not bool(result.diagnostics.censored_resolution)
    assert bool(result.search_observe.diagnostics.future_anchor_evidence_valid)
    np.testing.assert_array_equal(
        result.state.search.pending_anchor_observation,
        ANCHORS[3],
    )


def test_snapshot_is_the_actual_learned_world_and_stomp_models() -> None:
    adapter = PrototypeSTOMPCalibratedSearchAgent(_config())
    state = _adapter_state(adapter, selected_extended_action=0)
    started = adapter.start(state, ANCHORS[0])
    result = adapter.update_transition(started.state, _transition(started.state, ANCHORS[1]))
    snapshot = result.model_snapshot
    world_state = cast(ActionConditionedWorldModelState, result.state.prototype.world_model_state)
    world = ActionConditionedWorldModel(
        cast(
            ActionConditionedWorldModelConfig,
            adapter.config.prototype.world_model,
        )
    )
    expected = jax.vmap(
        lambda anchor: jax.vmap(
            lambda action: world.predict(world_state, anchor, action)
        )(jnp.arange(2, dtype=jnp.int32))
    )(ANCHORS)
    np.testing.assert_allclose(snapshot.primitive_reward_predictions, expected.reward)
    np.testing.assert_allclose(snapshot.primitive_discount_predictions, expected.discount)
    np.testing.assert_allclose(
        snapshot.primitive_next_observation_predictions,
        expected.next_observation,
    )
    stomp = cast(OaKState, result.state.prototype.oak_state).stomp_state
    np.testing.assert_allclose(
        snapshot.option_return_predictions,
        jnp.broadcast_to(stomp.option_models.env_return_ema[None, :], (4, 1)),
    )
    np.testing.assert_allclose(
        snapshot.option_baseline_mass_predictions,
        jnp.broadcast_to(stomp.option_models.baseline_mass_ema[None, :], (4, 1)),
    )
    np.testing.assert_allclose(
        snapshot.option_discount_predictions,
        jnp.broadcast_to(stomp.option_models.discount_ema[None, :], (4, 1)),
    )
    np.testing.assert_array_equal(
        snapshot.option_completion_counts,
        stomp.option_models.n_completions,
    )


def test_disabled_adapter_is_bit_exact_to_raw_prototype() -> None:
    config = _config(enabled=False)
    adapter = PrototypeSTOMPCalibratedSearchAgent(config)
    raw = PrototypeAgent(config.prototype)
    wrapped = adapter.init(
        jr.key(41),
        anchor_bank=ANCHORS,
        anchor_active=ACTIVE,
        source_digest=SOURCE,
        representation_generation=0,
        lifecycle_id=jnp.asarray((9, 13), dtype=jnp.uint32),
    )
    raw_state = raw.init(
        jr.key(41), lifecycle_id=jnp.asarray((9, 13), dtype=jnp.uint32)
    )
    chex.assert_trees_all_equal(wrapped.prototype, raw_state)
    wrapped_start = adapter.start(wrapped, ANCHORS[0])
    raw_state = raw.start(raw_state, ANCHORS[0])
    chex.assert_trees_all_equal(wrapped_start.state.prototype, raw_state)
    transition = _transition(wrapped_start.state, ANCHORS[1])
    raw_result = raw.update_transition(raw_state, transition)
    wrapped_result = adapter.update_transition(wrapped_start.state, transition)
    chex.assert_trees_all_equal(wrapped_result.prototype, raw_result)
    assert wrapped_result.search_arm is None
    assert wrapped_result.search_observe is None


def test_search_capacity_failure_keeps_successful_prototype_update_bit_exact() -> None:
    adapter = PrototypeSTOMPCalibratedSearchAgent(_config(max_observations=2))
    state = _adapter_state(adapter, selected_extended_action=0)
    current = adapter.start(state, ANCHORS[0]).state
    for future in (ANCHORS[1], ANCHORS[0]):
        transition = _transition(current, future)
        raw = adapter.prototype.update_transition(current.prototype, transition)
        result = adapter.update_transition(current, transition)
        chex.assert_trees_all_equal(result.state.prototype, raw.state)
        assert int(result.prototype.action) == int(raw.action)
        current = result.state
    assert bool(current.search_unavailable)
    assert int(current.search_error) == PROTOTYPE_STOMP_CALIBRATED_SEARCH_ERROR_CAPACITY
    assert bool(adapter.validate_state(current))


def test_rejected_transition_is_an_exact_state_noop_with_unarmed_result() -> None:
    adapter = PrototypeSTOMPCalibratedSearchAgent(_config())
    state = _adapter_state(adapter, selected_extended_action=0)
    started = adapter.start(state, ANCHORS[0]).state
    transition = _transition(started, ANCHORS[1]).replace(
        decision_id=started.prototype.current_decision_id.at[-1].add(
            jnp.uint32(1)
        )
    )

    result = adapter.update_transition(started, transition)

    assert not bool(result.diagnostics.prototype_transition_applied)
    assert not bool(result.diagnostics.transaction_committed)
    assert int(result.prototype.action) == -1
    assert not bool(result.decision.armed)
    assert not bool(result.diagnostics.observe_applied)
    assert not bool(result.diagnostics.arm_attempted)
    assert not bool(result.diagnostics.arm_applied)
    chex.assert_trees_all_equal(result.state, started)


def test_pending_checkpoint_and_rebind_censor_are_strict() -> None:
    adapter = PrototypeSTOMPCalibratedSearchAgent(_config(max_option_steps=8))
    state = _adapter_state(
        adapter,
        selected_extended_action=adapter.config.prototype.oak.n_primitive_actions,
    )
    started = adapter.start(state, ANCHORS[0])
    payload = adapter.checkpoint_payload(started.state)
    restored = adapter.restore_checkpoint(
        payload,
        source_digest=SOURCE,
        representation_generation=3,
    )
    chex.assert_trees_all_equal(restored, started.state)
    rebound = adapter.rebind(
        restored,
        prototype_state=restored.prototype,
        source_digest=SOURCE,
        representation_generation=3,
    )
    assert bool(rebound.transaction_applied)
    assert bool(rebound.pending_censored)
    assert not bool(rebound.state.search.pending)
    assert not bool(rebound.state.adapter_pending)
    assert bool(adapter.validate_state(rebound.state))


def test_resource_declaration_has_one_budget_zero_rng_and_no_authority() -> None:
    adapter = PrototypeSTOMPCalibratedSearchAgent(_config())
    state = adapter.init(
        jr.key(2),
        anchor_bank=ANCHORS,
        anchor_active=ACTIVE,
        source_digest=SOURCE,
        representation_generation=0,
    )
    budget = adapter.resource_budget(state)
    assert budget.total_secondary_backup_attempts_per_resolution == 1
    assert budget.primitive_and_option_share_one_budget
    assert budget.planner_rng_draws_total == 0
    assert not budget.search_exhaustion_can_block_prototype
    assert budget.q_surface_sidecar_only
    assert not budget.policy_authority
    assert not budget.dispatch_authority
    assert not budget.automatic_keyboard_dispatch
    assert not budget.control_benefit_established
    assert not budget.scientific_promotion_allowed
