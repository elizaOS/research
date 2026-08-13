"""Exact lifetime-boundary tests for the integrated hidden-partner kernel."""

from __future__ import annotations

import dataclasses
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.grounded_joint_world_model import (
    GroundedJointWorldModelConfig,
)
from alberta_framework.core.integrated_hidden_partner import (
    ACTIVE_PAIR_SLOTS,
    CANDIDATE_PAIR_SLOTS,
    DEPLOYED_FEATURE_DIM,
    INTEGRATED_CHILD_CLOCK_ALIGNMENT_ORDER,
    INTEGRATED_HIDDEN_PARTNER_LIFETIME_COUNTER_DELTA_NBYTES,
    INTEGRATED_HIDDEN_PARTNER_LIFETIME_COUNTER_NBYTES,
    INTEGRATED_HIDDEN_PARTNER_SCHEMA_VERSION,
    RAW_OBSERVATION_DIM,
    IntegratedHiddenPartnerAgent,
    IntegratedHiddenPartnerConfig,
)
from alberta_framework.core.representation_gradient_mixer import (
    RepresentationGradientMixerConfig,
)
from alberta_framework.streams.hidden_partner_mapping import (
    HiddenPartnerMappingConfig,
    HiddenPartnerMappingWorld,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1


def _environment() -> HiddenPartnerMappingWorld:
    return HiddenPartnerMappingWorld(
        HiddenPartnerMappingConfig(
            base_segment_lengths=(4,) * 9,
            jitter_radius=0,
            partner_flip_probability=0.0,
        )
    )


def _start_and_transition(
    agent: IntegratedHiddenPartnerAgent,
    *,
    seed: int,
) -> tuple[Any, Any]:
    environment = _environment()
    environment_state = environment.init(jr.key(seed))
    start = agent.start(
        environment.observe(environment_state),
        jr.key(seed + 10_000),
    )
    transition, _ = environment.step(environment_state, start.action)
    return start, transition


def _words(high: int, low: int) -> Any:
    return jnp.asarray((high, low), dtype=jnp.uint32)


def _telemetry(high: int, low: int) -> Any:
    value = low if high == 0 and low < _INT32_MAX else _INT32_MAX
    return jnp.asarray(value, dtype=jnp.int32)


def _aligned_state(
    agent: IntegratedHiddenPartnerAgent,
    state: Any,
    *,
    high: int,
    low: int,
) -> Any:
    outer_value = (high << 32) | low
    if outer_value >= (1 << 64) - 1:
        raise ValueError("an aligned builder offset requires outer < 2^64 - 1")
    builder_value = outer_value + 1
    builder_high = builder_value >> 32
    builder_low = builder_value & _UINT32_MAX
    outer_words = _words(high, low)
    outer_count = _telemetry(high, low)
    builder_words = _words(builder_high, builder_low)
    builder_count = _telemetry(builder_high, builder_low)
    zero_words = _words(0, 0)
    builder_update_words = (
        outer_words if agent.config.state_learning_enabled else zero_words
    )
    builder_update_count = (
        outer_count
        if agent.config.state_learning_enabled
        else jnp.asarray(0, dtype=jnp.int32)
    )
    interval = agent.config.replacement_interval
    replacement_phase = 0 if interval == 0 else outer_value % interval
    interaction = state.interaction.replace(
        step_count=outer_count,
        step_words=outer_words,
        replacement_phase=jnp.asarray(replacement_phase, dtype=jnp.int32),
    )
    builder = state.state_builder.replace(
        step_count=builder_count,
        step_words=builder_words,
        update_count=builder_update_count,
        update_words=builder_update_words,
    )
    behavior = state.behavior.replace(
        step_count=outer_count,
        step_words=outer_words,
    )
    joint_world = state.joint_world.replace(
        step_count=outer_count,
        step_words=outer_words,
    )
    control = state.control.replace(
        step_count=outer_count,
        step_words=outer_words,
    )
    router = dataclasses.replace(
        state.router,
        route_count=outer_count,
        route_words=outer_words,
    )
    grounded_world = state.grounded_world
    if grounded_world is not None:
        grounded_words = (
            outer_words
            if agent.config.grounded_world_learning_enabled
            else zero_words
        )
        grounded_count = (
            outer_count
            if agent.config.grounded_world_learning_enabled
            else jnp.asarray(0, dtype=jnp.int32)
        )
        grounded_world = grounded_world.replace(
            update_count=grounded_count,
            update_words=grounded_words,
        )
    return state.replace(
        state_builder=builder,
        interaction=interaction,
        behavior=behavior,
        joint_world=joint_world,
        grounded_world=grounded_world,
        control=control,
        router=router,
        step_count=outer_count,
        step_words=outer_words,
    )


def test_start_authenticates_zero_outer_clock_and_builder_observation_offset() -> None:
    agent = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(replacement_interval=0)
    )
    start, _ = _start_and_transition(agent, seed=90_001)
    zero = _words(0, 0)

    chex.assert_trees_all_equal(start.state.step_words, zero)
    chex.assert_trees_all_equal(start.state.behavior.step_words, zero)
    chex.assert_trees_all_equal(start.state.joint_world.step_words, zero)
    chex.assert_trees_all_equal(start.state.control.step_words, zero)
    chex.assert_trees_all_equal(start.state.router.route_words, zero)
    chex.assert_trees_all_equal(start.state.interaction.step_words, zero)
    chex.assert_trees_all_equal(start.state.state_builder.step_words, _words(0, 1))
    chex.assert_trees_all_equal(start.state.state_builder.update_words, zero)
    assert len(INTEGRATED_CHILD_CLOCK_ALIGNMENT_ORDER) == 9
    assert bool(start.diagnostics.outer_lifetime_counter_valid)
    assert bool(start.diagnostics.child_clocks_aligned)
    assert bool(jnp.all(start.diagnostics.child_clock_alignment_vector))


@pytest.mark.parametrize(
    "child",
    (
        "behavior",
        "joint_world",
        "control",
        "router",
        "interaction",
        "state_builder_step",
        "state_builder_update",
    ),
)
def test_locally_valid_but_outer_misaligned_child_clock_rejects_atomically(
    child: str,
) -> None:
    agent = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(replacement_interval=0)
    )
    start, transition = _start_and_transition(agent, seed=90_010)
    state = start.state
    one_words = _words(0, 1)
    one = jnp.asarray(1, dtype=jnp.int32)
    if child == "behavior":
        corrupted = state.replace(
            behavior=state.behavior.replace(step_count=one, step_words=one_words)
        )
    elif child == "joint_world":
        corrupted = state.replace(
            joint_world=state.joint_world.replace(
                step_count=one,
                step_words=one_words,
            )
        )
    elif child == "control":
        corrupted = state.replace(
            control=state.control.replace(step_count=one, step_words=one_words)
        )
    elif child == "router":
        corrupted = state.replace(
            router=dataclasses.replace(
                state.router,
                route_count=one,
                route_words=one_words,
            )
        )
    elif child == "interaction":
        corrupted = state.replace(
            interaction=state.interaction.replace(
                step_count=one,
                step_words=one_words,
                replacement_phase=jnp.asarray(0, dtype=jnp.int32),
            )
        )
    elif child == "state_builder_step":
        corrupted = state.replace(
            state_builder=state.state_builder.replace(
                step_count=jnp.asarray(2, dtype=jnp.int32),
                step_words=_words(0, 2),
            )
        )
    else:
        corrupted = state.replace(
            state_builder=state.state_builder.replace(
                update_count=one,
                update_words=one_words,
            )
        )

    result = agent.update(corrupted, transition)

    chex.assert_trees_all_equal(result.state, corrupted)
    assert not bool(result.diagnostics.pre_child_clocks_aligned)
    assert not bool(result.diagnostics.transition_applied)
    assert bool(result.diagnostics.transition_rejected)
    chex.assert_trees_all_equal(
        result.diagnostics.outer_committed_post_step_words,
        corrupted.step_words,
    )


def test_low_word_carry_preserves_every_exact_child_alignment() -> None:
    agent = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(
            feature_lifecycle_enabled=False,
            replacement_interval=0,
        )
    )
    start, transition = _start_and_transition(agent, seed=90_020)
    aligned = _aligned_state(
        agent,
        start.state,
        high=0,
        low=_UINT32_MAX - 1,
    )

    result = agent.update(aligned, transition)

    assert bool(result.diagnostics.transition_applied)
    chex.assert_trees_all_equal(result.state.step_words, _words(0, _UINT32_MAX))
    chex.assert_trees_all_equal(
        result.state.state_builder.step_words,
        _words(1, 0),
    )
    for words in (
        result.state.behavior.step_words,
        result.state.joint_world.step_words,
        result.state.control.step_words,
        result.state.router.route_words,
        result.state.interaction.step_words,
        result.state.state_builder.update_words,
    ):
        chex.assert_trees_all_equal(words, result.state.step_words)
    assert bool(result.diagnostics.proposed_post_child_clocks_aligned)
    assert bool(result.diagnostics.committed_post_child_clocks_aligned)


def test_corrupt_outer_telemetry_rejects_before_transaction_authorization() -> None:
    agent = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(replacement_interval=0)
    )
    start, transition = _start_and_transition(agent, seed=90_025)
    corrupted = start.state.replace(
        step_count=jnp.asarray(1, dtype=jnp.int32)
    )

    result = agent.update(corrupted, transition)

    chex.assert_trees_all_equal(result.state, corrupted)
    assert not bool(result.diagnostics.outer_lifetime_counter_valid)
    assert bool(result.diagnostics.pre_child_clocks_aligned)
    assert not bool(result.diagnostics.transition_applied)
    assert bool(result.diagnostics.transition_rejected)


def test_builder_offset_reserves_terminal_outer_identity_and_rolls_back() -> None:
    agent = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(
            feature_lifecycle_enabled=False,
            replacement_interval=0,
        )
    )
    start, transition = _start_and_transition(agent, seed=90_030)
    terminal = _aligned_state(
        agent,
        start.state,
        high=_UINT32_MAX,
        low=_UINT32_MAX - 1,
    )

    result = agent.update(terminal, transition)

    chex.assert_trees_all_equal(result.state, terminal)
    assert bool(result.diagnostics.outer_lifetime_capacity_available)
    assert not bool(result.diagnostics.state_builder_step_capacity_available)
    assert not bool(result.diagnostics.transaction_capacity_available)
    assert not bool(result.diagnostics.transition_applied)
    assert bool(result.diagnostics.transition_rejected)
    chex.assert_trees_all_equal(
        result.diagnostics.outer_proposed_post_step_words,
        _words(_UINT32_MAX, _UINT32_MAX),
    )
    chex.assert_trees_all_equal(
        result.diagnostics.outer_committed_post_step_words,
        terminal.step_words,
    )


def test_rejected_proposals_and_committed_clocks_are_distinct_under_jit_and_scan() -> None:
    agent = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(replacement_interval=0)
    )
    start, transition = _start_and_transition(agent, seed=90_040)
    invalid = transition.replace(
        reward=jnp.asarray(jnp.nan, dtype=jnp.float32)
    )

    def apply(state: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
        result = agent.update(state, invalid)
        diagnostics = result.diagnostics
        return (
            result.state,
            diagnostics.transition_applied,
            diagnostics.outer_proposed_post_step_words,
            diagnostics.outer_committed_post_step_words,
            diagnostics.behavior_proposed_post_step_words,
            diagnostics.behavior_post_step_words,
        )

    eager = apply(start.state)
    compiled = jax.jit(apply)(start.state)
    chex.assert_trees_all_equal(eager, compiled)
    chex.assert_trees_all_equal(eager[0], start.state)
    assert not bool(eager[1])
    chex.assert_trees_all_equal(eager[2], _words(0, 1))
    chex.assert_trees_all_equal(eager[3], _words(0, 0))
    chex.assert_trees_all_equal(eager[4], _words(0, 1))
    chex.assert_trees_all_equal(eager[5], _words(0, 0))

    rejected = agent.update(start.state, invalid).diagnostics
    for proposed in (
        rejected.interaction_proposed_post_step_words,
        rejected.world_proposed_post_step_words,
        rejected.control_proposed_post_step_words,
        rejected.router_proposed_post_route_words,
    ):
        chex.assert_trees_all_equal(proposed, _words(0, 1))
    for committed in (
        rejected.interaction_post_step_words,
        rejected.world_post_step_words,
        rejected.control_post_step_words,
        rejected.router_committed_post_route_words,
    ):
        chex.assert_trees_all_equal(committed, _words(0, 0))
    chex.assert_trees_all_equal(
        rejected.state_builder_proposed_post_step_words,
        _words(0, 2),
    )
    chex.assert_trees_all_equal(
        rejected.state_builder_post_step_words,
        _words(0, 1),
    )

    def scan_step(state: Any, _: Any) -> tuple[Any, tuple[Any, Any]]:
        result = agent.update(state, invalid)
        return result.state, (
            result.diagnostics.transition_applied,
            result.diagnostics.outer_committed_post_step_words,
        )

    final_state, (applied, committed_words) = jax.jit(
        lambda state: jax.lax.scan(
            scan_step,
            state,
            jnp.arange(2, dtype=jnp.int32),
        )
    )(start.state)
    chex.assert_trees_all_equal(final_state, start.state)
    chex.assert_trees_all_equal(applied, jnp.zeros((2,), dtype=jnp.bool_))
    chex.assert_trees_all_equal(
        committed_words,
        jnp.zeros((2, 2), dtype=jnp.uint32),
    )


def test_outer_rejection_preserves_interaction_proposal_but_applies_no_lifecycle() -> None:
    agent = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(
            replacement_interval=1,
            min_feature_age=0,
            candidate_min_age=0,
        )
    )
    start, transition = _start_and_transition(agent, seed=90_045)
    prepared = start.state.replace(
        interaction=start.state.interaction.replace(
            ages=jnp.ones((ACTIVE_PAIR_SLOTS,), dtype=jnp.int32),
            candidate_ages=jnp.ones((CANDIDATE_PAIR_SLOTS,), dtype=jnp.int32),
            utilities=jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.float32),
            candidate_utilities=jnp.full(
                (CANDIDATE_PAIR_SLOTS,),
                100.0,
                dtype=jnp.float32,
            ),
        )
    )
    invalid = transition.replace(
        reward=jnp.asarray(jnp.nan, dtype=jnp.float32)
    )

    result = agent.update(prepared, invalid)
    diagnostics = result.diagnostics

    chex.assert_trees_all_equal(result.state, prepared)
    assert bool(diagnostics.interaction_proposal_applied)
    assert bool(diagnostics.interaction_lifecycle_proposed)
    assert int(diagnostics.interaction_proposal_replaced_slot) >= 0
    assert int(diagnostics.interaction_replaced_slot) == int(
        diagnostics.interaction_proposal_replaced_slot
    )
    assert not bool(diagnostics.interaction_update_applied)
    assert not bool(diagnostics.interaction_lifecycle_applied)
    assert int(diagnostics.interaction_applied_replaced_slot) == -1
    assert bool(
        jnp.any(
            diagnostics.interaction_proposal_descriptors
            != prepared.router.descriptors
        )
    )
    chex.assert_trees_all_equal(
        diagnostics.interaction_applied_descriptors,
        prepared.router.descriptors,
    )
    assert not bool(diagnostics.transition_applied)
    assert bool(diagnostics.transition_rejected)


def test_disabled_learning_clocks_remain_zero_while_transition_clocks_advance() -> None:
    grounded = GroundedJointWorldModelConfig(
        representation_dim=DEPLOYED_FEATURE_DIM,
        target_observation_dim=RAW_OBSERVATION_DIM,
        n_focal_actions=2,
        n_partner_actions=2,
        step_size=0.2,
        initialization_scale=0.05,
        max_input_magnitude=100.0,
        max_parameter_magnitude=100.0,
    )
    agent = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(
            state_learning_enabled=False,
            grounded_world_model=grounded,
            representation_gradient_mixer=RepresentationGradientMixerConfig(
                representation_dim=DEPLOYED_FEATURE_DIM,
                mode="world_only",
            ),
            grounded_world_learning_enabled=False,
            feature_lifecycle_enabled=False,
            replacement_interval=0,
        )
    )
    start, transition = _start_and_transition(agent, seed=90_050)

    result = agent.update(start.state, transition)

    assert result.state.grounded_world is not None
    assert bool(result.diagnostics.transition_applied)
    chex.assert_trees_all_equal(result.state.step_words, _words(0, 1))
    chex.assert_trees_all_equal(
        result.state.state_builder.step_words,
        _words(0, 2),
    )
    chex.assert_trees_all_equal(
        result.state.state_builder.update_words,
        _words(0, 0),
    )
    chex.assert_trees_all_equal(
        result.state.grounded_world.update_words,
        _words(0, 0),
    )
    assert bool(result.diagnostics.committed_post_child_clocks_aligned)


def test_integrated_exact_counter_schema_and_resource_delta_are_explicit() -> None:
    agent = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(replacement_interval=0)
    )
    start, _ = _start_and_transition(agent, seed=90_060)
    budget = agent.resource_budget(start.state)
    measured = sum(
        int(getattr(leaf, "nbytes", 0))
        for leaf in jax.tree_util.tree_leaves(start.state)
    )

    assert INTEGRATED_HIDDEN_PARTNER_SCHEMA_VERSION.endswith(".v16")
    assert INTEGRATED_HIDDEN_PARTNER_LIFETIME_COUNTER_NBYTES == 12
    assert INTEGRATED_HIDDEN_PARTNER_LIFETIME_COUNTER_DELTA_NBYTES == 8
    assert int(start.state.step_count.nbytes) == 4
    assert int(start.state.step_words.nbytes) == 8
    assert budget.integrated_transition_counter_nbytes == 12
    assert budget.total_state_nbytes == measured
    assert budget.decision_cache_nbytes >= 12


def test_integrated_exact_clock_contract_is_exported_once() -> None:
    import alberta_framework.core as core

    names = (
        "INTEGRATED_CHILD_CLOCK_ALIGNMENT_ORDER",
        "INTEGRATED_HIDDEN_PARTNER_LIFETIME_COUNTER_DELTA_NBYTES",
        "INTEGRATED_HIDDEN_PARTNER_LIFETIME_COUNTER_NBYTES",
        "INTEGRATED_HIDDEN_PARTNER_SCHEMA_VERSION",
    )
    assert all(hasattr(core, name) for name in names)
    assert all(core.__all__.count(name) == 1 for name in names)
