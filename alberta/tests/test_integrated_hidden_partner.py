"""Focused L0 mechanism tests for the integrated hidden-partner kernel.

L0 is the lowest rung of this repo's evidence ladder (``RESEARCH_STATUS.md``):
API, shape, finite-value, serialization, and local-update contracts — never
learning or performance claims.  The kernel under test
(:mod:`alberta_framework.core.integrated_hidden_partner`) composes one
explicit causal transition per step: learned state -> bounded pair discovery
-> partner prediction -> joint-outcome planning -> differential SARSA.  Its
central design property, enforced throughout this suite, is that every
ablation is *shape-matched*: disabled paths still compute their update (then
discard it), so resource accounting, array shapes, and RNG advancement are
identical across arms and eager/jit/scan execution.

Tests are deliberately white-box and comment-light — the long test names
state each contract; fixtures pin the frozen dimension constants
(``RAW_OBSERVATION_DIM``/``HIDDEN_STATE_DIM``/``ACTIVE_PAIR_SLOTS`` etc.)
whose values come from the frozen v6 design manifest
(:mod:`alberta_framework.evaluation.hidden_partner_lifecycle_world_v6`).
"""

from __future__ import annotations

import copy
import dataclasses
from types import SimpleNamespace
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework.core.average_reward as average_reward_module
import alberta_framework.core.interaction_features as interaction_features_module
from alberta_framework.core.grounded_joint_world_model import (
    GroundedJointWorldModelConfig,
)
from alberta_framework.core.integrated_hidden_partner import (
    ACTIVE_PAIR_SLOTS,
    BASE_FEATURE_DIM,
    CANDIDATE_PAIR_SLOTS,
    DEPLOYED_FEATURE_DIM,
    HIDDEN_STATE_DIM,
    INITIAL_ACTIVE_DESCRIPTORS,
    INTEGRATED_HIDDEN_PARTNER_SCHEMA_VERSION,
    RAW_OBSERVATION_DIM,
    IntegratedHiddenPartnerAgent,
    IntegratedHiddenPartnerConfig,
)
from alberta_framework.core.interaction_features import (
    RELEVANCE_PROBE_MODE_CONDITIONAL_V1,
    RELEVANCE_PROBE_MODE_TARGET_ONLY_V1,
)
from alberta_framework.core.representation_gradient_mixer import (
    RepresentationGradientMixerConfig,
)
from alberta_framework.streams.hidden_partner_mapping import (
    HiddenPartnerMappingConfig,
    HiddenPartnerMappingTransition,
    HiddenPartnerMappingWorld,
)

# ~6 min serial (eager/jit/scan parity arms compile repeatedly); keep out of
# the fast per-PR CI lane (-m "not slow").
pytestmark = pytest.mark.slow

# Same values as V6_INITIAL_ACTIVE_DESCRIPTORS in the frozen v6 design
# manifest (evaluation.hidden_partner_lifecycle_world_v6).  Deliberately
# different from the kernel's default INITIAL_ACTIVE_DESCRIPTORS, so the
# custom-bank round-trip and routing tests exercise a genuinely nondefault
# descriptor bank.
V6_TEST_ACTIVE_DESCRIPTORS: tuple[tuple[int, int], ...] = (
    (0, 4),
    (0, 5),
    (1, 4),
    (1, 5),
    (1, 6),
    (1, 7),
    (2, 4),
    (2, 5),
    (2, 6),
    (2, 7),
    (4, 6),
    (5, 7),
)


def _environment() -> HiddenPartnerMappingWorld:
    return HiddenPartnerMappingWorld(
        HiddenPartnerMappingConfig(
            base_segment_lengths=(4,) * 9,
            jitter_radius=0,
            partner_flip_probability=0.0,
        )
    )


def _grounded_integrated_config(
    *,
    mode: str = "full",
    grounded_planning: bool = False,
    **overrides: Any,
) -> IntegratedHiddenPartnerConfig:
    values: dict[str, Any] = {
        "grounded_world_model": GroundedJointWorldModelConfig(
            representation_dim=DEPLOYED_FEATURE_DIM,
            target_observation_dim=RAW_OBSERVATION_DIM,
            n_focal_actions=2,
            n_partner_actions=2,
            step_size=0.2,
            initialization_scale=0.05,
            max_input_magnitude=100.0,
            max_parameter_magnitude=100.0,
        ),
        "representation_gradient_mixer": RepresentationGradientMixerConfig(
            representation_dim=DEPLOYED_FEATURE_DIM,
            mode=mode,  # type: ignore[arg-type]
        ),
        "grounded_world_planning_enabled": grounded_planning,
        "feature_lifecycle_enabled": False,
        "replacement_interval": 0,
    }
    values.update(overrides)
    return IntegratedHiddenPartnerConfig(**values)


def _start_and_transition(
    agent: IntegratedHiddenPartnerAgent,
    *,
    seed: int = 0,
) -> tuple[
    Any,
    HiddenPartnerMappingTransition,
    Any,
]:
    environment = _environment()
    environment_state = environment.init(jr.key(seed))
    start = agent.start(
        environment.observe(environment_state),
        jr.key(seed + 10_000),
    )
    transition, next_environment_state = environment.step(
        environment_state,
        start.action,
    )
    return start, transition, next_environment_state


def _tree_array_nbytes(tree: object) -> int:
    return sum(int(getattr(leaf, "nbytes", 0)) for leaf in jax.tree_util.tree_leaves(tree))


def _assert_current_q_value_delta(
    agent: IntegratedHiddenPartnerAgent,
    state: Any,
) -> None:
    expected = agent.control_agent.q_values(state.control, state.chi) - (
        state.current_evaluation.q_values
    )
    chex.assert_trees_all_equal(state.current_q_value_delta, expected)


def _stack_state_trees(states: tuple[Any, ...]) -> Any:
    return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *states)


def _assert_cache_mutations_reject_in_eager_jit_and_scan(
    agent: IntegratedHiddenPartnerAgent,
    transition: Any,
    states: tuple[Any, ...],
) -> None:
    for corrupted in states:
        eager = agent.update(corrupted, transition)
        chex.assert_trees_all_equal(eager.state, corrupted)
        assert bool(eager.diagnostics.transition_rejected)
        assert not bool(eager.diagnostics.all_finite)
        assert int(eager.diagnostics.integrated_step_delta) == 0

    batched_states = _stack_state_trees(states)

    def update_summary(state: Any) -> tuple[Any, Any, Any, Any]:
        result = agent.update(state, transition)
        return (
            result.state,
            result.diagnostics.transition_rejected,
            result.diagnostics.all_finite,
            result.diagnostics.integrated_step_delta,
        )

    compiled_states, compiled_rejected, compiled_finite, compiled_delta = jax.jit(
        jax.vmap(update_summary)
    )(batched_states)
    chex.assert_trees_all_equal(compiled_states, batched_states)
    chex.assert_trees_all_equal(
        compiled_rejected,
        jnp.ones((len(states),), dtype=jnp.bool_),
    )
    chex.assert_trees_all_equal(
        compiled_finite,
        jnp.zeros((len(states),), dtype=jnp.bool_),
    )
    chex.assert_trees_all_equal(
        compiled_delta,
        jnp.zeros((len(states),), dtype=jnp.int32),
    )

    def scan_step(carry: Any, corrupted: Any) -> tuple[Any, tuple[Any, Any, Any, Any]]:
        result = agent.update(corrupted, transition)
        return carry, (
            result.state,
            result.diagnostics.transition_rejected,
            result.diagnostics.all_finite,
            result.diagnostics.integrated_step_delta,
        )

    _, (scan_states, scan_rejected, scan_finite, scan_delta) = jax.jit(
        lambda inputs: jax.lax.scan(
            scan_step,
            jnp.asarray(0, dtype=jnp.int32),
            inputs,
        )
    )(batched_states)
    chex.assert_trees_all_equal(scan_states, batched_states)
    chex.assert_trees_all_equal(scan_rejected, compiled_rejected)
    chex.assert_trees_all_equal(scan_finite, compiled_finite)
    chex.assert_trees_all_equal(scan_delta, compiled_delta)


def _force_next_interaction_promotion(
    state: Any,
    pair: tuple[int, int] = (4, 5),
) -> Any:
    """Make one unique native archive identity replace active slot zero at step 64."""
    interaction = state.interaction
    matching = (interaction.candidate_left == pair[0]) & (
        interaction.candidate_right == pair[1]
    )
    assert int(jnp.sum(matching)) == 1
    candidate_index = int(jnp.argmax(matching))
    candidate_utilities = (
        jnp.zeros(
            (CANDIDATE_PAIR_SLOTS,),
            dtype=jnp.float32,
        )
        .at[candidate_index]
        .set(10.0)
    )
    return state.replace(
        interaction=interaction.replace(
            step_count=jnp.asarray(63, dtype=jnp.int32),
            ages=jnp.full(
                (ACTIVE_PAIR_SLOTS,),
                256,
                dtype=jnp.int32,
            ),
            utilities=jnp.zeros(
                (ACTIVE_PAIR_SLOTS,),
                dtype=jnp.float32,
            ),
            candidate_ages=jnp.full(
                (CANDIDATE_PAIR_SLOTS,),
                128,
                dtype=jnp.int32,
            ),
            candidate_utilities=candidate_utilities,
        )
    )


def test_config_round_trip_and_exact_default_composition() -> None:
    config = IntegratedHiddenPartnerConfig()
    agent = IntegratedHiddenPartnerAgent(config)
    restored = IntegratedHiddenPartnerConfig.from_config(config.to_config())

    assert restored == config
    assert config.initial_active_descriptors == INITIAL_ACTIVE_DESCRIPTORS
    assert config.replacement_interval == 64
    assert config.min_feature_age == 256
    assert config.candidate_min_age == 128
    assert config.state_step_size == pytest.approx(0.005)
    assert config.state_gradient_clip == pytest.approx(5.0)
    assert config.planner_lambda == pytest.approx(2.0)
    assert config.q_step_size == pytest.approx(0.03)
    assert config.average_reward_step_size == pytest.approx(0.003)
    assert config.active_utility_retention_decay == pytest.approx(0.9999)
    assert config.active_utility_retention_grace_steps is None
    assert config.active_utility_evidence_threshold == 0.0
    assert not config.evidence_gated_feature_memory
    assert config.feature_evidence_confirmation_steps == 1
    assert not config.independent_relevance_probe
    assert config.relevance_probe_mode == RELEVANCE_PROBE_MODE_CONDITIONAL_V1
    assert not config.evidence_gated_consumer_memory
    assert config.consumer_evidence_confirmation_steps == 1
    assert config.consumer_read_confirmation_steps == 1
    assert config.consumer_read_lease_steps == 32
    assert not config.retire_stale_features
    assert config.candidate_promotion_floor == 0.0
    assert config.candidate_promotion_confirmation_steps == 1
    assert config.candidate_reacquisition_confirmation_steps == 1
    assert config.candidate_utility_retention_decay == pytest.approx(0.9995)
    assert agent.state_builder.feature_dim() == BASE_FEATURE_DIM == 12
    assert agent.interaction_learner.n_features == ACTIVE_PAIR_SLOTS == 12
    assert agent.interaction_learner.n_tasks == 1
    assert CANDIDATE_PAIR_SLOTS == 66
    assert agent.interaction_learner.to_config()["candidate_strategy"] == "all_pairs"
    assert agent.interaction_learner.to_config()["refresh_candidates"] is False
    assert agent.interaction_learner.to_config()["refresh_promoted_candidate"] is False
    assert agent.interaction_learner.to_config()["utility_retention_decay"] == pytest.approx(0.9999)
    assert agent.interaction_learner.to_config()["utility_retention_grace_steps"] is None
    assert agent.interaction_learner.to_config()["evidence_gated_active_output_memory"] is False
    assert agent.interaction_learner.to_config()["utility_evidence_confirmation_steps"] == 1
    assert agent.interaction_learner.to_config()["independent_relevance_probe"] is False
    assert (
        agent.interaction_learner.to_config()["relevance_probe_mode"]
        == RELEVANCE_PROBE_MODE_CONDITIONAL_V1
    )
    assert agent.interaction_learner.to_config()["retire_stale_features"] is False
    assert agent.interaction_learner.to_config()["candidate_promotion_confirmation_steps"] == 1
    assert agent.interaction_learner.to_config()["candidate_reacquisition_confirmation_steps"] == 1
    assert agent.behavior_model.config.n_actions == 2
    assert agent.joint_world_model.resource_budget.joint_cells == 4
    assert agent.control_agent.config.n_actions == 2
    assert not agent.control_agent.config.use_bias
    assert agent.router.config.total_feature_dim == DEPLOYED_FEATURE_DIM == 24

    payload = config.to_config()
    assert payload["schema_version"] == "alberta.integrated-hidden-partner.l0.v15"
    assert payload["schema_version"] == INTEGRATED_HIDDEN_PARTNER_SCHEMA_VERSION
    assert payload["development_level"] == "L0"
    assert payload["accepted_scientific_evidence"] is False
    assert payload["initial_active_descriptors"] == [
        list(pair) for pair in INITIAL_ACTIVE_DESCRIPTORS
    ]
    extra = copy.deepcopy(payload)
    extra["promoted_evidence"] = True
    with pytest.raises(ValueError, match="v15 schema"):
        IntegratedHiddenPartnerConfig.from_config(extra)
    invalid_claim = copy.deepcopy(payload)
    invalid_claim["accepted_scientific_evidence"] = True
    with pytest.raises(ValueError, match="not accepted"):
        IntegratedHiddenPartnerConfig.from_config(invalid_claim)
    old_schema = copy.deepcopy(payload)
    old_schema["schema_version"] = "alberta.integrated-hidden-partner.l0.v14"
    with pytest.raises(ValueError, match="unsupported"):
        IntegratedHiddenPartnerConfig.from_config(old_schema)
    assert payload["action_selection_mode"] == "agent"
    forced_payload = IntegratedHiddenPartnerConfig(
        action_selection_mode="externally_forced"
    ).to_config()
    assert (
        IntegratedHiddenPartnerConfig.from_config(forced_payload).action_selection_mode
        == "externally_forced"
    )
    missing_mode = copy.deepcopy(payload)
    missing_mode.pop("action_selection_mode")
    with pytest.raises(ValueError, match="v15 schema"):
        IntegratedHiddenPartnerConfig.from_config(missing_mode)
    for invalid_mode in (None, 1, "forced"):
        with pytest.raises(ValueError, match="action_selection_mode"):
            IntegratedHiddenPartnerConfig(action_selection_mode=invalid_mode)  # type: ignore[arg-type]


def test_start_binds_exact_pre_td_q_provenance_and_control_invariants() -> None:
    agent = IntegratedHiddenPartnerAgent()
    start, _, _ = _start_and_transition(agent, seed=8_400)
    state = start.state

    chex.assert_shape(state.current_q_value_delta, (2,))
    assert state.current_q_value_delta.dtype == jnp.float32
    chex.assert_trees_all_equal(
        state.current_q_value_delta,
        jnp.zeros((2,), dtype=jnp.float32),
    )
    _assert_current_q_value_delta(agent, state)
    chex.assert_trees_all_equal(state.control.last_observation, state.chi)
    chex.assert_trees_all_equal(
        state.control.epsilon,
        jnp.asarray(agent.config.epsilon, dtype=jnp.float32),
    )
    chex.assert_trees_all_equal(
        state.control.q_bias,
        jnp.zeros((2,), dtype=jnp.float32),
    )
    chex.assert_trees_all_equal(
        state.control.q_trace_bias,
        jnp.zeros((2,), dtype=jnp.float32),
    )
    assert int(state.control.step_count) == int(state.step_count) == 0


def test_externally_forced_start_is_strict_and_preserves_policy_rng_primitives() -> None:
    policy = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(action_selection_mode="agent")
    )
    forced = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(action_selection_mode="externally_forced")
    )
    environment = _environment()
    world_state = environment.init(jr.key(8_401))
    raw = environment.observe(world_state)
    key = jr.key(18_401)
    policy_start = policy.start(raw, key)
    forced_action = jnp.asarray(1 - int(policy_start.action), dtype=jnp.int32)

    with pytest.raises(ValueError, match="externally_forced"):
        forced.start(raw, key)
    with pytest.raises(ValueError, match="agent"):
        policy.start_with_forced_action(raw, key, forced_action)
    forced_start = forced.start_with_forced_action(raw, key, forced_action)

    assert int(forced_start.action) == int(forced_action)
    assert int(forced_start.state.control.last_action) == int(forced_action)
    assert bool(forced_start.state.current_selection.externally_forced)
    assert not bool(policy_start.state.current_selection.externally_forced)
    for field in (
        "noisy_greedy_action",
        "random_action",
        "explored",
        "rng_key_before",
        "rng_key_after",
    ):
        chex.assert_trees_all_equal(
            getattr(policy_start.state.current_selection, field),
            getattr(forced_start.state.current_selection, field),
        )
    _assert_current_q_value_delta(forced, forced_start.state)

    for invalid in (-1, 2):
        with pytest.raises(ValueError, match="forced_action"):
            forced.start_with_forced_action(
                raw,
                key,
                jnp.asarray(invalid, dtype=jnp.int32),
            )
    with pytest.raises(ValueError, match="shape"):
        forced.start_with_forced_action(
            raw,
            key,
            jnp.asarray([0], dtype=jnp.int32),
        )
    with pytest.raises(TypeError, match="dtype"):
        forced.start_with_forced_action(
            raw,
            key,
            jnp.asarray(0.0, dtype=jnp.float32),
        )


def test_forced_update_replaces_only_applied_action_and_is_fail_closed() -> None:
    policy = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(action_selection_mode="agent")
    )
    forced = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(action_selection_mode="externally_forced")
    )
    environment = _environment()
    world_state = environment.init(jr.key(8_402))
    raw = environment.observe(world_state)
    key = jr.key(18_402)
    policy_start = policy.start(raw, key)
    forced_start = forced.start_with_forced_action(raw, key, policy_start.action)
    transition, _ = environment.step(world_state, policy_start.action)
    policy_result = policy.update(policy_start.state, transition)
    forced_next_action = jnp.asarray(1 - int(policy_result.action), dtype=jnp.int32)
    forced_result = forced.update_with_forced_next_action(
        forced_start.state,
        transition,
        forced_next_action,
    )

    chex.assert_trees_all_equal(
        forced_result.diagnostics.next_evaluation,
        policy_result.diagnostics.next_evaluation,
    )
    for field in (
        "noisy_greedy_action",
        "random_action",
        "explored",
        "rng_key_before",
        "rng_key_after",
    ):
        chex.assert_trees_all_equal(
            getattr(forced_result.state.current_selection, field),
            getattr(policy_result.state.current_selection, field),
        )
    assert int(forced_result.action) == int(forced_next_action)
    assert int(forced_result.state.control.last_action) == int(forced_next_action)
    assert bool(forced_result.state.current_selection.externally_forced)
    assert not bool(policy_result.state.current_selection.externally_forced)
    _assert_current_q_value_delta(forced, forced_result.state)

    with pytest.raises(ValueError, match="externally_forced"):
        forced.update(forced_start.state, transition)
    with pytest.raises(ValueError, match="agent"):
        policy.update_with_forced_next_action(
            policy_start.state,
            transition,
            jnp.asarray(0, dtype=jnp.int32),
        )
    with pytest.raises(ValueError, match="shape"):
        forced.update_with_forced_next_action(
            forced_start.state,
            transition,
            jnp.asarray([0], dtype=jnp.int32),
        )
    with pytest.raises(TypeError, match="dtype"):
        forced.update_with_forced_next_action(
            forced_start.state,
            transition,
            jnp.asarray(0.0, dtype=jnp.float32),
        )

    invalid = forced.update_with_forced_next_action(
        forced_start.state,
        transition,
        jnp.asarray(2, dtype=jnp.int32),
    )
    chex.assert_trees_all_equal(invalid.state, forced_start.state)
    assert int(invalid.action) == int(forced_start.action)
    assert bool(invalid.diagnostics.transition_rejected)
    assert not bool(invalid.diagnostics.all_finite)
    assert int(invalid.diagnostics.integrated_step_delta) == 0


def test_forced_actions_scan_and_closed_form_td_are_causal() -> None:
    agent = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(
            action_selection_mode="externally_forced",
            feature_lifecycle_enabled=False,
            replacement_interval=0,
            trace_decay=0.5,
        )
    )
    environment = _environment()
    world_state = environment.init(jr.key(8_403))
    preview, _ = environment.step(
        world_state,
        jnp.asarray(0, dtype=jnp.int32),
    )
    initial_action = preview.partner_action
    start = agent.start_with_forced_action(
        environment.observe(world_state),
        jr.key(18_403),
        initial_action,
    )
    first_transition, second_world_state = environment.step(
        world_state,
        start.action,
    )
    assert float(first_transition.reward) == 1.0
    warm = agent.update_with_forced_next_action(
        start.state,
        first_transition,
        jnp.asarray(0, dtype=jnp.int32),
    )
    second_transition, _ = environment.step(second_world_state, warm.action)
    branch_zero = agent.update_with_forced_next_action(
        warm.state,
        second_transition,
        jnp.asarray(0, dtype=jnp.int32),
    )
    branch_one = agent.update_with_forced_next_action(
        warm.state,
        second_transition,
        jnp.asarray(1, dtype=jnp.int32),
    )
    chex.assert_trees_all_equal(
        branch_zero.diagnostics.next_evaluation,
        branch_one.diagnostics.next_evaluation,
    )
    q_next = branch_zero.diagnostics.next_evaluation.q_values
    assert float(q_next[0]) != float(q_next[1])
    assert float(
        branch_zero.diagnostics.td_error - branch_one.diagnostics.td_error
    ) == pytest.approx(float(q_next[0] - q_next[1]), abs=1e-7)
    assert int(branch_zero.action) == 0
    assert int(branch_one.action) == 1

    forced_actions = jnp.asarray([0, 1, 0], dtype=jnp.int32)

    def scan_step(carry: Any, forced_action: Any) -> tuple[Any, Any]:
        agent_state, environment_state = carry
        transition, next_environment_state = environment.step(
            environment_state,
            agent_state.control.last_action,
        )
        fresh_current_q = agent.control_agent.q_values(
            agent_state.control,
            agent_state.chi,
        )
        update = agent.update_with_forced_next_action(
            agent_state,
            transition,
            forced_action,
        )
        expected_td_error = (
            transition.reward
            - agent_state.control.average_reward
            + transition.discount
            * update.diagnostics.next_evaluation.q_values[forced_action]
            - fresh_current_q[agent_state.control.last_action]
        )
        return (update.state, next_environment_state), (
            update.action,
            update.diagnostics.next_selection.externally_forced,
            update.diagnostics.td_error,
            expected_td_error,
            update.diagnostics.all_finite,
        )

    (final_state, _), outputs = jax.jit(
        lambda state, world: jax.lax.scan(
            scan_step,
            (state, world),
            forced_actions,
        )
    )(start.state, world_state)
    actions, externally_forced, td_errors, expected_td_errors, finite = outputs

    chex.assert_trees_all_equal(actions, forced_actions)
    chex.assert_trees_all_equal(
        externally_forced,
        jnp.ones((3,), dtype=jnp.bool_),
    )
    chex.assert_trees_all_close(td_errors, expected_td_errors, atol=0.0, rtol=0.0)
    assert bool(jnp.all(finite))
    assert int(final_state.step_count) == 3
    assert int(final_state.control.last_action) == 0
    _assert_current_q_value_delta(agent, final_state)

    invalid_actions = jnp.asarray([0, 2], dtype=jnp.int32)

    def invalid_scan_step(carry: Any, forced_action: Any) -> tuple[Any, Any]:
        agent_state, environment_state = carry
        transition, next_environment_state = environment.step(
            environment_state,
            agent_state.control.last_action,
        )
        update = agent.update_with_forced_next_action(
            agent_state,
            transition,
            forced_action,
        )
        return (
            update.state,
            next_environment_state,
        ), update.diagnostics.integrated_step_delta

    (invalid_final, _), deltas = jax.jit(
        lambda state, world: jax.lax.scan(
            invalid_scan_step,
            (state, world),
            invalid_actions,
        )
    )(start.state, world_state)
    assert int(deltas[0]) == 1
    assert int(deltas[1]) == 0
    assert int(invalid_final.step_count) == 1


def test_custom_initial_descriptor_config_roundtrips_ordered_json_lists() -> None:
    config = IntegratedHiddenPartnerConfig(
        initial_active_descriptors=V6_TEST_ACTIVE_DESCRIPTORS,
    )
    payload = config.to_config()
    restored = IntegratedHiddenPartnerConfig.from_config(payload)
    agent = IntegratedHiddenPartnerAgent(restored)

    assert payload["initial_active_descriptors"] == [
        list(pair) for pair in V6_TEST_ACTIVE_DESCRIPTORS
    ]
    assert restored.initial_active_descriptors == V6_TEST_ACTIVE_DESCRIPTORS
    agent_payload = agent.to_config()
    assert agent_payload["initial_active_descriptors"] == [
        list(pair) for pair in V6_TEST_ACTIVE_DESCRIPTORS
    ]
    assert agent_payload["config"]["initial_active_descriptors"] == [
        list(pair) for pair in V6_TEST_ACTIVE_DESCRIPTORS
    ]

    reordered = (
        V6_TEST_ACTIVE_DESCRIPTORS[1],
        V6_TEST_ACTIVE_DESCRIPTORS[0],
        *V6_TEST_ACTIVE_DESCRIPTORS[2:],
    )
    reordered_payload = IntegratedHiddenPartnerConfig(
        initial_active_descriptors=reordered,
    ).to_config()
    assert (
        IntegratedHiddenPartnerConfig.from_config(reordered_payload).initial_active_descriptors
        == reordered
    )

    non_json = copy.deepcopy(payload)
    non_json["initial_active_descriptors"] = V6_TEST_ACTIVE_DESCRIPTORS
    with pytest.raises(ValueError, match="ordered JSON lists"):
        IntegratedHiddenPartnerConfig.from_config(non_json)


def test_random_curation_accepts_zero_utility_decay_as_selection_only() -> None:
    config = IntegratedHiddenPartnerConfig(
        random_feature_curation=True,
        interaction_utility_decay=0.0,
    )

    assert IntegratedHiddenPartnerConfig.from_config(config.to_config()) == config


@pytest.mark.parametrize(
    "descriptors",
    [
        list(V6_TEST_ACTIVE_DESCRIPTORS),
        V6_TEST_ACTIVE_DESCRIPTORS[:-1],
        V6_TEST_ACTIVE_DESCRIPTORS + ((8, 11),),
        (list(V6_TEST_ACTIVE_DESCRIPTORS[0]),) + V6_TEST_ACTIVE_DESCRIPTORS[1:],
        ((False, 4),) + V6_TEST_ACTIVE_DESCRIPTORS[1:],
        ((0.0, 4),) + V6_TEST_ACTIVE_DESCRIPTORS[1:],
        ((-1, -1),) + V6_TEST_ACTIVE_DESCRIPTORS[1:],
        ((4, 4),) + V6_TEST_ACTIVE_DESCRIPTORS[1:],
        ((5, 4),) + V6_TEST_ACTIVE_DESCRIPTORS[1:],
        ((0, BASE_FEATURE_DIM),) + V6_TEST_ACTIVE_DESCRIPTORS[1:],
        (V6_TEST_ACTIVE_DESCRIPTORS[1],) + V6_TEST_ACTIVE_DESCRIPTORS[1:],
    ],
)
def test_custom_initial_descriptor_config_rejects_nonexact_banks(
    descriptors: Any,
) -> None:
    with pytest.raises(ValueError, match="initial_active_descriptors"):
        IntegratedHiddenPartnerConfig(initial_active_descriptors=descriptors)


def test_grounded_learning_disable_requires_an_enabled_grounded_model() -> None:
    with pytest.raises(ValueError, match="grounded_world_model"):
        IntegratedHiddenPartnerConfig(grounded_world_learning_enabled=False)


def test_reacquisition_confirmation_config_reaches_interaction_learner() -> None:
    config = IntegratedHiddenPartnerConfig(
        evidence_gated_feature_memory=True,
        independent_relevance_probe=True,
        evidence_gated_consumer_memory=True,
        active_utility_retention_grace_steps=8,
        active_utility_evidence_threshold=0.01,
        retire_stale_features=True,
        candidate_promotion_floor=0.01,
        candidate_reacquisition_confirmation_steps=4,
    )
    restored = IntegratedHiddenPartnerConfig.from_config(config.to_config())
    agent = IntegratedHiddenPartnerAgent(restored)

    assert restored.candidate_reacquisition_confirmation_steps == 4
    assert agent.interaction_learner.to_config()["candidate_reacquisition_confirmation_steps"] == 4


def test_lifecycle_freeze_accepts_the_full_evidence_gated_configuration() -> None:
    config = IntegratedHiddenPartnerConfig(
        feature_lifecycle_enabled=False,
        evidence_gated_feature_memory=True,
        feature_evidence_confirmation_steps=2,
        independent_relevance_probe=True,
        evidence_gated_consumer_memory=True,
        consumer_evidence_confirmation_steps=2,
        consumer_read_confirmation_steps=2,
        active_utility_retention_grace_steps=8,
        active_utility_evidence_threshold=0.01,
        retire_stale_features=True,
        candidate_promotion_floor=0.01,
        candidate_promotion_confirmation_steps=2,
        candidate_reacquisition_confirmation_steps=4,
    )
    restored = IntegratedHiddenPartnerConfig.from_config(config.to_config())
    agent = IntegratedHiddenPartnerAgent(restored)

    assert not restored.feature_lifecycle_enabled
    assert restored.evidence_gated_feature_memory
    assert restored.evidence_gated_consumer_memory
    assert restored.independent_relevance_probe
    assert agent.interaction_learner.to_config()["retire_stale_features"] is True


def test_reacquisition_confirmation_is_a_matched_no_retirement_control() -> None:
    retiring = IntegratedHiddenPartnerConfig(
        evidence_gated_feature_memory=True,
        independent_relevance_probe=True,
        evidence_gated_consumer_memory=True,
        active_utility_retention_grace_steps=8,
        active_utility_evidence_threshold=0.01,
        retire_stale_features=True,
        candidate_promotion_floor=0.01,
        candidate_reacquisition_confirmation_steps=8,
    )
    no_retirement = dataclasses.replace(retiring, retire_stale_features=False)
    restored = IntegratedHiddenPartnerConfig.from_config(no_retirement.to_config())
    agent = IntegratedHiddenPartnerAgent(restored)
    interaction = agent.interaction_learner.init(BASE_FEATURE_DIM, jr.key(8812))

    assert not restored.retire_stale_features
    assert restored.candidate_reacquisition_confirmation_steps == 8
    assert agent.interaction_learner.to_config()["retire_stale_features"] is False
    assert agent.interaction_learner.to_config()["candidate_reacquisition_confirmation_steps"] == 8
    chex.assert_trees_all_equal(
        interaction.candidate_reacquisition_required,
        jnp.zeros((CANDIDATE_PAIR_SLOTS,), dtype=jnp.bool_),
    )


def test_versioned_probe_mode_round_trip_reaches_agent_with_resource_parity() -> None:
    common = {
        "evidence_gated_feature_memory": True,
        "independent_relevance_probe": True,
        "evidence_gated_consumer_memory": True,
        "active_utility_retention_grace_steps": 8,
        "active_utility_evidence_threshold": 0.01,
    }
    conditional = IntegratedHiddenPartnerConfig(
        **common,
        relevance_probe_mode=RELEVANCE_PROBE_MODE_CONDITIONAL_V1,
    )
    target_only = IntegratedHiddenPartnerConfig(
        **common,
        relevance_probe_mode=RELEVANCE_PROBE_MODE_TARGET_ONLY_V1,
    )
    restored = IntegratedHiddenPartnerConfig.from_config(target_only.to_config())
    conditional_agent = IntegratedHiddenPartnerAgent(conditional)
    target_agent = IntegratedHiddenPartnerAgent(restored)
    conditional_start, _, _ = _start_and_transition(conditional_agent, seed=701)
    target_start, _, _ = _start_and_transition(target_agent, seed=701)

    assert restored.relevance_probe_mode == RELEVANCE_PROBE_MODE_TARGET_ONLY_V1
    assert (
        target_agent.interaction_learner.to_config()["relevance_probe_mode"]
        == RELEVANCE_PROBE_MODE_TARGET_ONLY_V1
    )
    chex.assert_trees_all_equal(
        conditional_start.state.interaction,
        target_start.state.interaction,
    )
    assert (
        conditional_agent.resource_budget(conditional_start.state).total_state_nbytes
        == target_agent.resource_budget(target_start.state).total_state_nbytes
    )

    unknown = target_only.to_config()
    unknown["relevance_probe_mode"] = "unknown_v1"
    with pytest.raises(ValueError, match="relevance_probe_mode"):
        IntegratedHiddenPartnerConfig.from_config(unknown)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"planning_enabled": 1},
        {"state_learning_enabled": "yes"},
        {"grounded_world_learning_enabled": 1},
        {"uniform_partner_belief": 1},
        {"random_feature_curation": "yes"},
        {"evidence_gated_feature_memory": "yes"},
        {"independent_relevance_probe": "yes"},
        {"relevance_probe_mode": "unknown_v1"},
        {"feature_evidence_confirmation_steps": True},
        {"feature_evidence_confirmation_steps": -1},
        {"evidence_gated_consumer_memory": "yes"},
        {"consumer_evidence_confirmation_steps": True},
        {"consumer_evidence_confirmation_steps": -1},
        {"consumer_read_confirmation_steps": True},
        {"consumer_read_confirmation_steps": -1},
        {"consumer_read_lease_steps": True},
        {"consumer_read_lease_steps": 2**31 - 1},
        {"planner_lambda": float("nan")},
        {"planner_lambda": True},
        {"state_step_size": 0.0},
        {"state_step_size": True},
        {"interaction_utility_decay": 1.0},
        {"interaction_utility_decay": True},
        {"active_utility_retention_decay": 0.9},
        {"active_utility_retention_decay": True},
        {
            "active_utility_retention_grace_steps": 4,
            "active_utility_evidence_threshold": 0.0,
        },
        {
            "retire_stale_features": True,
            "candidate_promotion_floor": 0.1,
        },
        {
            "active_utility_retention_grace_steps": 4,
            "active_utility_evidence_threshold": 0.01,
            "retire_stale_features": True,
            "candidate_promotion_floor": 0.0,
        },
        {"evidence_gated_consumer_memory": True},
        {"evidence_gated_feature_memory": True},
        {"independent_relevance_probe": True},
        {
            "independent_relevance_probe": True,
            "evidence_gated_feature_memory": True,
            "active_utility_retention_grace_steps": 4,
            "active_utility_evidence_threshold": 0.01,
        },
        {
            "independent_relevance_probe": True,
            "evidence_gated_consumer_memory": True,
            "active_utility_retention_grace_steps": 4,
            "active_utility_evidence_threshold": 0.01,
        },
        {
            "evidence_gated_feature_memory": True,
            "feature_evidence_confirmation_steps": 0,
            "active_utility_retention_grace_steps": 4,
            "active_utility_evidence_threshold": 0.01,
        },
        {
            "evidence_gated_consumer_memory": True,
            "consumer_evidence_confirmation_steps": 0,
            "active_utility_retention_grace_steps": 4,
            "active_utility_evidence_threshold": 0.01,
        },
        {
            "evidence_gated_consumer_memory": True,
            "consumer_evidence_confirmation_steps": 2,
            "consumer_read_confirmation_steps": 0,
            "active_utility_retention_grace_steps": 4,
            "active_utility_evidence_threshold": 0.01,
        },
        {
            "evidence_gated_consumer_memory": True,
            "consumer_evidence_confirmation_steps": 2,
            "consumer_read_confirmation_steps": 3,
            "active_utility_retention_grace_steps": 4,
            "active_utility_evidence_threshold": 0.01,
        },
        {
            "evidence_gated_consumer_memory": True,
            "consumer_read_lease_steps": 0,
            "active_utility_retention_grace_steps": 4,
            "active_utility_evidence_threshold": 0.01,
        },
        {"candidate_utility_retention_decay": 0.9},
        {"candidate_promotion_confirmation_steps": True},
        {"candidate_promotion_confirmation_steps": 0},
        {"candidate_promotion_confirmation_steps": 2**31 - 1},
        {"candidate_promotion_confirmation_steps": 1.5},
        {"candidate_reacquisition_confirmation_steps": True},
        {"candidate_reacquisition_confirmation_steps": 0},
        {"candidate_reacquisition_confirmation_steps": 2**31 - 1},
        {"candidate_reacquisition_confirmation_steps": 1.5},
        {"candidate_reacquisition_confirmation_steps": 2},
        {"replacement_interval": True},
        {"replacement_interval": -1},
        {"min_feature_age": 2**31 - 1},
        {"q_step_size": True},
        {"world_step_size": 1.01},
        {"trace_decay": 1.1},
        {"trace_decay": True},
        {"epsilon": float("inf")},
    ],
)
def test_config_rejects_invalid_static_controls(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        IntegratedHiddenPartnerConfig(**kwargs)


def test_pair_product_gradient_chain_rule_matches_autodiff() -> None:
    agent = IntegratedHiddenPartnerAgent()
    phi = jnp.linspace(-1.2, 1.4, BASE_FEATURE_DIM, dtype=jnp.float32)
    descriptors = jnp.asarray(INITIAL_ACTIVE_DESCRIPTORS, dtype=jnp.int32)
    chi_gradient = jnp.linspace(
        -0.7,
        0.9,
        DEPLOYED_FEATURE_DIM,
        dtype=jnp.float32,
    )

    expected = jax.grad(
        lambda value: jnp.vdot(
            agent.build_chi(value, descriptors),
            chi_gradient,
        )
    )(phi)
    actual = agent.chain_chi_gradient_to_phi(
        phi,
        descriptors,
        chi_gradient,
    )

    chex.assert_trees_all_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert float(jnp.linalg.norm(actual)) > 0.0


def test_consumer_read_gate_masks_pair_values_and_exact_chain_credit() -> None:
    agent = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(
            evidence_gated_consumer_memory=True,
            active_utility_retention_grace_steps=32,
            active_utility_evidence_threshold=0.01,
        )
    )
    phi = jnp.linspace(-1.2, 1.4, BASE_FEATURE_DIM, dtype=jnp.float32)
    descriptors = jnp.asarray(INITIAL_ACTIVE_DESCRIPTORS, dtype=jnp.int32)
    mask = (jnp.arange(ACTIVE_PAIR_SLOTS) % 3) == 0
    chi_gradient = jnp.linspace(
        -0.7,
        0.9,
        DEPLOYED_FEATURE_DIM,
        dtype=jnp.float32,
    )

    chi = agent.build_chi(phi, descriptors, mask)
    expected_gradient = jax.grad(
        lambda value: jnp.vdot(
            agent.build_chi(value, descriptors, mask),
            chi_gradient,
        )
    )(phi)
    actual_gradient = agent.chain_chi_gradient_to_phi(
        phi,
        descriptors,
        chi_gradient,
        mask,
    )

    chex.assert_trees_all_equal(chi[:BASE_FEATURE_DIM], phi)
    chex.assert_trees_all_equal(
        chi[BASE_FEATURE_DIM:][~mask],
        jnp.zeros_like(chi[BASE_FEATURE_DIM:][~mask]),
    )
    chex.assert_trees_all_close(
        actual_gradient,
        expected_gradient,
        atol=1e-6,
        rtol=1e-6,
    )


def test_consumer_write_gate_commits_only_evidenced_behavior_and_control_columns() -> None:
    agent = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(
            evidence_gated_consumer_memory=True,
            active_utility_retention_grace_steps=32,
            active_utility_evidence_threshold=0.01,
        )
    )
    start, _, _ = _start_and_transition(agent, seed=43)
    state = start.state
    gate = jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.bool_).at[1].set(True).at[7].set(True)

    behavior_previous = state.behavior.replace(
        weights=jnp.arange(
            2 * DEPLOYED_FEATURE_DIM,
            dtype=jnp.float32,
        ).reshape((2, DEPLOYED_FEATURE_DIM))
    )
    behavior_proposed = behavior_previous.replace(
        weights=behavior_previous.weights + 100.0,
        bias=behavior_previous.bias + 2.0,
        step_count=behavior_previous.step_count + 1,
        nll_ema=jnp.asarray(3.0, dtype=jnp.float32),
    )
    behavior_committed = agent._commit_behavior_consumer_update(  # noqa: SLF001
        behavior_previous,
        behavior_proposed,
        gate,
    )

    chex.assert_trees_all_equal(
        behavior_committed.weights[:, :BASE_FEATURE_DIM],
        behavior_proposed.weights[:, :BASE_FEATURE_DIM],
    )
    chex.assert_trees_all_equal(
        behavior_committed.weights[:, BASE_FEATURE_DIM:][:, gate],
        behavior_proposed.weights[:, BASE_FEATURE_DIM:][:, gate],
    )
    chex.assert_trees_all_equal(
        behavior_committed.weights[:, BASE_FEATURE_DIM:][:, ~gate],
        behavior_previous.weights[:, BASE_FEATURE_DIM:][:, ~gate],
    )
    chex.assert_trees_all_equal(behavior_committed.bias, behavior_proposed.bias)
    chex.assert_trees_all_equal(
        behavior_committed.step_count,
        behavior_proposed.step_count,
    )
    chex.assert_trees_all_equal(behavior_committed.nll_ema, behavior_proposed.nll_ema)

    control_previous = state.control.replace(
        q_weights=jnp.arange(
            2 * DEPLOYED_FEATURE_DIM,
            dtype=jnp.float32,
        ).reshape((2, DEPLOYED_FEATURE_DIM)),
        q_trace_weights=jnp.full(
            (2, DEPLOYED_FEATURE_DIM),
            4.0,
            dtype=jnp.float32,
        ),
    )
    control_proposed = control_previous.replace(
        q_weights=control_previous.q_weights + 200.0,
        q_trace_weights=control_previous.q_trace_weights + 300.0,
        average_reward=jnp.asarray(0.75, dtype=jnp.float32),
        last_observation=jnp.full(
            (DEPLOYED_FEATURE_DIM,),
            5.0,
            dtype=jnp.float32,
        ),
        step_count=control_previous.step_count + 1,
    )
    control_committed = agent._commit_control_consumer_update(  # noqa: SLF001
        control_previous,
        control_proposed,
        gate,
    )

    chex.assert_trees_all_equal(
        control_committed.q_weights[:, :BASE_FEATURE_DIM],
        control_proposed.q_weights[:, :BASE_FEATURE_DIM],
    )
    chex.assert_trees_all_equal(
        control_committed.q_weights[:, BASE_FEATURE_DIM:][:, gate],
        control_proposed.q_weights[:, BASE_FEATURE_DIM:][:, gate],
    )
    chex.assert_trees_all_equal(
        control_committed.q_weights[:, BASE_FEATURE_DIM:][:, ~gate],
        control_previous.q_weights[:, BASE_FEATURE_DIM:][:, ~gate],
    )
    chex.assert_trees_all_equal(
        control_committed.q_trace_weights[:, BASE_FEATURE_DIM:][:, gate],
        control_proposed.q_trace_weights[:, BASE_FEATURE_DIM:][:, gate],
    )
    chex.assert_trees_all_equal(
        control_committed.q_trace_weights[:, BASE_FEATURE_DIM:][:, ~gate],
        jnp.zeros_like(control_committed.q_trace_weights[:, BASE_FEATURE_DIM:][:, ~gate]),
    )
    chex.assert_trees_all_equal(
        control_committed.average_reward,
        control_proposed.average_reward,
    )
    chex.assert_trees_all_equal(
        control_committed.last_observation,
        control_proposed.last_observation,
    )
    chex.assert_trees_all_equal(control_committed.step_count, control_proposed.step_count)


def test_read_lease_alone_never_authorizes_feature_column_overwrite() -> None:
    agent = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(
            evidence_gated_consumer_memory=True,
            active_utility_retention_grace_steps=32,
            active_utility_evidence_threshold=0.01,
        )
    )
    start, _, _ = _start_and_transition(agent, seed=49)
    state = start.state
    read_lease = jnp.ones((ACTIVE_PAIR_SLOTS,), dtype=jnp.bool_)
    confirmed_write = jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.bool_)
    assert bool(jnp.all(read_lease))

    behavior_proposed = state.behavior.replace(
        weights=state.behavior.weights + 3.0,
    )
    behavior_committed = agent._commit_behavior_consumer_update(  # noqa: SLF001
        state.behavior,
        behavior_proposed,
        confirmed_write,
    )
    chex.assert_trees_all_equal(
        behavior_committed.weights[:, BASE_FEATURE_DIM:],
        state.behavior.weights[:, BASE_FEATURE_DIM:],
    )

    control_previous = state.control.replace(
        q_trace_weights=jnp.ones_like(state.control.q_trace_weights),
    )
    control_proposed = control_previous.replace(
        q_weights=control_previous.q_weights + 5.0,
        q_trace_weights=control_previous.q_trace_weights + 7.0,
    )
    control_committed = agent._commit_control_consumer_update(  # noqa: SLF001
        control_previous,
        control_proposed,
        confirmed_write,
    )
    chex.assert_trees_all_equal(
        control_committed.q_weights[:, BASE_FEATURE_DIM:],
        control_previous.q_weights[:, BASE_FEATURE_DIM:],
    )
    chex.assert_trees_all_equal(
        control_committed.q_trace_weights[:, BASE_FEATURE_DIM:],
        jnp.zeros((2, ACTIVE_PAIR_SLOTS), dtype=jnp.float32),
    )


def test_consumer_gate_routes_streak_write_and_read_by_identity() -> None:
    agent = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(
            evidence_gated_consumer_memory=True,
            active_utility_retention_grace_steps=32,
            active_utility_evidence_threshold=0.01,
        )
    )
    start, _, _ = _start_and_transition(agent, seed=44)
    old = start.state.router.descriptors
    proposed = (
        old.at[0].set(old[2]).at[1].set(old[0]).at[2].set(jnp.asarray([4, 5], dtype=jnp.int32))
    )
    _, _, route, _ = agent._route_feature_consumers(  # noqa: SLF001
        start.state.router,
        start.state.behavior,
        start.state.control,
        proposed,
    )
    updated_streak = jnp.arange(ACTIVE_PAIR_SLOTS, dtype=jnp.int32) + 1
    confirmed_write = (jnp.arange(ACTIVE_PAIR_SLOTS) % 2) == 0
    read_acquire = (jnp.arange(ACTIVE_PAIR_SLOTS) % 4) == 1
    previous_read = (jnp.arange(ACTIVE_PAIR_SLOTS) % 3) == 0
    evidence_idle_post = jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.int32).at[0].set(33)
    routed_streak = agent._route_consumer_evidence_streak(  # noqa: SLF001
        updated_streak,
        route,
    )
    routed_write = agent._route_consumer_confirmed_write(  # noqa: SLF001
        confirmed_write,
        route,
    )
    routed_acquire = agent._route_consumer_read_acquire(  # noqa: SLF001
        read_acquire,
        route,
    )
    routed_read = agent._route_consumer_active_mask(  # noqa: SLF001
        previous_read,
        read_acquire,
        route,
        evidence_idle_post,
    )

    source = np.asarray([2, 0, -1] + list(range(3, ACTIVE_PAIR_SLOTS)))
    survivors = source >= 0
    expected_streak = np.zeros((ACTIVE_PAIR_SLOTS,), dtype=np.int32)
    expected_streak[survivors] = np.asarray(updated_streak)[source[survivors]]
    expected_write = np.zeros((ACTIVE_PAIR_SLOTS,), dtype=np.bool_)
    expected_write[survivors] = np.asarray(confirmed_write)[source[survivors]]
    expected_acquire = np.zeros((ACTIVE_PAIR_SLOTS,), dtype=np.bool_)
    expected_acquire[survivors] = np.asarray(read_acquire)[source[survivors]]
    acquired = np.asarray(previous_read | read_acquire)
    expected_read = np.zeros((ACTIVE_PAIR_SLOTS,), dtype=np.bool_)
    expected_read[survivors] = acquired[source[survivors]]
    expected_read[0] = False
    np.testing.assert_array_equal(route.source_slots, source)
    np.testing.assert_array_equal(routed_streak, expected_streak)
    np.testing.assert_array_equal(routed_write, expected_write)
    np.testing.assert_array_equal(routed_acquire, expected_acquire)
    np.testing.assert_array_equal(routed_read, expected_read)
    assert int(routed_streak[2]) == 0
    assert not bool(routed_write[2])
    assert not bool(routed_acquire[2])
    assert not bool(routed_read[2])


def test_consumer_confirmation_and_read_lease_have_distinct_timing() -> None:
    agent = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(
            evidence_gated_consumer_memory=True,
            consumer_evidence_confirmation_steps=2,
            consumer_read_confirmation_steps=1,
            consumer_read_lease_steps=2,
            active_utility_retention_grace_steps=32,
            active_utility_evidence_threshold=0.01,
        )
    )
    start, _, _ = _start_and_transition(agent, seed=48)
    _, _, route, _ = agent._route_feature_consumers(  # noqa: SLF001
        start.state.router,
        start.state.behavior,
        start.state.control,
        start.state.router.descriptors,
    )
    slot = 4
    streak = jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.int32)
    read_mask = jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.bool_)

    def advance(
        old_streak: Any,
        old_read_mask: Any,
        *,
        has_evidence: bool,
        idle_steps: int,
    ) -> tuple[Any, Any, Any, Any]:
        evidence = jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.bool_).at[slot].set(has_evidence)
        updated, read_acquire_pre, write_pre = agent._update_consumer_evidence_streak(  # noqa: SLF001
            old_streak,
            evidence,
        )
        streak_post = agent._route_consumer_evidence_streak(  # noqa: SLF001
            updated,
            route,
        )
        write_post = agent._route_consumer_confirmed_write(  # noqa: SLF001
            write_pre,
            route,
        )
        read_acquire_post = agent._route_consumer_read_acquire(  # noqa: SLF001
            read_acquire_pre,
            route,
        )
        idle_post = jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.int32).at[slot].set(idle_steps)
        read_post = agent._route_consumer_active_mask(  # noqa: SLF001
            old_read_mask,
            read_acquire_pre,
            route,
            idle_post,
        )
        return streak_post, read_acquire_post, write_post, read_post

    streak, read_acquire, write, read_mask = advance(
        streak,
        read_mask,
        has_evidence=True,
        idle_steps=0,
    )
    assert int(streak[slot]) == 1
    assert bool(read_acquire[slot])
    assert not bool(write[slot])
    assert bool(read_mask[slot])

    (
        saturated,
        saturated_read_acquire,
        saturated_write,
    ) = agent._update_consumer_evidence_streak(  # noqa: SLF001
        jnp.full((ACTIVE_PAIR_SLOTS,), 2**31 - 1, dtype=jnp.int32),
        jnp.ones((ACTIVE_PAIR_SLOTS,), dtype=jnp.bool_),
    )
    chex.assert_trees_all_equal(
        saturated,
        jnp.full((ACTIVE_PAIR_SLOTS,), 2**31 - 1, dtype=jnp.int32),
    )
    assert bool(jnp.all(saturated_read_acquire))
    assert bool(jnp.all(saturated_write))

    streak, read_acquire, write, read_mask = advance(
        streak,
        read_mask,
        has_evidence=True,
        idle_steps=0,
    )
    assert int(streak[slot]) == 2
    assert bool(read_acquire[slot])
    assert bool(write[slot])
    assert bool(read_mask[slot])

    for idle_steps in (1, 2):
        streak, read_acquire, write, read_mask = advance(
            streak,
            read_mask,
            has_evidence=False,
            idle_steps=idle_steps,
        )
        assert int(streak[slot]) == 0
        assert not bool(read_acquire[slot])
        assert not bool(write[slot])
        assert bool(read_mask[slot])

    streak, read_acquire, write, read_mask = advance(
        streak,
        read_mask,
        has_evidence=False,
        idle_steps=3,
    )
    assert not bool(read_acquire[slot])
    assert not bool(read_mask[slot])

    streak, read_acquire, write, read_mask = advance(
        streak,
        read_mask,
        has_evidence=True,
        idle_steps=0,
    )
    assert int(streak[slot]) == 1
    assert bool(read_acquire[slot])
    assert not bool(write[slot])
    assert bool(read_mask[slot])


def test_disabled_consumer_gate_preserves_unconditional_update_contract() -> None:
    agent = IntegratedHiddenPartnerAgent()
    start, _, _ = _start_and_transition(agent, seed=45)
    state = start.state
    false_gate = jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.bool_)
    behavior_proposed = state.behavior.replace(
        weights=state.behavior.weights + 1.0,
        step_count=state.behavior.step_count + 1,
    )
    control_proposed = state.control.replace(
        q_weights=state.control.q_weights + 2.0,
        q_trace_weights=state.control.q_trace_weights + 3.0,
        step_count=state.control.step_count + 1,
    )

    chex.assert_trees_all_equal(
        agent._consumer_write_gate(false_gate),  # noqa: SLF001
        jnp.ones((ACTIVE_PAIR_SLOTS,), dtype=jnp.bool_),
    )
    (
        disabled_streak,
        disabled_read_acquire,
        disabled_write,
    ) = agent._update_consumer_evidence_streak(  # noqa: SLF001
        jnp.full((ACTIVE_PAIR_SLOTS,), 7, dtype=jnp.int32),
        false_gate,
    )
    chex.assert_trees_all_equal(
        disabled_streak,
        jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.int32),
    )
    chex.assert_trees_all_equal(
        disabled_read_acquire,
        jnp.ones((ACTIVE_PAIR_SLOTS,), dtype=jnp.bool_),
    )
    chex.assert_trees_all_equal(
        disabled_write,
        jnp.ones((ACTIVE_PAIR_SLOTS,), dtype=jnp.bool_),
    )
    chex.assert_trees_all_equal(
        agent._commit_behavior_consumer_update(  # noqa: SLF001
            state.behavior,
            behavior_proposed,
            false_gate,
        ),
        behavior_proposed,
    )
    chex.assert_trees_all_equal(
        agent._commit_control_consumer_update(  # noqa: SLF001
            state.control,
            control_proposed,
            false_gate,
        ),
        control_proposed,
    )
    chex.assert_trees_all_equal(
        state.chi,
        agent.build_chi(state.phi, state.router.descriptors),
    )
    chex.assert_trees_all_equal(
        state.consumer_evidence_streak,
        jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.int32),
    )


def test_consumer_gate_initial_state_and_budget_include_exact_streak_bytes() -> None:
    agent = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(
            evidence_gated_consumer_memory=True,
            active_utility_retention_grace_steps=32,
            active_utility_evidence_threshold=0.01,
        )
    )
    start, transition, _ = _start_and_transition(agent, seed=46)
    budget = agent.resource_budget(start.state)

    assert start.state.consumer_active_mask.shape == (ACTIVE_PAIR_SLOTS,)
    assert start.state.consumer_active_mask.dtype == jnp.bool_
    assert not bool(jnp.any(start.state.consumer_active_mask))
    assert start.state.consumer_evidence_streak.shape == (ACTIVE_PAIR_SLOTS,)
    assert start.state.consumer_evidence_streak.dtype == jnp.int32
    chex.assert_trees_all_equal(
        start.state.consumer_evidence_streak,
        jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.int32),
    )
    chex.assert_trees_all_equal(
        start.state.chi[BASE_FEATURE_DIM:],
        jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.float32),
    )
    assert budget.consumer_active_mask_nbytes == ACTIVE_PAIR_SLOTS == 12
    assert budget.consumer_evidence_streak_nbytes == 4 * ACTIVE_PAIR_SLOTS == 48
    assert budget.total_state_nbytes == _tree_array_nbytes(start.state)

    result = agent.update(start.state, transition)
    chex.assert_trees_all_equal(
        result.diagnostics.consumer_active_mask_pre,
        start.state.consumer_active_mask,
    )
    chex.assert_trees_all_equal(
        result.diagnostics.consumer_read_mask_pre,
        result.diagnostics.consumer_active_mask_pre,
    )
    chex.assert_trees_all_equal(
        result.diagnostics.consumer_evidence_streak_pre,
        start.state.consumer_evidence_streak,
    )
    chex.assert_trees_all_equal(
        result.diagnostics.consumer_write_gate_pre,
        result.diagnostics.consumer_confirmed_write_pre,
    )
    chex.assert_trees_all_equal(
        result.diagnostics.consumer_confirmed_write_pre,
        result.diagnostics.interaction_evidence_refreshed,
    )
    chex.assert_trees_all_equal(
        result.diagnostics.consumer_read_acquire_pre,
        result.diagnostics.interaction_evidence_refreshed,
    )
    chex.assert_trees_all_equal(
        result.diagnostics.consumer_confirmed_write_post,
        agent._route_consumer_confirmed_write(  # noqa: SLF001
            result.diagnostics.consumer_confirmed_write_pre,
            result.diagnostics.route,
        ),
    )
    chex.assert_trees_all_equal(
        result.diagnostics.consumer_read_acquire_post,
        agent._route_consumer_read_acquire(  # noqa: SLF001
            result.diagnostics.consumer_read_acquire_pre,
            result.diagnostics.route,
        ),
    )
    chex.assert_trees_all_equal(
        result.diagnostics.consumer_active_mask_post,
        result.state.consumer_active_mask,
    )
    chex.assert_trees_all_equal(
        result.diagnostics.consumer_read_mask_post,
        result.diagnostics.consumer_active_mask_post,
    )
    chex.assert_trees_all_equal(
        result.diagnostics.consumer_evidence_streak_post,
        result.state.consumer_evidence_streak,
    )
    assert agent.resource_budget(result.state).total_state_nbytes == _tree_array_nbytes(
        result.state
    )


def test_integrated_transition_acquires_read_before_write_confirmation() -> None:
    agent = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(
            evidence_gated_consumer_memory=True,
            consumer_evidence_confirmation_steps=2,
            consumer_read_confirmation_steps=1,
            active_utility_retention_grace_steps=32,
            active_utility_evidence_threshold=0.01,
            trace_decay=0.9,
        )
    )
    start, transition, _ = _start_and_transition(agent, seed=47)
    descriptors = start.state.router.descriptors
    products = start.state.phi[descriptors[:, 0]] * start.state.phi[descriptors[:, 1]]
    evidenced_slot = int(jnp.argmax(jnp.abs(products)))
    partner_sign = 2.0 * float(transition.partner_action) - 1.0
    aligned_weight = partner_sign * 0.25 / float(products[evidenced_slot])
    interaction = start.state.interaction.replace(
        output_weights=start.state.interaction.output_weights.at[
            0,
            evidenced_slot,
        ].set(aligned_weight)
    )
    control = start.state.control.replace(
        q_trace_weights=jnp.full_like(
            start.state.control.q_trace_weights,
            9.0,
        )
    )

    result = agent.update(
        start.state.replace(
            interaction=interaction,
            control=control,
        ),
        transition,
    )
    expected = jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.bool_).at[evidenced_slot].set(True)

    chex.assert_trees_all_equal(
        result.diagnostics.interaction_evidence_refreshed,
        expected,
    )
    chex.assert_trees_all_equal(
        result.diagnostics.consumer_read_acquire_pre,
        expected,
    )
    chex.assert_trees_all_equal(
        result.diagnostics.consumer_confirmed_write_pre,
        jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.bool_),
    )
    assert int(result.state.consumer_evidence_streak[evidenced_slot]) == 1
    chex.assert_trees_all_equal(result.state.consumer_active_mask, expected)
    assert float(jnp.abs(result.state.chi[BASE_FEATURE_DIM + evidenced_slot])) > 0.0
    chex.assert_trees_all_equal(
        result.state.chi[BASE_FEATURE_DIM:][~expected],
        jnp.zeros((ACTIVE_PAIR_SLOTS - 1,), dtype=jnp.float32),
    )
    chex.assert_trees_all_equal(
        result.state.behavior.weights[:, BASE_FEATURE_DIM:],
        start.state.behavior.weights[:, BASE_FEATURE_DIM:],
    )
    chex.assert_trees_all_equal(
        result.state.control.q_weights[:, BASE_FEATURE_DIM:],
        start.state.control.q_weights[:, BASE_FEATURE_DIM:],
    )
    chex.assert_trees_all_equal(
        result.state.control.q_trace_weights[:, BASE_FEATURE_DIM:],
        jnp.zeros((2, ACTIVE_PAIR_SLOTS), dtype=jnp.float32),
    )
    _assert_current_q_value_delta(agent, result.state)


def test_integrated_idle_read_lease_cannot_write_behavior_or_q_columns() -> None:
    agent = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(
            evidence_gated_consumer_memory=True,
            consumer_evidence_confirmation_steps=2,
            consumer_read_lease_steps=2,
            active_utility_retention_grace_steps=32,
            active_utility_evidence_threshold=0.01,
            trace_decay=0.9,
        )
    )
    start, transition, _ = _start_and_transition(agent, seed=50)
    slot = 0
    read_mask = jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.bool_).at[slot].set(True)
    streak = jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.int32).at[slot].set(2)
    chi = agent.build_chi(
        start.state.phi,
        start.state.router.descriptors,
        read_mask,
    )
    control = start.state.control.replace(
        average_reward=jnp.asarray(-1.0, dtype=jnp.float32),
        last_observation=chi,
        q_trace_weights=jnp.full_like(start.state.control.q_trace_weights, 9.0),
    )
    current_evaluation = agent.evaluate_models(
        start.state.behavior,
        start.state.joint_world,
        control,
        chi,
    )
    prepared = start.state.replace(
        chi=chi,
        consumer_active_mask=read_mask,
        consumer_evidence_streak=streak,
        control=control,
        current_evaluation=current_evaluation,
    )

    result = agent.update(prepared, transition)
    no_writes = jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.bool_)

    chex.assert_trees_all_equal(
        result.diagnostics.interaction_evidence_refreshed,
        no_writes,
    )
    chex.assert_trees_all_equal(
        result.diagnostics.consumer_confirmed_write_pre,
        no_writes,
    )
    assert int(result.state.consumer_evidence_streak[slot]) == 0
    assert bool(result.state.consumer_active_mask[slot])
    assert int(result.state.interaction.evidence_idle_steps[slot]) == 1
    assert float(jnp.abs(result.state.chi[BASE_FEATURE_DIM + slot])) > 0.0
    chex.assert_trees_all_equal(
        result.state.behavior.weights[:, BASE_FEATURE_DIM:],
        prepared.behavior.weights[:, BASE_FEATURE_DIM:],
    )
    chex.assert_trees_all_equal(
        result.state.control.q_weights[:, BASE_FEATURE_DIM:],
        prepared.control.q_weights[:, BASE_FEATURE_DIM:],
    )
    chex.assert_trees_all_equal(
        result.state.control.q_trace_weights[:, BASE_FEATURE_DIM:],
        jnp.zeros((2, ACTIVE_PAIR_SLOTS), dtype=jnp.float32),
    )


def test_active_utility_retention_has_a_shape_matched_normal_decay_ablation() -> None:
    retained = IntegratedHiddenPartnerAgent()
    normal_decay = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(active_utility_retention_decay=None)
    )
    retained_state = retained.interaction_learner.init(
        BASE_FEATURE_DIM,
        jr.key(19),
    ).replace(utilities=jnp.ones((ACTIVE_PAIR_SLOTS,), dtype=jnp.float32))
    normal_state = normal_decay.interaction_learner.init(
        BASE_FEATURE_DIM,
        jr.key(19),
    ).replace(utilities=jnp.ones((ACTIVE_PAIR_SLOTS,), dtype=jnp.float32))
    observation = jnp.zeros((BASE_FEATURE_DIM,), dtype=jnp.float32)
    target = jnp.zeros((1,), dtype=jnp.float32)

    retained_update = retained.interaction_learner.update(
        retained_state,
        observation,
        target,
    )
    normal_update = normal_decay.interaction_learner.update(
        normal_state,
        observation,
        target,
    )

    chex.assert_trees_all_close(
        retained_update.state.utilities,
        jnp.full((ACTIVE_PAIR_SLOTS,), 0.9999, dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        normal_update.state.utilities,
        jnp.full((ACTIVE_PAIR_SLOTS,), 0.995, dtype=jnp.float32),
    )
    assert (
        retained.interaction_learner.memory_accounting(retained_state)["persistent_array_bytes"]
        == normal_decay.interaction_learner.memory_accounting(normal_state)[
            "persistent_array_bytes"
        ]
    )


def test_evidence_lease_has_exact_shape_matched_48_byte_counters() -> None:
    historical = IntegratedHiddenPartnerAgent()
    leased = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(
            active_utility_retention_grace_steps=4_096,
            active_utility_evidence_threshold=0.01,
            retire_stale_features=True,
            candidate_promotion_floor=0.01,
        )
    )
    historical_start, _, _ = _start_and_transition(historical, seed=41)
    leased_start, _, _ = _start_and_transition(leased, seed=41)
    historical_budget = historical.resource_budget(historical_start.state)
    leased_budget = leased.resource_budget(leased_start.state)

    assert historical_budget.total_state_nbytes == leased_budget.total_state_nbytes
    assert historical_budget.interaction_evidence_idle_nbytes == 48
    assert leased_budget.interaction_evidence_idle_nbytes == 48
    assert (
        leased.interaction_learner.memory_accounting(leased_start.state.interaction)[
            "evidence_idle_step_bytes"
        ]
        == 48
    )


def test_confirmed_feature_memory_is_shape_matched_and_exposes_causal_diagnostics() -> None:
    default = IntegratedHiddenPartnerAgent()
    confirmed = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(
            evidence_gated_feature_memory=True,
            feature_evidence_confirmation_steps=2,
            active_utility_retention_grace_steps=32,
            active_utility_evidence_threshold=0.01,
        )
    )
    default_start, _, _ = _start_and_transition(default, seed=56)
    start, transition, _ = _start_and_transition(confirmed, seed=56)
    default_budget = default.resource_budget(default_start.state)
    confirmed_budget = confirmed.resource_budget(start.state)

    assert default_budget.total_state_nbytes == confirmed_budget.total_state_nbytes
    assert confirmed_budget.interaction_utility_evidence_streak_nbytes == 48
    assert confirmed_budget.interaction_active_output_memory_committed_nbytes == 12
    assert confirmed_budget.interaction_relevance_probe_nbytes == 52
    assert confirmed_budget.interaction_relevance_probe_bias_nbytes == 4
    assert (
        confirmed_budget.interaction_candidate_promotion_evidence_streak_nbytes
        == 4 * CANDIDATE_PAIR_SLOTS
        == 264
    )
    assert (
        confirmed_budget.interaction_candidate_reacquisition_required_nbytes
        == CANDIDATE_PAIR_SLOTS
        == 66
    )
    assert (
        confirmed.interaction_learner.memory_accounting(start.state.interaction)[
            "utility_evidence_streak_bytes"
        ]
        == 48
    )
    assert (
        confirmed.interaction_learner.memory_accounting(start.state.interaction)[
            "active_output_memory_committed_bytes"
        ]
        == 12
    )
    assert (
        confirmed.interaction_learner.memory_accounting(start.state.interaction)[
            "candidate_promotion_evidence_streak_bytes"
        ]
        == 264
    )
    assert (
        confirmed.interaction_learner.memory_accounting(start.state.interaction)[
            "candidate_reacquisition_required_bytes"
        ]
        == 66
    )

    descriptors = start.state.router.descriptors
    products = start.state.phi[descriptors[:, 0]] * start.state.phi[descriptors[:, 1]]
    evidenced_slot = int(jnp.argmax(jnp.abs(products)))
    partner_sign = 2.0 * float(transition.partner_action) - 1.0
    aligned_weight = partner_sign * 0.25 / float(products[evidenced_slot])
    interaction = start.state.interaction.replace(
        output_weights=start.state.interaction.output_weights.at[
            0,
            evidenced_slot,
        ].set(aligned_weight),
        utility_evidence_streak=start.state.interaction.utility_evidence_streak.at[
            evidenced_slot
        ].set(1),
        active_output_memory_committed=start.state.interaction.active_output_memory_committed.at[
            evidenced_slot
        ].set(True),
        evidence_idle_steps=start.state.interaction.evidence_idle_steps.at[evidenced_slot].set(7),
    )

    result = confirmed.update(start.state.replace(interaction=interaction), transition)
    expected = jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.bool_).at[evidenced_slot].set(True)

    chex.assert_trees_all_equal(
        result.diagnostics.interaction_evidence_refreshed,
        expected,
    )
    chex.assert_trees_all_equal(
        result.diagnostics.interaction_retention_evidence_refreshed,
        expected,
    )
    assert int(result.state.interaction.utility_evidence_streak[evidenced_slot]) == 2
    assert bool(result.state.interaction.active_output_memory_committed[evidenced_slot])
    assert int(result.state.interaction.evidence_idle_steps[evidenced_slot]) == 0
    assert confirmed.resource_budget(result.state).total_state_nbytes == _tree_array_nbytes(
        result.state
    )


def test_independent_probe_commits_then_reacquires_under_consumer_lease() -> None:
    agent = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(
            evidence_gated_feature_memory=True,
            feature_evidence_confirmation_steps=2,
            independent_relevance_probe=True,
            candidate_promotion_confirmation_steps=3,
            evidence_gated_consumer_memory=True,
            consumer_evidence_confirmation_steps=2,
            consumer_read_confirmation_steps=1,
            consumer_read_lease_steps=2,
            active_utility_retention_grace_steps=32,
            active_utility_evidence_threshold=0.01,
        )
    )
    start, transition, next_environment_state = _start_and_transition(agent, seed=67)
    assert agent.interaction_learner.to_config()["candidate_promotion_confirmation_steps"] == 3
    slot = 0
    left, right = INITIAL_ACTIVE_DESCRIPTORS[slot]
    product = float(start.state.phi[left] * start.state.phi[right])
    partner_sign = 2.0 * float(transition.partner_action) - 1.0
    marginal_residual = partner_sign - float(start.state.interaction.relevance_probe_biases[0])
    preupdate_probe = 0.5 * marginal_residual / product
    interaction = start.state.interaction.replace(
        output_weights=start.state.interaction.output_weights.at[0, slot].set(-0.0),
        relevance_probe_weights=(
            start.state.interaction.relevance_probe_weights.at[0, slot].set(preupdate_probe)
        ),
    )
    first = agent.update(start.state.replace(interaction=interaction), transition)

    assert bool(first.diagnostics.interaction_evidence_refreshed[slot])
    assert not bool(first.diagnostics.interaction_retention_evidence_refreshed[slot])
    assert not bool(first.state.interaction.active_output_memory_committed[slot])
    assert bool(first.state.consumer_active_mask[slot])
    chex.assert_trees_all_equal(
        first.state.chi[BASE_FEATURE_DIM:],
        jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.float32),
    )
    assert int(np.asarray(first.state.interaction.output_weights[0, slot]).view(np.uint32)) == 0

    environment = _environment()
    second_transition, _ = environment.step(next_environment_state, first.action)
    second_product = float(first.state.phi[left] * first.state.phi[right])
    second_sign = 2.0 * float(second_transition.partner_action) - 1.0
    second_residual = second_sign - float(first.state.interaction.relevance_probe_biases[0])
    second_preupdate_probe = 0.5 * second_residual / second_product
    second_interaction = first.state.interaction.replace(
        relevance_probe_weights=(
            first.state.interaction.relevance_probe_weights.at[0, slot].set(second_preupdate_probe)
        )
    )
    second = agent.update(
        first.state.replace(interaction=second_interaction),
        second_transition,
    )

    assert bool(second.diagnostics.interaction_retention_evidence_refreshed[slot])
    assert bool(second.state.interaction.active_output_memory_committed[slot])
    chex.assert_trees_all_equal(
        second.diagnostics.interaction_relevance_probe_biases_pre,
        first.state.interaction.relevance_probe_biases,
    )
    chex.assert_trees_all_equal(
        second.diagnostics.interaction_relevance_probe_biases_post,
        second.state.interaction.relevance_probe_biases,
    )
    chex.assert_shape(
        second.diagnostics.interaction_candidate_promotion_signal,
        (CANDIDATE_PAIR_SLOTS,),
    )
    chex.assert_trees_all_equal(
        second.diagnostics.interaction_candidate_promotion_evidence_streak_post,
        second.state.interaction.candidate_promotion_evidence_streak,
    )
    chex.assert_trees_all_equal(
        second.diagnostics.interaction_candidate_reacquisition_required_post,
        second.state.interaction.candidate_reacquisition_required,
    )
    chex.assert_shape(
        second.diagnostics.interaction_candidate_reacquisition_required_pre,
        (CANDIDATE_PAIR_SLOTS,),
    )
    chex.assert_shape(
        second.diagnostics.interaction_candidate_reacquisition_confirmed,
        (CANDIDATE_PAIR_SLOTS,),
    )
    assert second.state.interaction.candidate_promotion_evidence_streak.dtype == jnp.int32
    assert bool(
        jnp.all(second.diagnostics.interaction_candidate_promotion_evidence_streak_updated <= 3)
    )
    assert float(second.state.interaction.output_weights[0, slot]) == pytest.approx(
        second_preupdate_probe
    )
    assert bool(second.diagnostics.consumer_confirmed_write_pre[slot])
    expected_next_product = second.state.phi[left] * second.state.phi[right]
    assert float(second.state.chi[BASE_FEATURE_DIM + slot]) == pytest.approx(
        float(expected_next_product)
    )

    recurrent_start, recurrent_transition, _ = _start_and_transition(agent, seed=68)
    recurrent_product = float(recurrent_start.state.phi[left] * recurrent_start.state.phi[right])
    recurrent_sign = 2.0 * float(recurrent_transition.partner_action) - 1.0
    recurrent_residual = recurrent_sign - float(
        recurrent_start.state.interaction.relevance_probe_biases[0]
    )
    recurrent_probe = 0.5 * recurrent_residual / recurrent_product
    durable = jnp.asarray(7.0, dtype=jnp.float32)
    recurrent_interaction = recurrent_start.state.interaction.replace(
        output_weights=(recurrent_start.state.interaction.output_weights.at[0, slot].set(durable)),
        relevance_probe_weights=(
            recurrent_start.state.interaction.relevance_probe_weights.at[0, slot].set(
                recurrent_probe
            )
        ),
        active_output_memory_committed=(
            recurrent_start.state.interaction.active_output_memory_committed.at[slot].set(True)
        ),
    )
    reacquired = agent.update(
        recurrent_start.state.replace(interaction=recurrent_interaction),
        recurrent_transition,
    )

    assert not bool(reacquired.diagnostics.interaction_durable_read_mask[slot])
    assert bool(reacquired.diagnostics.interaction_evidence_refreshed[slot])
    assert not bool(reacquired.diagnostics.consumer_confirmed_write_pre[slot])
    assert bool(reacquired.diagnostics.consumer_read_acquire_pre[slot])
    assert bool(reacquired.state.consumer_active_mask[slot])
    assert float(reacquired.state.interaction.output_weights[0, slot]) == 7.0
    expected_reacquired_product = reacquired.state.phi[left] * reacquired.state.phi[right]
    assert float(reacquired.state.chi[BASE_FEATURE_DIM + slot]) == pytest.approx(
        float(expected_reacquired_product)
    )
    chex.assert_trees_all_equal(
        reacquired.state.behavior.weights[:, BASE_FEATURE_DIM:],
        recurrent_start.state.behavior.weights[:, BASE_FEATURE_DIM:],
    )
    chex.assert_trees_all_equal(
        reacquired.state.control.q_weights[:, BASE_FEATURE_DIM:],
        recurrent_start.state.control.q_weights[:, BASE_FEATURE_DIM:],
    )
    reacquired_budget = agent.resource_budget(reacquired.state)
    assert reacquired_budget.interaction_relevance_probe_nbytes == 52
    assert reacquired_budget.interaction_relevance_probe_bias_nbytes == 4


def test_combined_feature_and_consumer_gates_keep_raw_read_lease_independent() -> None:
    agent = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(
            evidence_gated_feature_memory=True,
            feature_evidence_confirmation_steps=8,
            evidence_gated_consumer_memory=True,
            consumer_evidence_confirmation_steps=8,
            consumer_read_confirmation_steps=1,
            consumer_read_lease_steps=32,
            active_utility_retention_grace_steps=64,
            active_utility_evidence_threshold=0.01,
        )
    )
    start, transition, _ = _start_and_transition(agent, seed=58)
    descriptors = start.state.router.descriptors
    products = start.state.phi[descriptors[:, 0]] * start.state.phi[descriptors[:, 1]]
    evidenced_slot = int(jnp.argmax(jnp.abs(products)))
    partner_sign = 2.0 * float(transition.partner_action) - 1.0
    aligned_weight = partner_sign * 0.25 / float(products[evidenced_slot])
    interaction = start.state.interaction.replace(
        output_weights=start.state.interaction.output_weights.at[
            0,
            evidenced_slot,
        ].set(aligned_weight),
        active_output_memory_committed=jnp.ones(
            (ACTIVE_PAIR_SLOTS,),
            dtype=jnp.bool_,
        ),
        evidence_idle_steps=jnp.full(
            (ACTIVE_PAIR_SLOTS,),
            40,
            dtype=jnp.int32,
        ),
    )
    prepared = start.state.replace(
        interaction=interaction,
        consumer_read_idle_steps=jnp.full(
            (ACTIVE_PAIR_SLOTS,),
            40,
            dtype=jnp.int32,
        ),
    )

    acquired = agent.update(prepared, transition)
    expected = jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.bool_).at[evidenced_slot].set(True)
    no_confirmed_feature_evidence = jnp.zeros(
        (ACTIVE_PAIR_SLOTS,),
        dtype=jnp.bool_,
    )

    chex.assert_trees_all_equal(
        acquired.diagnostics.interaction_evidence_refreshed,
        expected,
    )
    chex.assert_trees_all_equal(
        acquired.diagnostics.interaction_retention_evidence_refreshed,
        no_confirmed_feature_evidence,
    )
    assert int(acquired.state.interaction.evidence_idle_steps[evidenced_slot]) == 41
    assert int(acquired.state.consumer_read_idle_steps[evidenced_slot]) == 0
    assert bool(acquired.state.consumer_active_mask[evidenced_slot])

    idle = acquired.state.consumer_read_idle_steps
    read_mask = acquired.state.consumer_active_mask
    no_raw_evidence = jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.bool_)
    live = acquired.diagnostics.route.new_validation.live_mask
    route = acquired.diagnostics.route
    for _ in range(32):
        updated_idle = agent._update_consumer_read_idle_steps(  # noqa: SLF001
            idle,
            no_raw_evidence,
            live,
        )
        idle = agent._route_consumer_read_idle_steps(  # noqa: SLF001
            updated_idle,
            route,
        )
        read_mask = agent._route_consumer_active_mask(  # noqa: SLF001
            read_mask,
            no_raw_evidence,
            route,
            idle,
        )

    assert int(idle[evidenced_slot]) == 32
    assert bool(read_mask[evidenced_slot])
    updated_idle = agent._update_consumer_read_idle_steps(  # noqa: SLF001
        idle,
        no_raw_evidence,
        live,
    )
    idle = agent._route_consumer_read_idle_steps(  # noqa: SLF001
        updated_idle,
        route,
    )
    read_mask = agent._route_consumer_active_mask(  # noqa: SLF001
        read_mask,
        no_raw_evidence,
        route,
        idle,
    )
    assert int(idle[evidenced_slot]) == 33
    assert not bool(read_mask[evidenced_slot])


def test_consumer_read_idle_counter_resets_and_stays_zero_for_retired_vacancy() -> None:
    agent = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(
            evidence_gated_feature_memory=True,
            feature_evidence_confirmation_steps=8,
            evidence_gated_consumer_memory=True,
            consumer_evidence_confirmation_steps=8,
            consumer_read_confirmation_steps=1,
            consumer_read_lease_steps=32,
            active_utility_retention_grace_steps=0,
            active_utility_evidence_threshold=0.99,
            retire_stale_features=True,
            candidate_promotion_floor=100.0,
            replacement_interval=1,
            min_feature_age=0,
            candidate_min_age=0,
        )
    )
    environment = _environment()
    environment_state = environment.init(jr.key(59))
    start = agent.start(
        environment.observe(environment_state),
        jr.key(10_059),
    )
    transition, environment_state = environment.step(environment_state, start.action)
    prepared = start.state.replace(
        consumer_read_idle_steps=jnp.full(
            (ACTIVE_PAIR_SLOTS,),
            7,
            dtype=jnp.int32,
        ),
        consumer_active_mask=jnp.ones(
            (ACTIVE_PAIR_SLOTS,),
            dtype=jnp.bool_,
        ),
    )

    retired = agent.update(prepared, transition)

    assert int(retired.diagnostics.interaction_retired_slot) == 0
    assert int(retired.diagnostics.consumer_read_idle_steps_pre[0]) == 7
    assert int(retired.diagnostics.consumer_read_idle_steps_updated_pre[0]) == 8
    assert int(retired.diagnostics.consumer_read_idle_steps_post[0]) == 0
    assert int(retired.state.consumer_read_idle_steps[0]) == 0
    assert not bool(retired.state.consumer_active_mask[0])

    transition, _ = environment.step(environment_state, retired.action)
    vacancy = agent.update(
        retired.state.replace(
            interaction=retired.state.interaction.replace(
                candidate_utilities=jnp.zeros(
                    (CANDIDATE_PAIR_SLOTS,),
                    dtype=jnp.float32,
                )
            )
        ),
        transition,
    )

    assert int(vacancy.diagnostics.consumer_read_idle_steps_pre[0]) == 0
    assert int(vacancy.diagnostics.consumer_read_idle_steps_updated_pre[0]) == 0
    assert int(vacancy.state.consumer_read_idle_steps[0]) == 0


def test_memory_mask_blocks_hidden_deployment_and_recurrent_learning_credit() -> None:
    full = IntegratedHiddenPartnerAgent(IntegratedHiddenPartnerConfig(memory_masked=False))
    masked = IntegratedHiddenPartnerAgent(IntegratedHiddenPartnerConfig(memory_masked=True))
    raw = jnp.asarray(
        [1.0, 0.0, 0.0, 0.0, -1.0, 1.0, -1.0, 1.0],
        dtype=jnp.float32,
    )
    full_start = full.start(raw, jr.key(20))
    masked_start = masked.start(raw, jr.key(20))

    chex.assert_trees_all_equal(
        full_start.state.state_builder,
        masked_start.state.state_builder,
    )
    chex.assert_trees_all_equal(
        full_start.state.phi,
        masked_start.state.phi,
    )
    chex.assert_trees_all_equal(
        masked_start.state.chi[:RAW_OBSERVATION_DIM],
        raw,
    )
    chex.assert_trees_all_equal(
        masked_start.state.chi[RAW_OBSERVATION_DIM:BASE_FEATURE_DIM],
        jnp.zeros((HIDDEN_STATE_DIM,), dtype=jnp.float32),
    )

    gradient = masked.chain_chi_gradient_to_phi(
        masked_start.state.phi,
        masked_start.state.router.descriptors,
        jnp.ones((DEPLOYED_FEATURE_DIM,), dtype=jnp.float32),
    )
    chex.assert_trees_all_equal(
        gradient[-HIDDEN_STATE_DIM:],
        jnp.zeros((HIDDEN_STATE_DIM,), dtype=jnp.float32),
    )
    learned_builder, learning = masked.state_builder.learn(
        masked_start.state.state_builder,
        gradient,
    )
    chex.assert_trees_all_equal(
        learned_builder.parameters,
        masked_start.state.state_builder.parameters,
    )
    assert float(learning.parameter_update_norm) == 0.0
    assert (
        full.resource_budget(full_start.state).total_state_nbytes
        == masked.resource_budget(masked_start.state).total_state_nbytes
    )


def test_initial_active_descriptors_are_deterministic_unique_and_canonical() -> None:
    agent = IntegratedHiddenPartnerAgent()
    raw = jnp.asarray(
        [1.0, 0.0, 0.0, 0.0, -1.0, 1.0, -1.0, 1.0],
        dtype=jnp.float32,
    )
    banks = []
    for seed in range(3):
        start = agent.start(raw, jr.key(seed))
        descriptors = np.asarray(start.state.router.descriptors)
        banks.append(descriptors)
        assert descriptors.shape == (ACTIVE_PAIR_SLOTS, 2)
        assert len({tuple(pair) for pair in descriptors.tolist()}) == ACTIVE_PAIR_SLOTS
        assert np.all(descriptors[:, 0] >= 0)
        assert np.all(descriptors[:, 0] < descriptors[:, 1])
        assert np.all(descriptors[:, 1] < BASE_FEATURE_DIM)
        birth_bank = {tuple(pair) for pair in descriptors.tolist()}
        assert (0, 2) not in birth_bank
        assert (4, 5) not in birth_bank
        assert int(np.sum(descriptors[:, 0] == 0)) == 2
        np.testing.assert_array_equal(
            start.state.interaction.feature_left,
            descriptors[:, 0],
        )
        np.testing.assert_array_equal(
            start.state.interaction.feature_right,
            descriptors[:, 1],
        )
        assert bool(start.diagnostics.descriptors_valid)
    for bank in banks[1:]:
        np.testing.assert_array_equal(bank, banks[0])


def test_custom_initial_bank_starts_router_and_interaction_in_exact_order() -> None:
    custom_agent = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(
            initial_active_descriptors=V6_TEST_ACTIVE_DESCRIPTORS,
        )
    )
    legacy_agent = IntegratedHiddenPartnerAgent()
    environment = _environment()
    environment_state = environment.init(jr.key(8701))
    observation = environment.observe(environment_state)
    key = jr.key(18701)
    custom = custom_agent.start(observation, key)
    legacy = legacy_agent.start(observation, key)
    expected = np.asarray(V6_TEST_ACTIVE_DESCRIPTORS, dtype=np.int32)

    np.testing.assert_array_equal(custom.state.router.descriptors, expected)
    np.testing.assert_array_equal(custom.diagnostics.descriptors, expected)
    np.testing.assert_array_equal(custom.state.interaction.feature_left, expected[:, 0])
    np.testing.assert_array_equal(custom.state.interaction.feature_right, expected[:, 1])
    assert bool(custom.diagnostics.descriptors_valid)

    for pair in ((0, 2), (4, 5)):
        archived = (custom.state.interaction.candidate_left == pair[0]) & (
            custom.state.interaction.candidate_right == pair[1]
        )
        assert int(jnp.sum(archived)) == 1

    custom_budget = custom_agent.resource_budget(custom.state)
    legacy_budget = legacy_agent.resource_budget(legacy.state)
    assert custom_budget.total_state_nbytes == legacy_budget.total_state_nbytes == 6757
    assert jax.tree_util.tree_structure(custom.state) == jax.tree_util.tree_structure(legacy.state)
    chex.assert_trees_all_equal(
        custom.diagnostics.selection.rng_key_before,
        legacy.diagnostics.selection.rng_key_before,
    )
    chex.assert_trees_all_equal(
        custom.diagnostics.selection.rng_key_after,
        legacy.diagnostics.selection.rng_key_after,
    )
    assert int(custom.action) == int(legacy.action)


@pytest.mark.parametrize("pair", ((0, 2), (4, 5)))
def test_custom_initial_bank_routes_a_forced_archive_promotion_by_identity(
    pair: tuple[int, int],
) -> None:
    agent = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(
            initial_active_descriptors=V6_TEST_ACTIVE_DESCRIPTORS,
        )
    )
    start, transition, _ = _start_and_transition(agent, seed=8702)
    prepared = _force_next_interaction_promotion(start.state, pair)
    archive = list(
        zip(
            np.asarray(prepared.interaction.candidate_left).tolist(),
            np.asarray(prepared.interaction.candidate_right).tolist(),
            strict=True,
        )
    )
    assert len(archive) == CANDIDATE_PAIR_SLOTS == len(set(archive))
    assert archive.count(pair) == 1

    result = agent.update(prepared, transition)

    assert not bool(result.diagnostics.transition_rejected)
    assert int(result.diagnostics.interaction_replaced_slot) == 0
    assert int(result.diagnostics.router_generation_delta) == 1
    assert bool(result.diagnostics.route.valid)
    assert bool(result.diagnostics.route.descriptors_changed)
    assert int(result.diagnostics.route.new_count) == 1
    assert int(result.diagnostics.route.evicted_count) == 1
    np.testing.assert_array_equal(result.state.router.descriptors[0], pair)
    np.testing.assert_array_equal(
        result.state.router.descriptors[1:],
        np.asarray(V6_TEST_ACTIVE_DESCRIPTORS[1:], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        result.diagnostics.route.source_slots,
        np.asarray([-1] + list(range(1, ACTIVE_PAIR_SLOTS)), dtype=np.int32),
    )
    result_archive = list(
        zip(
            np.asarray(result.state.interaction.candidate_left).tolist(),
            np.asarray(result.state.interaction.candidate_right).tolist(),
            strict=True,
        )
    )
    assert len(result_archive) == CANDIDATE_PAIR_SLOTS == len(set(result_archive))
    assert result_archive.count(pair) == 1


def test_start_owns_and_advances_sarsa_rng_without_control_update() -> None:
    agent = IntegratedHiddenPartnerAgent()
    start, _, _ = _start_and_transition(agent, seed=2)
    selection = start.diagnostics.selection
    expected_key = jr.split(selection.rng_key_before, 4)[0]

    chex.assert_trees_all_equal(selection.rng_key_after, expected_key)
    chex.assert_trees_all_equal(
        start.state.control.rng_key,
        selection.rng_key_after,
    )
    chex.assert_trees_all_equal(start.state.current_selection, selection)
    assert int(start.action) == int(selection.action)
    assert int(start.state.current_selection.action) == int(start.state.control.last_action)
    assert int(start.state.control.last_action) == int(start.action)
    assert int(start.state.control.step_count) == 0
    assert int(start.state.state_builder.step_count) == 1
    assert int(start.diagnostics.state_advances) == 1
    assert int(start.diagnostics.evaluation.cell_evaluations) == 4
    assert bool(start.diagnostics.all_finite)


def test_repeated_start_has_fresh_array_timing_and_stable_resource_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interaction_times = iter((100.25, 200.25))
    control_times = iter((101.25, 201.25))
    monkeypatch.setattr(
        interaction_features_module,
        "time",
        SimpleNamespace(time=lambda: next(interaction_times)),
    )
    monkeypatch.setattr(
        average_reward_module,
        "time",
        SimpleNamespace(time=lambda: next(control_times)),
    )
    agent = IntegratedHiddenPartnerAgent()
    raw = jnp.asarray(
        [1.0, 0.0, 0.0, 0.0, -1.0, 1.0, -1.0, 1.0],
        dtype=jnp.float32,
    )

    first = agent.start(raw, jr.key(200))
    second = agent.start(raw, jr.key(201))

    for start in (first, second):
        assert jnp.shape(start.state.interaction.birth_timestamp) == ()
        assert start.state.interaction.birth_timestamp.dtype == jnp.float32
        assert jnp.shape(start.state.interaction.uptime_s) == ()
        assert start.state.interaction.uptime_s.dtype == jnp.float32
        assert jnp.shape(start.state.control.birth_timestamp) == ()
        assert start.state.control.birth_timestamp.dtype == jnp.float32
        assert jnp.shape(start.state.control.uptime_s) == ()
        assert start.state.control.uptime_s.dtype == jnp.float32
        assert agent.resource_budget(start.state).total_state_nbytes == _tree_array_nbytes(
            start.state
        )
    assert float(first.state.interaction.birth_timestamp) == pytest.approx(100.25)
    assert float(second.state.interaction.birth_timestamp) == pytest.approx(200.25)
    assert float(first.state.control.birth_timestamp) == pytest.approx(101.25)
    assert float(second.state.control.birth_timestamp) == pytest.approx(201.25)
    assert (
        agent.resource_budget(first.state).total_state_nbytes
        == agent.resource_budget(second.state).total_state_nbytes
    )


def test_no_planning_masks_only_additive_term_with_compute_and_resource_parity() -> None:
    full = IntegratedHiddenPartnerAgent(IntegratedHiddenPartnerConfig(planning_enabled=True))
    no_planning = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(planning_enabled=False)
    )
    start, _, _ = _start_and_transition(full, seed=3)
    state = start.state
    behavior = state.behavior.replace(
        weights=state.behavior.weights.at[0, 0].set(0.7).at[1, 0].set(-0.4)
    )
    world = state.joint_world.replace(
        reward_predictions=jnp.asarray(
            [[0.0, 1.0], [0.8, 0.2]],
            dtype=jnp.float32,
        ),
        outcome_predictions=jnp.asarray(
            [[[-1.0], [1.0]], [[0.5], [-0.5]]],
            dtype=jnp.float32,
        ),
    )
    control = state.control.replace(
        q_weights=state.control.q_weights.at[0, 0].set(0.2).at[1, 0].set(0.1)
    )

    planned = full.evaluate_models(behavior, world, control, state.chi)
    masked = no_planning.evaluate_models(
        behavior,
        world,
        control,
        state.chi,
    )

    chex.assert_trees_all_equal(
        planned.predicted_partner_probabilities,
        masked.predicted_partner_probabilities,
    )
    chex.assert_trees_all_equal(
        planned.partner_probabilities,
        masked.partner_probabilities,
    )
    chex.assert_trees_all_equal(
        planned.expected_rewards,
        masked.expected_rewards,
    )
    chex.assert_trees_all_equal(planned.expected_outcomes, masked.expected_outcomes)
    chex.assert_trees_all_equal(planned.model_term, masked.model_term)
    chex.assert_trees_all_equal(planned.q_values, masked.q_values)
    chex.assert_trees_all_equal(
        masked.applied_model_term,
        jnp.zeros((2,), dtype=jnp.float32),
    )
    chex.assert_trees_all_equal(masked.planner_scores, masked.q_values)
    assert int(planned.cell_evaluations) == 4
    assert int(masked.cell_evaluations) == 4

    no_planning_start, _, _ = _start_and_transition(no_planning, seed=3)
    assert (
        full.resource_budget(start.state).total_state_nbytes
        == no_planning.resource_budget(no_planning_start.state).total_state_nbytes
    )


def test_uniform_partner_belief_masks_only_applied_planner_distribution() -> None:
    full = IntegratedHiddenPartnerAgent()
    uniform = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(uniform_partner_belief=True)
    )
    start, _, _ = _start_and_transition(full, seed=31)
    state = start.state
    chi = jnp.ones((DEPLOYED_FEATURE_DIM,), dtype=jnp.float32)
    behavior = state.behavior.replace(
        weights=(jnp.zeros_like(state.behavior.weights).at[0, 0].set(1.5).at[1, 0].set(-1.5))
    )
    world = state.joint_world.replace(
        reward_predictions=jnp.asarray(
            [[0.0, 1.0], [1.0, 0.0]],
            dtype=jnp.float32,
        )
    )

    learned = full.evaluate_models(behavior, world, state.control, chi)
    masked = uniform.evaluate_models(behavior, world, state.control, chi)

    chex.assert_trees_all_close(
        learned.predicted_partner_probabilities,
        masked.predicted_partner_probabilities,
    )
    assert not bool(
        jnp.allclose(
            learned.predicted_partner_probabilities,
            jnp.asarray([0.5, 0.5], dtype=jnp.float32),
        )
    )
    chex.assert_trees_all_equal(
        learned.partner_probabilities,
        learned.predicted_partner_probabilities,
    )
    chex.assert_trees_all_equal(
        masked.partner_probabilities,
        jnp.asarray([0.5, 0.5], dtype=jnp.float32),
    )
    assert not bool(jnp.allclose(learned.expected_rewards, masked.expected_rewards))
    assert int(learned.cell_evaluations) == int(masked.cell_evaluations) == 4

    uniform_start, transition, _ = _start_and_transition(uniform, seed=31)
    uniform_result = uniform.update(uniform_start.state, transition)
    assert int(uniform_result.state.behavior.step_count) == 1
    assert bool(uniform_result.diagnostics.behavior_prediction_matches_decision)
    assert (
        full.resource_budget(start.state).total_state_nbytes
        == uniform.resource_budget(uniform_start.state).total_state_nbytes
    )


def test_random_curation_without_cadence_is_an_exact_matched_state_control() -> None:
    full = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(random_feature_curation=False)
    )
    random = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(random_feature_curation=True)
    )
    full_start, transition, _ = _start_and_transition(full, seed=32)

    full_result = full.update(full_start.state, transition)
    random_result = random.update(full_start.state, transition)

    chex.assert_trees_all_equal(full_result.state, random_result.state)
    assert not bool(full_result.diagnostics.random_curation_enabled)
    assert bool(random_result.diagnostics.random_curation_enabled)
    assert not bool(full_result.diagnostics.random_curation_attempted)
    assert not bool(random_result.diagnostics.random_curation_attempted)
    assert not bool(full_result.diagnostics.random_curation_applied)
    assert not bool(random_result.diagnostics.random_curation_applied)
    chex.assert_trees_all_equal(
        full_result.diagnostics.random_active_priorities,
        random_result.diagnostics.random_active_priorities,
    )
    chex.assert_trees_all_equal(
        full_result.diagnostics.random_candidate_priorities,
        random_result.diagnostics.random_candidate_priorities,
    )
    assert (
        jax.tree_util.tree_structure(full_result.state)
        == jax.tree_util.tree_structure(random_result.state)
    )
    assert (
        full.resource_budget(full_result.state).total_state_nbytes
        == random.resource_budget(random_result.state).total_state_nbytes
    )


def test_random_curation_priorities_replace_only_transaction_ranking() -> None:
    agent = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(
            random_feature_curation=True,
            min_feature_age=0,
            candidate_min_age=0,
        )
    )
    start, transition, _ = _start_and_transition(agent, seed=32)
    interaction = start.state.interaction.replace(
        step_count=jnp.asarray(63, dtype=jnp.int32),
        ages=jnp.ones((ACTIVE_PAIR_SLOTS,), dtype=jnp.int32),
        candidate_ages=jnp.ones((CANDIDATE_PAIR_SLOTS,), dtype=jnp.int32),
        utilities=jnp.linspace(1.0, 10.0, ACTIVE_PAIR_SLOTS, dtype=jnp.float32),
        candidate_utilities=jnp.linspace(
            1_000.0,
            2_000.0,
            CANDIDATE_PAIR_SLOTS,
            dtype=jnp.float32,
        ),
    )
    priority_override = agent._interaction_curation_input(
        interaction
    )  # noqa: SLF001
    active_priorities = priority_override.active_ranks
    candidate_priorities = priority_override.candidate_ranks
    adversarial = interaction.replace(
        utilities=jnp.linspace(10.0, 1.0, ACTIVE_PAIR_SLOTS, dtype=jnp.float32),
        candidate_utilities=jnp.linspace(
            2_000.0,
            1_000.0,
            CANDIDATE_PAIR_SLOTS,
            dtype=jnp.float32,
        ),
    )
    repeated_override = agent._interaction_curation_input(
        adversarial
    )  # noqa: SLF001
    full_override = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(random_feature_curation=False)
    )._interaction_curation_input(interaction)  # noqa: SLF001

    assert bool(priority_override.enabled)
    assert not bool(full_override.enabled)
    chex.assert_trees_all_equal(priority_override.active_ranks, repeated_override.active_ranks)
    chex.assert_trees_all_equal(
        priority_override.candidate_ranks,
        repeated_override.candidate_ranks,
    )
    chex.assert_trees_all_equal(priority_override.active_ranks, full_override.active_ranks)
    chex.assert_trees_all_equal(
        priority_override.candidate_ranks,
        full_override.candidate_ranks,
    )
    chex.assert_shape(priority_override.enabled, ())
    chex.assert_shape(active_priorities, (ACTIVE_PAIR_SLOTS,))
    chex.assert_shape(candidate_priorities, (CANDIDATE_PAIR_SLOTS,))
    assert priority_override.enabled.dtype == jnp.bool_
    assert active_priorities.dtype == jnp.float32
    assert candidate_priorities.dtype == jnp.float32
    assert len(np.unique(np.asarray(active_priorities))) == ACTIVE_PAIR_SLOTS
    assert len(np.unique(np.asarray(candidate_priorities))) == CANDIDATE_PAIR_SLOTS

    candidate_matches_active = jnp.any(
        (interaction.candidate_left[:, None] == interaction.feature_left[None, :])
        & (interaction.candidate_right[:, None] == interaction.feature_right[None, :]),
        axis=1,
    )
    expected_active = int(jnp.argmin(active_priorities))
    expected_candidate = int(
        jnp.argmax(jnp.where(~candidate_matches_active, candidate_priorities, -jnp.inf))
    )

    state_a = start.state.replace(interaction=interaction)
    state_b = start.state.replace(interaction=adversarial)
    result_a = agent.update(state_a, transition)
    result_b = agent.update(state_b, transition)
    assert bool(result_a.diagnostics.random_curation_enabled)
    assert bool(result_a.diagnostics.random_curation_attempted)
    assert bool(result_a.diagnostics.random_curation_applied)
    chex.assert_trees_all_equal(
        result_a.diagnostics.random_active_priorities,
        result_b.diagnostics.random_active_priorities,
    )
    chex.assert_trees_all_equal(
        result_a.diagnostics.random_candidate_priorities,
        result_b.diagnostics.random_candidate_priorities,
    )
    assert (
        int(result_a.diagnostics.curation_selected_active_worst_slot)
        == int(result_b.diagnostics.curation_selected_active_worst_slot)
        == expected_active
    )
    assert (
        int(result_a.diagnostics.curation_selected_promotion_candidate)
        == int(result_b.diagnostics.curation_selected_promotion_candidate)
        == expected_candidate
    )
    assert int(result_a.diagnostics.interaction_replaced_slot) == expected_active
    assert int(result_b.diagnostics.interaction_replaced_slot) == expected_active
    assert int(result_a.diagnostics.interaction_promoted_candidate) == expected_candidate
    assert int(result_b.diagnostics.interaction_promoted_candidate) == expected_candidate
    assert not bool(
        jnp.array_equal(result_a.state.interaction.utilities, active_priorities)
    )
    assert not bool(
        jnp.array_equal(
            result_a.state.interaction.candidate_utilities,
            candidate_priorities,
        )
    )


def test_one_update_is_prequential_and_advances_every_online_counter_once() -> None:
    agent = IntegratedHiddenPartnerAgent()
    start, transition, _ = _start_and_transition(agent, seed=4)
    before = start.state
    result = agent.update(before, transition)
    diagnostics = result.diagnostics

    chex.assert_trees_all_equal(
        diagnostics.behavior_probabilities_preupdate,
        before.current_evaluation.predicted_partner_probabilities,
    )
    chex.assert_trees_all_close(
        diagnostics.behavior_probabilities_preupdate,
        jnp.asarray([0.5, 0.5], dtype=jnp.float32),
    )
    assert bool(diagnostics.behavior_prediction_matches_decision)
    assert float(diagnostics.world_reward_prediction_preupdate) == 0.0
    chex.assert_trees_all_equal(
        diagnostics.world_outcome_prediction_preupdate,
        jnp.zeros((1,), dtype=jnp.float32),
    )
    expected_partner_sign = 2.0 * float(transition.partner_action) - 1.0
    assert float(diagnostics.interaction_error_preupdate[0]) == pytest.approx(expected_partner_sign)

    assert int(diagnostics.current_evaluation.cell_evaluations) == 4
    assert int(diagnostics.next_evaluation.cell_evaluations) == 4
    assert int(diagnostics.state_builder_step_delta) == 1
    assert int(diagnostics.state_builder_learning_delta) == 1
    assert int(diagnostics.behavior_step_delta) == 1
    assert int(diagnostics.interaction_step_delta) == 1
    assert int(diagnostics.world_step_delta) == 1
    assert int(diagnostics.control_step_delta) == 1
    assert int(diagnostics.router_route_delta) == 1
    assert int(diagnostics.router_generation_delta) == 0
    assert int(diagnostics.integrated_step_delta) == 1
    assert int(result.state.state_builder.step_count) == 2
    assert int(result.state.state_builder.update_count) == 1
    assert int(result.state.behavior.step_count) == 1
    assert int(result.state.interaction.step_count) == 1
    assert int(result.state.joint_world.step_count) == 1
    assert int(result.state.control.step_count) == 1
    assert int(result.state.router.route_count) == 1
    assert int(result.state.step_count) == 1
    assert int(result.action) == int(diagnostics.next_selection.action)
    assert int(result.state.control.last_action) == int(result.action)
    chex.assert_trees_all_equal(
        result.state.current_evaluation,
        diagnostics.next_evaluation,
    )
    chex.assert_trees_all_equal(
        result.state.current_selection,
        diagnostics.next_selection,
    )
    _assert_current_q_value_delta(agent, result.state)
    assert not bool(result.state.current_selection.externally_forced)
    chex.assert_trees_all_equal(
        result.state.control.rng_key,
        diagnostics.next_selection.rng_key_after,
    )
    chex.assert_trees_all_equal(
        result.state.raw_observation,
        transition.next_observation,
    )
    assert bool(diagnostics.transition_observation_matches)
    assert bool(diagnostics.transition_action_matches)
    assert bool(diagnostics.transition_semantics_valid)
    assert bool(diagnostics.model_valid)
    assert bool(diagnostics.route.valid)
    assert bool(diagnostics.world_target_valid)
    assert bool(diagnostics.all_finite)


def test_state_learning_ablation_computes_gradient_but_discards_parameter_update() -> None:
    enabled = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(state_learning_enabled=True)
    )
    disabled = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(state_learning_enabled=False)
    )
    enabled_start, transition, _ = _start_and_transition(enabled, seed=5)
    disabled_start, disabled_transition, _ = _start_and_transition(
        disabled,
        seed=5,
    )
    seeded_weights = (
        jnp.zeros_like(enabled_start.state.behavior.weights)
        .at[0, RAW_OBSERVATION_DIM]
        .set(1.0)
        .at[1, RAW_OBSERVATION_DIM]
        .set(-1.0)
    )

    def seed_behavior(agent, state):
        behavior = state.behavior.replace(weights=seeded_weights)
        evaluation = agent.evaluate_models(
            behavior,
            state.joint_world,
            state.control,
            state.chi,
        )
        return state.replace(
            behavior=behavior,
            current_evaluation=evaluation,
        )

    enabled_state = seed_behavior(enabled, enabled_start.state)
    disabled_state = seed_behavior(disabled, disabled_start.state)
    enabled_result = enabled.update(enabled_state, transition)
    disabled_result = disabled.update(
        disabled_state,
        disabled_transition,
    )

    assert int(enabled_result.diagnostics.state_builder_step_delta) == 1
    assert int(disabled_result.diagnostics.state_builder_step_delta) == 1
    assert int(enabled_result.diagnostics.state_builder_learning_delta) == 1
    assert int(disabled_result.diagnostics.state_builder_learning_delta) == 0
    chex.assert_trees_all_equal(
        enabled_result.diagnostics.behavior_gradient_phi,
        disabled_result.diagnostics.behavior_gradient_phi,
    )
    assert float(enabled_result.diagnostics.state_learning.parameter_update_norm) > 0.0
    assert float(disabled_result.diagnostics.state_learning.parameter_update_norm) > 0.0
    assert not bool(
        jnp.all(
            enabled_result.state.state_builder.parameters == enabled_state.state_builder.parameters
        )
    )
    chex.assert_trees_all_equal(
        disabled_result.state.state_builder.parameters,
        disabled_state.state_builder.parameters,
    )
    assert (
        enabled.resource_budget(enabled_start.state).total_state_nbytes
        == disabled.resource_budget(disabled_start.state).total_state_nbytes
    )


@pytest.mark.parametrize("pair", [(0, 2), (4, 5)], ids=["C", "D"])
def test_feature_lifecycle_freeze_commits_exact_pre_curation_learning(
    pair: tuple[int, int],
) -> None:
    enabled = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(feature_lifecycle_enabled=True)
    )
    frozen = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(feature_lifecycle_enabled=False)
    )
    enabled_start, transition, _ = _start_and_transition(enabled, seed=6)
    frozen_start, frozen_transition, _ = _start_and_transition(
        frozen,
        seed=6,
    )
    enabled_state = _force_next_interaction_promotion(enabled_start.state, pair)
    frozen_state = _force_next_interaction_promotion(frozen_start.state, pair)
    chex.assert_trees_all_equal(enabled_state, frozen_state)
    target = jnp.reshape(
        2.0 * transition.partner_action.astype(jnp.float32) - 1.0,
        (1,),
    )
    expected_proposal = frozen.interaction_learner.update(
        frozen_state.interaction,
        frozen_state.phi,
        target,
        external_read_mask=frozen_state.consumer_active_mask,
    )

    enabled_result = enabled.update(enabled_state, transition)
    frozen_result = frozen.update(frozen_state, frozen_transition)
    _assert_current_q_value_delta(enabled, enabled_result.state)
    _assert_current_q_value_delta(frozen, frozen_result.state)

    assert int(enabled_result.diagnostics.interaction_replaced_slot) == 0
    assert int(frozen_result.diagnostics.interaction_replaced_slot) == 0
    assert int(enabled_result.diagnostics.interaction_proposal_replaced_slot) == 0
    assert int(frozen_result.diagnostics.interaction_proposal_replaced_slot) == 0
    assert bool(enabled_result.diagnostics.interaction_lifecycle_proposed)
    assert bool(frozen_result.diagnostics.interaction_lifecycle_proposed)
    assert bool(enabled_result.diagnostics.interaction_lifecycle_applied)
    assert not bool(frozen_result.diagnostics.interaction_lifecycle_applied)
    assert int(enabled_result.diagnostics.interaction_applied_replaced_slot) == 0
    assert int(frozen_result.diagnostics.interaction_applied_replaced_slot) == -1
    assert bool(enabled_result.diagnostics.shadow_descriptors_changed)
    assert bool(frozen_result.diagnostics.shadow_descriptors_changed)
    assert bool(enabled_result.diagnostics.route.descriptors_changed)
    assert not bool(frozen_result.diagnostics.route.descriptors_changed)
    assert int(enabled_result.diagnostics.router_generation_delta) == 1
    assert int(frozen_result.diagnostics.router_generation_delta) == 0
    np.testing.assert_array_equal(
        enabled_result.state.router.descriptors[0],
        pair,
    )
    np.testing.assert_array_equal(
        frozen_result.state.router.descriptors,
        frozen_start.state.router.descriptors,
    )
    chex.assert_trees_all_equal(
        frozen_result.state.interaction,
        expected_proposal.pre_curation_state,
    )
    chex.assert_trees_all_equal(
        frozen_result.state.interaction.feature_left,
        frozen_state.interaction.feature_left,
    )
    chex.assert_trees_all_equal(
        frozen_result.state.interaction.feature_right,
        frozen_state.interaction.feature_right,
    )
    chex.assert_trees_all_equal(
        frozen_result.state.interaction.feature_parent_a,
        frozen_state.interaction.feature_parent_a,
    )
    chex.assert_trees_all_equal(
        frozen_result.state.interaction.feature_parent_b,
        frozen_state.interaction.feature_parent_b,
    )
    chex.assert_trees_all_equal(
        frozen_result.state.interaction.feature_generator,
        frozen_state.interaction.feature_generator,
    )
    chex.assert_trees_all_equal(
        frozen_result.state.interaction.candidate_left,
        frozen_state.interaction.candidate_left,
    )
    chex.assert_trees_all_equal(
        frozen_result.state.interaction.candidate_right,
        frozen_state.interaction.candidate_right,
    )
    assert int(frozen_result.state.interaction.step_count) == 64
    assert not bool(
        jnp.array_equal(
            frozen_result.state.interaction.key,
            frozen_state.interaction.key,
        )
    )
    assert (
        jax.tree_util.tree_structure(enabled_result.state)
        == jax.tree_util.tree_structure(frozen_result.state)
    )
    assert (
        enabled.resource_budget(enabled_result.state).total_state_nbytes
        == frozen.resource_budget(frozen_result.state).total_state_nbytes
    )


def test_evidence_lease_retires_then_fills_one_vacancy_atomically() -> None:
    config = IntegratedHiddenPartnerConfig(
        active_utility_retention_grace_steps=0,
        active_utility_evidence_threshold=0.99,
        retire_stale_features=True,
        candidate_promotion_floor=0.001,
        replacement_interval=1,
        min_feature_age=0,
        candidate_min_age=0,
    )
    agent = IntegratedHiddenPartnerAgent(config)
    environment = _environment()
    environment_state = environment.init(jr.key(42))
    start = agent.start(
        environment.observe(environment_state),
        jr.key(10_042),
    )
    transition, environment_state = environment.step(
        environment_state,
        start.action,
    )

    retired = agent.update(start.state, transition)
    retired_diagnostics = retired.diagnostics
    dynamic_slot = BASE_FEATURE_DIM

    assert int(retired_diagnostics.interaction_retired_slot) == 0
    np.testing.assert_array_equal(
        [
            retired_diagnostics.interaction_retired_left,
            retired_diagnostics.interaction_retired_right,
        ],
        INITIAL_ACTIVE_DESCRIPTORS[0],
    )
    assert int(retired_diagnostics.interaction_replaced_slot) == -1
    assert int(retired_diagnostics.interaction_promoted_candidate) == -1
    assert int(retired_diagnostics.interaction_live_feature_count) == 11
    assert int(retired_diagnostics.interaction_vacancy_count) == 1
    assert int(retired_diagnostics.interaction_matching_candidate_reset_count) == 1
    assert int(retired_diagnostics.route.evicted_count) == 1
    assert int(retired_diagnostics.route.new_count) == 0
    assert int(retired_diagnostics.route.old_live_count) == 12
    assert int(retired_diagnostics.route.new_live_count) == 11
    assert bool(retired_diagnostics.consumer_route_values_exact)
    assert bool(retired_diagnostics.consumer_lifecycle_destination_reset_exact)
    np.testing.assert_array_equal(
        retired.state.router.descriptors[0],
        [-1, -1],
    )
    np.testing.assert_array_equal(
        retired.state.interaction.feature_left[0],
        -1,
    )
    np.testing.assert_array_equal(
        retired.state.interaction.feature_right[0],
        -1,
    )
    for consumer in (
        retired.state.behavior.weights,
        retired.state.control.q_weights,
        retired.state.control.q_trace_weights,
        retired.state.control.last_observation,
    ):
        np.testing.assert_array_equal(
            np.asarray(consumer)[..., dynamic_slot],
            0.0,
        )

    interaction = retired.state.interaction
    candidate_matches_c = (interaction.candidate_left == 0) & (interaction.candidate_right == 2)
    candidate_index = int(jnp.argmax(candidate_matches_c))
    forced_interaction = interaction.replace(
        candidate_output_weights=interaction.candidate_output_weights.at[:, candidate_index].set(
            1.0
        ),
        candidate_utilities=interaction.candidate_utilities.at[candidate_index].set(10.0),
        candidate_ages=interaction.candidate_ages.at[candidate_index].set(10),
    )
    forced_state = retired.state.replace(interaction=forced_interaction)
    transition, _ = environment.step(environment_state, retired.action)
    filled = agent.update(forced_state, transition)
    filled_diagnostics = filled.diagnostics

    assert int(filled_diagnostics.interaction_retired_slot) == -1
    assert int(filled_diagnostics.interaction_replaced_slot) == 0
    assert int(filled_diagnostics.interaction_promoted_candidate) == candidate_index
    assert bool(filled_diagnostics.interaction_promoted_into_vacancy)
    assert int(filled_diagnostics.interaction_live_feature_count) == 12
    assert int(filled_diagnostics.interaction_vacancy_count) == 0
    assert int(filled_diagnostics.route.evicted_count) == 0
    assert int(filled_diagnostics.route.new_count) == 1
    np.testing.assert_array_equal(
        filled.state.router.descriptors[0],
        [0, 2],
    )
    for consumer in (
        filled.state.behavior.weights,
        filled.state.control.q_weights,
        filled.state.control.q_trace_weights,
    ):
        np.testing.assert_array_equal(
            np.asarray(consumer)[..., dynamic_slot],
            0.0,
        )
    assert float(filled.state.control.last_observation[dynamic_slot]) == pytest.approx(
        float(filled.state.chi[dynamic_slot])
    )


def test_lifecycle_freeze_diagnoses_retirement_without_committing_resets() -> None:
    config = IntegratedHiddenPartnerConfig(
        feature_lifecycle_enabled=False,
        active_utility_retention_grace_steps=0,
        active_utility_evidence_threshold=0.99,
        retire_stale_features=True,
        candidate_promotion_floor=0.001,
        replacement_interval=1,
        min_feature_age=0,
        candidate_min_age=0,
    )
    agent = IntegratedHiddenPartnerAgent(config)
    start, transition, _ = _start_and_transition(agent, seed=42)
    target = jnp.reshape(
        2.0 * transition.partner_action.astype(jnp.float32) - 1.0,
        (1,),
    )
    proposal = agent.interaction_learner.update(
        start.state.interaction,
        start.state.phi,
        target,
        external_read_mask=start.state.consumer_active_mask,
    )

    assert int(proposal.retired_slot) == 0
    result = agent.update(start.state, transition)
    diagnostics = result.diagnostics

    assert int(diagnostics.interaction_proposal_retired_slot) == 0
    assert int(diagnostics.interaction_applied_retired_slot) == -1
    assert bool(diagnostics.interaction_lifecycle_proposed)
    assert not bool(diagnostics.interaction_lifecycle_applied)
    assert int(diagnostics.interaction_proposal_live_feature_count) == 11
    assert int(diagnostics.interaction_applied_live_feature_count) == 12
    assert int(diagnostics.interaction_matching_candidate_reset_count) == 1
    assert int(diagnostics.interaction_applied_matching_candidate_reset_count) == 0
    assert not bool(diagnostics.route.descriptors_changed)
    assert int(diagnostics.router_generation_delta) == 0
    chex.assert_trees_all_equal(
        result.state.interaction,
        proposal.pre_curation_state,
    )
    chex.assert_trees_all_equal(
        result.state.interaction.feature_left,
        start.state.interaction.feature_left,
    )
    chex.assert_trees_all_equal(
        result.state.interaction.feature_right,
        start.state.interaction.feature_right,
    )
    chex.assert_trees_all_equal(
        result.state.interaction.feature_parent_a,
        start.state.interaction.feature_parent_a,
    )
    chex.assert_trees_all_equal(
        result.state.interaction.feature_parent_b,
        start.state.interaction.feature_parent_b,
    )
    chex.assert_trees_all_equal(
        result.state.interaction.feature_generator,
        start.state.interaction.feature_generator,
    )
    chex.assert_trees_all_equal(
        result.state.interaction.candidate_reacquisition_required,
        proposal.pre_curation_state.candidate_reacquisition_required,
    )
    assert int(result.state.interaction.step_count) == 1
    assert not bool(
        jnp.array_equal(
            result.state.interaction.key,
            start.state.interaction.key,
        )
    )
def test_atomic_route_moves_all_four_downstream_feature_consumers() -> None:
    agent = IntegratedHiddenPartnerAgent()
    start, _, _ = _start_and_transition(agent, seed=7)
    state = start.state
    old = state.router.descriptors
    proposed = (
        old.at[0].set(old[2]).at[1].set(old[0]).at[2].set(jnp.asarray([4, 5], dtype=jnp.int32))
    )
    behavior_weights = jnp.arange(
        2 * DEPLOYED_FEATURE_DIM,
        dtype=jnp.float32,
    ).reshape((2, DEPLOYED_FEATURE_DIM))
    q_weights = behavior_weights + 100.0
    q_traces = behavior_weights + 200.0
    last_observation = (
        jnp.arange(
            DEPLOYED_FEATURE_DIM,
            dtype=jnp.float32,
        )
        + 300.0
    )
    behavior = state.behavior.replace(weights=behavior_weights)
    control = state.control.replace(
        q_weights=q_weights,
        q_trace_weights=q_traces,
        last_observation=last_observation,
    )

    routed_behavior, routed_control, diagnostics, routed_router = agent._route_feature_consumers(  # noqa: SLF001
        state.router,
        behavior,
        control,
        proposed,
    )
    source = np.asarray([2, 0, -1] + list(range(3, ACTIVE_PAIR_SLOTS)))

    def expected(array: np.ndarray) -> np.ndarray:
        prefix = array[..., :BASE_FEATURE_DIM]
        old_tail = array[..., BASE_FEATURE_DIM:]
        tail = np.zeros_like(old_tail)
        live = source >= 0
        tail[..., live] = old_tail[..., source[live]]
        return np.concatenate((prefix, tail), axis=-1)

    np.testing.assert_array_equal(
        routed_behavior.weights,
        expected(np.asarray(behavior_weights)),
    )
    np.testing.assert_array_equal(
        routed_control.q_weights,
        expected(np.asarray(q_weights)),
    )
    np.testing.assert_array_equal(
        routed_control.q_trace_weights,
        expected(np.asarray(q_traces)),
    )
    np.testing.assert_array_equal(
        routed_control.last_observation,
        expected(np.asarray(last_observation)),
    )
    chex.assert_trees_all_equal(routed_behavior.bias, behavior.bias)
    chex.assert_trees_all_equal(routed_behavior.rng_key, behavior.rng_key)
    chex.assert_trees_all_equal(routed_control.q_bias, control.q_bias)
    chex.assert_trees_all_equal(routed_control.rng_key, control.rng_key)
    np.testing.assert_array_equal(diagnostics.source_slots, source)
    assert bool(diagnostics.valid)
    assert int(routed_router.generation_count) == 1


@pytest.mark.parametrize("compiled", [False, True])
def test_update_reports_independent_consumer_identity_value_audit(compiled: bool) -> None:
    agent = IntegratedHiddenPartnerAgent()
    start, transition, _ = _start_and_transition(agent, seed=7_701)
    update = jax.jit(agent.update) if compiled else agent.update

    diagnostics = update(start.state, transition).diagnostics

    for verdict in (
        diagnostics.consumer_route_source_slots_exact,
        diagnostics.consumer_route_identity_masks_exact,
        diagnostics.consumer_route_stable_prefix_exact,
        diagnostics.consumer_route_survivor_values_exact,
        diagnostics.consumer_route_reset_values_exact,
        diagnostics.consumer_route_no_carry_reset_exact,
        diagnostics.consumer_route_behavior_values_exact,
        diagnostics.consumer_route_q_values_exact,
        diagnostics.consumer_route_trace_values_exact,
        diagnostics.consumer_route_last_observation_exact,
        diagnostics.consumer_route_grounded_values_exact,
        diagnostics.consumer_route_values_exact,
        diagnostics.consumer_lifecycle_destination_reset_exact,
    ):
        assert bool(verdict)


def test_consumer_identity_audit_rejects_each_corrupt_routed_column_eager_and_jit() -> None:
    agent = IntegratedHiddenPartnerAgent(
        _grounded_integrated_config(
            feature_lifecycle_enabled=True,
            replacement_interval=64,
        )
    )
    start, _, _ = _start_and_transition(agent, seed=7_702)
    state = start.state
    assert state.grounded_world is not None
    old = state.router.descriptors
    proposed = (
        old.at[0]
        .set(old[2])
        .at[1]
        .set(old[0])
        .at[2]
        .set(jnp.asarray([4, 5], dtype=jnp.int32))
    )
    base = jnp.arange(2 * DEPLOYED_FEATURE_DIM, dtype=jnp.float32).reshape(
        (2, DEPLOYED_FEATURE_DIM)
    )
    behavior = state.behavior.replace(weights=base)
    control = state.control.replace(
        q_weights=base + 100.0,
        q_trace_weights=base + 200.0,
        last_observation=jnp.arange(DEPLOYED_FEATURE_DIM, dtype=jnp.float32) + 300.0,
    )
    grounded = state.grounded_world.replace(
        weights=jnp.arange(
            state.grounded_world.weights.size,
            dtype=jnp.float32,
        ).reshape(state.grounded_world.weights.shape)
    )
    routed_behavior, routed_control, routed_grounded, route, _ = (
        agent._route_feature_consumers_with_grounded(  # noqa: SLF001
            state.router,
            behavior,
            control,
            grounded,
            proposed,
        )
    )

    def audit(
        route_diagnostics: Any,
        behavior_after: jax.Array,
        q_after: jax.Array,
        trace_after: jax.Array,
        observation_after: jax.Array,
        grounded_after: jax.Array,
    ) -> Any:
        return agent._audit_consumer_identity_route(  # noqa: SLF001
            old_descriptors=old,
            new_descriptors=proposed,
            route=route_diagnostics,
            behavior_before=behavior.weights,
            behavior_after=behavior_after,
            q_before=control.q_weights,
            q_after=q_after,
            trace_before=control.q_trace_weights,
            trace_after=trace_after,
            last_observation_before=control.last_observation,
            last_observation_after=observation_after,
            grounded_before=grounded.weights,
            grounded_after=grounded_after,
            retired_slot=jnp.asarray(-1, dtype=jnp.int32),
            replaced_slot=jnp.asarray(2, dtype=jnp.int32),
        )

    compiled_audit = jax.jit(audit)
    valid_arguments = (
        route,
        routed_behavior.weights,
        routed_control.q_weights,
        routed_control.q_trace_weights,
        routed_control.last_observation,
        routed_grounded.weights,
    )
    for evaluate in (audit, compiled_audit):
        assert bool(evaluate(*valid_arguments).values_exact)

    destination = BASE_FEATURE_DIM
    corruptions = (
        (
            dataclasses.replace(route, source_slots=route.source_slots.at[0].set(1)),
            *valid_arguments[1:],
            "source_slots_exact",
        ),
        (
            route,
            routed_behavior.weights.at[..., destination].add(0.25),
            *valid_arguments[2:],
            "behavior_values_exact",
        ),
        (
            route,
            valid_arguments[1],
            routed_control.q_weights.at[..., destination].add(0.25),
            *valid_arguments[3:],
            "q_values_exact",
        ),
        (
            route,
            *valid_arguments[1:3],
            routed_control.q_trace_weights.at[..., destination].add(0.25),
            *valid_arguments[4:],
            "trace_values_exact",
        ),
        (
            route,
            *valid_arguments[1:4],
            routed_control.last_observation.at[destination].add(0.25),
            valid_arguments[5],
            "last_observation_exact",
        ),
        (
            route,
            *valid_arguments[1:5],
            routed_grounded.weights.at[..., destination].add(0.25),
            "grounded_values_exact",
        ),
    )
    for corruption in corruptions:
        *arguments, failed_field = corruption
        for evaluate in (audit, compiled_audit):
            verdict = evaluate(*arguments)
            assert not bool(getattr(verdict, failed_field))
            assert not bool(verdict.values_exact)

    reset_destination = BASE_FEATURE_DIM + 2
    replacement_corruption = (
        route,
        routed_behavior.weights.at[..., reset_destination].set(1.0),
        *valid_arguments[2:],
    )
    for evaluate in (audit, compiled_audit):
        verdict = evaluate(*replacement_corruption)
        assert not bool(verdict.reset_values_exact)
        assert not bool(verdict.behavior_values_exact)
        assert not bool(verdict.values_exact)
        assert not bool(verdict.lifecycle_destination_reset_exact)


def test_consumer_identity_audit_proves_retired_destination_reset_eager_and_jit() -> None:
    agent = IntegratedHiddenPartnerAgent(
        _grounded_integrated_config(
            feature_lifecycle_enabled=True,
            replacement_interval=64,
        )
    )
    start, _, _ = _start_and_transition(agent, seed=7_703)
    state = start.state
    assert state.grounded_world is not None
    retired_descriptors = state.router.descriptors.at[0].set(
        jnp.asarray((-1, -1), dtype=jnp.int32)
    )
    routed_behavior, routed_control, routed_grounded, route, _ = (
        agent._route_feature_consumers_with_grounded(  # noqa: SLF001
            state.router,
            state.behavior,
            state.control,
            state.grounded_world,
            retired_descriptors,
        )
    )

    def audit(q_after: jax.Array) -> Any:
        return agent._audit_consumer_identity_route(  # noqa: SLF001
            old_descriptors=state.router.descriptors,
            new_descriptors=retired_descriptors,
            route=route,
            behavior_before=state.behavior.weights,
            behavior_after=routed_behavior.weights,
            q_before=state.control.q_weights,
            q_after=q_after,
            trace_before=state.control.q_trace_weights,
            trace_after=routed_control.q_trace_weights,
            last_observation_before=state.control.last_observation,
            last_observation_after=routed_control.last_observation,
            grounded_before=state.grounded_world.weights,
            grounded_after=routed_grounded.weights,
            retired_slot=jnp.asarray(0, dtype=jnp.int32),
            replaced_slot=jnp.asarray(-1, dtype=jnp.int32),
        )

    corrupted_q = routed_control.q_weights.at[..., BASE_FEATURE_DIM].set(1.0)
    for evaluate in (audit, jax.jit(audit)):
        assert bool(evaluate(routed_control.q_weights).lifecycle_destination_reset_exact)
        verdict = evaluate(corrupted_q)
        assert not bool(verdict.reset_values_exact)
        assert not bool(verdict.q_values_exact)
        assert not bool(verdict.lifecycle_destination_reset_exact)


def test_no_carry_preserves_between_transactions_and_zeros_on_change() -> None:
    agent = IntegratedHiddenPartnerAgent(IntegratedHiddenPartnerConfig(carry_survivors=False))
    start, _, _ = _start_and_transition(agent, seed=8)
    state = start.state
    behavior = state.behavior.replace(weights=jnp.ones_like(state.behavior.weights))
    control = state.control.replace(
        q_weights=jnp.full_like(state.control.q_weights, 2.0),
        q_trace_weights=jnp.full_like(state.control.q_trace_weights, 3.0),
        last_observation=jnp.full_like(state.control.last_observation, 4.0),
    )
    stable_behavior, stable_control, stable_diagnostics, stable_router = (
        agent._route_feature_consumers(  # noqa: SLF001
            state.router,
            behavior,
            control,
            state.router.descriptors,
        )
    )

    chex.assert_trees_all_equal(stable_behavior.weights, behavior.weights)
    chex.assert_trees_all_equal(stable_control.q_weights, control.q_weights)
    chex.assert_trees_all_equal(
        stable_control.q_trace_weights,
        control.q_trace_weights,
    )
    chex.assert_trees_all_equal(
        stable_control.last_observation,
        control.last_observation,
    )
    assert bool(stable_diagnostics.carry_survivors)
    assert not bool(stable_diagnostics.descriptors_changed)
    assert int(stable_router.route_count) == int(state.router.route_count) + 1

    old = stable_router.descriptors
    proposed = (
        old.at[0].set(old[2]).at[1].set(old[0]).at[2].set(jnp.asarray([4, 5], dtype=jnp.int32))
    )
    routed_behavior, routed_control, diagnostics, routed_router = agent._route_feature_consumers(  # noqa: SLF001
        stable_router,
        stable_behavior,
        stable_control,
        proposed,
    )

    for before, after in (
        (stable_behavior.weights, routed_behavior.weights),
        (stable_control.q_weights, routed_control.q_weights),
        (stable_control.q_trace_weights, routed_control.q_trace_weights),
        (
            stable_control.last_observation,
            routed_control.last_observation,
        ),
    ):
        chex.assert_trees_all_equal(
            after[..., :BASE_FEATURE_DIM],
            before[..., :BASE_FEATURE_DIM],
        )
        chex.assert_trees_all_equal(
            after[..., BASE_FEATURE_DIM:],
            jnp.zeros_like(after[..., BASE_FEATURE_DIM:]),
        )
    assert not bool(diagnostics.carry_survivors)
    assert bool(diagnostics.descriptors_changed)
    assert bool(diagnostics.valid)
    assert int(routed_router.route_count) == int(stable_router.route_count) + 1
    assert int(routed_router.generation_count) == int(stable_router.generation_count) + 1


def test_external_next_action_advances_rng_before_explicit_sarsa_update() -> None:
    agent = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(
            q_step_size=0.0,
            average_reward_step_size=0.0,
            epsilon=0.0,
        )
    )
    start, _, _ = _start_and_transition(agent, seed=9)
    control = start.state.control
    scores = jnp.asarray([-1.0, 2.0], dtype=jnp.float32)
    selection = agent.select_planner_action(control, scores)
    expected_key = jr.split(control.rng_key, 4)[0]

    assert int(selection.action) == 1
    chex.assert_trees_all_equal(selection.rng_key_after, expected_key)
    chex.assert_trees_all_equal(control.rng_key, selection.rng_key_before)
    advanced = control.replace(rng_key=selection.rng_key_after)
    update = agent.control_agent.update(
        advanced,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.ones((DEPLOYED_FEATURE_DIM,), dtype=jnp.float32),
        next_action=selection.action,
        discount=1.0,
    )
    chex.assert_trees_all_equal(update.state.rng_key, expected_key)
    assert int(update.state.last_action) == 1
    assert int(update.action) == 1
    assert int(update.state.step_count) == int(control.step_count) + 1


def test_zero_control_learning_rates_keep_exact_zero_q_provenance_delta() -> None:
    agent = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(
            q_step_size=0.0,
            average_reward_step_size=0.0,
            trace_decay=0.9,
        )
    )
    start, transition, _ = _start_and_transition(agent, seed=9_001)
    result = agent.update(start.state, transition)

    chex.assert_trees_all_equal(
        result.state.current_q_value_delta,
        jnp.zeros((2,), dtype=jnp.float32),
    )
    _assert_current_q_value_delta(agent, result.state)
    assert bool(result.diagnostics.all_finite)


def test_resource_accounting_is_exact_and_ablation_shape_matched() -> None:
    configs = [
        IntegratedHiddenPartnerConfig(),
        IntegratedHiddenPartnerConfig(planning_enabled=False),
        IntegratedHiddenPartnerConfig(state_learning_enabled=False),
        IntegratedHiddenPartnerConfig(feature_lifecycle_enabled=False),
        IntegratedHiddenPartnerConfig(carry_survivors=False),
        IntegratedHiddenPartnerConfig(memory_masked=True),
        IntegratedHiddenPartnerConfig(active_utility_retention_decay=None),
        IntegratedHiddenPartnerConfig(uniform_partner_belief=True),
        IntegratedHiddenPartnerConfig(random_feature_curation=True),
    ]
    totals = []
    for index, config in enumerate(configs):
        agent = IntegratedHiddenPartnerAgent(config)
        start, _, _ = _start_and_transition(agent, seed=30 + index)
        budget = agent.resource_budget(start.state)
        assert budget.raw_observation_dim == 8
        assert budget.base_feature_dim == 12
        assert budget.active_pair_slots == 12
        assert budget.candidate_pair_slots == CANDIDATE_PAIR_SLOTS == 66
        assert budget.deployed_feature_dim == 24
        assert budget.planner_cell_evaluations_per_decision == 4
        assert budget.replay_capacity == 0
        assert budget.total_state_nbytes == _tree_array_nbytes(start.state)
        totals.append(budget.total_state_nbytes)
    assert len(set(totals)) == 1


@pytest.mark.parametrize("random_feature_curation", [False, True])
def test_jitted_two_step_scan_preserves_causal_counters_and_finiteness(
    random_feature_curation: bool,
) -> None:
    agent = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(
            random_feature_curation=random_feature_curation,
        )
    )
    environment = _environment()
    environment_state = environment.init(jr.key(40))
    start = agent.start(
        environment.observe(environment_state),
        jr.key(41),
    )

    def scan_step(carry, _):
        agent_state, world_state = carry
        transition, next_world_state = environment.step(
            world_state,
            agent_state.control.last_action,
        )
        update = agent.update(agent_state, transition)
        outputs = (
            transition.reward,
            update.action,
            update.diagnostics.state_builder_step_delta,
            update.diagnostics.next_evaluation.cell_evaluations,
            update.diagnostics.all_finite,
        )
        return (update.state, next_world_state), outputs

    (final_agent, final_environment), outputs = jax.jit(
        lambda agent_state, world_state: jax.lax.scan(
            scan_step,
            (agent_state, world_state),
            xs=None,
            length=2,
        )
    )(start.state, environment_state)
    rewards, actions, builder_deltas, cell_evaluations, finite = outputs

    assert int(final_agent.step_count) == 2
    assert int(final_agent.state_builder.step_count) == 3
    assert int(final_agent.behavior.step_count) == 2
    assert int(final_agent.interaction.step_count) == 2
    assert int(final_agent.joint_world.step_count) == 2
    assert int(final_agent.control.step_count) == 2
    assert int(final_environment.step_count) == 2
    chex.assert_shape(rewards, (2,))
    chex.assert_shape(actions, (2,))
    chex.assert_trees_all_equal(
        builder_deltas,
        jnp.ones((2,), dtype=jnp.int32),
    )
    chex.assert_trees_all_equal(
        cell_evaluations,
        jnp.full((2,), 4, dtype=jnp.int32),
    )
    assert bool(jnp.all(finite))


def test_natural_candidate_lifecycle_promotes_routes_and_reaches_consumers() -> None:
    """Exercise discovery from ordinary initialization without state injection."""
    agent = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(
            evidence_gated_feature_memory=True,
            feature_evidence_confirmation_steps=1,
            independent_relevance_probe=True,
            evidence_gated_consumer_memory=True,
            consumer_evidence_confirmation_steps=1,
            consumer_read_confirmation_steps=1,
            consumer_read_lease_steps=32,
            active_utility_retention_grace_steps=32,
            active_utility_evidence_threshold=1e-5,
            candidate_promotion_confirmation_steps=2,
            replacement_interval=4,
            min_feature_age=0,
            candidate_min_age=0,
        )
    )
    environment = _environment()
    environment_state = environment.init(jr.key(900))
    start = agent.start(environment.observe(environment_state), jr.key(901))
    state = start.state
    chex.assert_trees_all_equal(
        state.interaction.candidate_output_weights,
        jnp.zeros_like(state.interaction.candidate_output_weights),
    )

    promoted_slot = -1
    promoted_descriptor: tuple[int, int] | None = None
    downstream_use_observed = False
    for _ in range(8):
        transition, environment_state = environment.step(
            environment_state,
            state.control.last_action,
        )
        result = agent.update(state, transition)
        assert not bool(result.diagnostics.transition_rejected)
        promoted_candidate = int(result.diagnostics.interaction_promoted_candidate)
        if promoted_candidate >= 0 and promoted_slot < 0:
            promoted_slot = int(result.diagnostics.interaction_replaced_slot)
            promoted_descriptor = (
                int(state.interaction.candidate_left[promoted_candidate]),
                int(state.interaction.candidate_right[promoted_candidate]),
            )
            assert (
                abs(
                    float(
                        state.interaction.candidate_output_weights[
                            0,
                            promoted_candidate,
                        ]
                    )
                )
                > 0.0
            )
            assert bool(
                result.diagnostics.interaction_candidate_promotion_raw_evidence[promoted_candidate]
            )
            assert bool(
                result.diagnostics.interaction_candidate_promotion_confirmed[promoted_candidate]
            )
            assert (
                int(
                    result.diagnostics.interaction_candidate_promotion_evidence_streak_updated[
                        promoted_candidate
                    ]
                )
                >= 2
            )
            assert bool(result.diagnostics.route.descriptors_changed)
            assert int(result.diagnostics.route.new_count) == 1
            assert tuple(map(int, result.state.router.descriptors[promoted_slot])) == (
                promoted_descriptor
            )

        state = result.state
        if promoted_slot >= 0:
            dynamic_column = BASE_FEATURE_DIM + promoted_slot
            descriptor_still_routed = (
                tuple(map(int, state.router.descriptors[promoted_slot])) == promoted_descriptor
            )
            downstream_use_observed = bool(
                descriptor_still_routed
                and state.interaction.active_output_memory_committed[promoted_slot]
                and state.consumer_active_mask[promoted_slot]
                and jnp.abs(state.chi[dynamic_column]) > 0.0
                and (
                    jnp.any(jnp.abs(state.behavior.weights[:, dynamic_column]) > 0.0)
                    or jnp.any(jnp.abs(state.control.q_weights[:, dynamic_column]) > 0.0)
                )
            )
            if downstream_use_observed:
                break

    assert promoted_slot >= 0
    assert promoted_descriptor is not None
    assert downstream_use_observed


@pytest.mark.parametrize(
    "invalid_case",
    (
        "observation_mismatch",
        "observation_nonfinite",
        "next_observation_nonfinite",
        "focal_action_mismatch",
        "focal_action_out_of_range",
        "partner_action_out_of_range",
        "reward_nonfinite",
        "reward_semantics",
        "outcome_nonfinite",
        "outcome_out_of_range",
        "discount_nonfinite",
        "discount_semantics",
        "terminated",
    ),
)
def test_invalid_transition_is_an_explicit_atomic_noop(invalid_case: str) -> None:
    agent = IntegratedHiddenPartnerAgent(IntegratedHiddenPartnerConfig(replacement_interval=0))
    start, transition, _ = _start_and_transition(agent, seed=80)
    replacement: dict[str, Any]
    if invalid_case == "observation_mismatch":
        replacement = {"observation": transition.observation.at[0].add(1.0)}
    elif invalid_case == "observation_nonfinite":
        replacement = {"observation": transition.observation.at[0].set(jnp.nan)}
    elif invalid_case == "next_observation_nonfinite":
        replacement = {"next_observation": transition.next_observation.at[0].set(jnp.inf)}
    elif invalid_case == "focal_action_mismatch":
        replacement = {
            "focal_action": 1 - start.state.control.last_action,
        }
    elif invalid_case == "focal_action_out_of_range":
        replacement = {"focal_action": jnp.asarray(-1, dtype=jnp.int32)}
    elif invalid_case == "partner_action_out_of_range":
        replacement = {"partner_action": jnp.asarray(2, dtype=jnp.int32)}
    elif invalid_case == "reward_nonfinite":
        replacement = {"reward": jnp.asarray(jnp.nan, dtype=jnp.float32)}
    elif invalid_case == "reward_semantics":
        replacement = {"reward": 1.0 - transition.reward}
    elif invalid_case == "outcome_nonfinite":
        replacement = {"outcome": jnp.asarray(jnp.inf, dtype=jnp.float32)}
    elif invalid_case == "outcome_out_of_range":
        replacement = {"outcome": jnp.asarray(0.0, dtype=jnp.float32)}
    elif invalid_case == "discount_nonfinite":
        replacement = {"discount": jnp.asarray(jnp.nan, dtype=jnp.float32)}
    elif invalid_case == "discount_semantics":
        replacement = {"discount": jnp.asarray(0.0, dtype=jnp.float32)}
    else:
        replacement = {"terminated": jnp.asarray(True, dtype=jnp.bool_)}

    result = agent.update(start.state, transition.replace(**replacement))

    chex.assert_trees_all_equal(result.state, start.state)
    chex.assert_trees_all_equal(result.action, start.state.control.last_action)
    assert bool(result.diagnostics.transition_rejected)
    assert not bool(result.diagnostics.transition_semantics_valid)
    assert int(result.diagnostics.interaction_replaced_slot) == -1
    assert int(result.diagnostics.interaction_promoted_candidate) == -1
    assert int(result.diagnostics.interaction_retired_slot) == -1
    assert not bool(result.diagnostics.route.route_applied)
    assert not bool(result.diagnostics.route.descriptors_changed)
    assert not bool(jnp.any(result.diagnostics.interaction_evidence_refreshed))
    assert not bool(jnp.any(result.diagnostics.consumer_read_acquire_pre))
    assert not bool(jnp.any(result.diagnostics.consumer_confirmed_write_pre))
    for delta in (
        result.diagnostics.state_builder_step_delta,
        result.diagnostics.state_builder_learning_delta,
        result.diagnostics.behavior_step_delta,
        result.diagnostics.interaction_step_delta,
        result.diagnostics.world_step_delta,
        result.diagnostics.control_step_delta,
        result.diagnostics.router_route_delta,
        result.diagnostics.router_generation_delta,
        result.diagnostics.integrated_step_delta,
    ):
        assert int(delta) == 0


def test_public_array_contracts_reject_static_shape_and_dtype_errors() -> None:
    agent = IntegratedHiddenPartnerAgent()
    environment = _environment()
    environment_state = environment.init(jr.key(81))
    observation = environment.observe(environment_state)
    descriptors = jnp.asarray(INITIAL_ACTIVE_DESCRIPTORS, dtype=jnp.int32)

    with pytest.raises(ValueError, match="raw_observation must have shape"):
        agent.start(observation.reshape((2, 4)), jr.key(82))
    with pytest.raises(TypeError, match="raw_observation must have dtype float32"):
        agent.start(observation.astype(jnp.int32), jr.key(82))
    with pytest.raises(ValueError, match="finite"):
        agent.start(observation.at[0].set(jnp.nan), jr.key(82))
    with pytest.raises(ValueError, match="phi must have shape"):
        agent.build_chi(jnp.ones((3, 4), dtype=jnp.float32), descriptors)
    with pytest.raises(TypeError, match="descriptors must have dtype int32"):
        agent.build_chi(
            jnp.ones((BASE_FEATURE_DIM,), dtype=jnp.float32),
            descriptors.astype(jnp.float32),
        )
    with pytest.raises(ValueError, match="chi_gradient must have shape"):
        agent.chain_chi_gradient_to_phi(
            jnp.ones((BASE_FEATURE_DIM,), dtype=jnp.float32),
            descriptors,
            jnp.ones((2, DEPLOYED_FEATURE_DIM // 2), dtype=jnp.float32),
        )

    start, transition, _ = _start_and_transition(agent, seed=82)
    with pytest.raises(ValueError, match="transition.observation must have shape"):
        agent.update(
            start.state,
            transition.replace(observation=transition.observation.reshape((2, 4))),
        )
    with pytest.raises(TypeError, match="transition.focal_action must have dtype int32"):
        agent.update(
            start.state,
            transition.replace(focal_action=transition.focal_action.astype(jnp.float32)),
        )
    with pytest.raises(TypeError, match="transition.terminated must have dtype bool"):
        agent.update(
            start.state,
            transition.replace(terminated=transition.terminated.astype(jnp.int32)),
        )


def test_stateless_public_kernels_return_neutral_output_for_dynamic_invalidity() -> None:
    agent = IntegratedHiddenPartnerAgent()
    phi = jnp.ones((BASE_FEATURE_DIM,), dtype=jnp.float32)
    descriptors = jnp.asarray(INITIAL_ACTIVE_DESCRIPTORS, dtype=jnp.int32)
    gradient = jnp.ones((DEPLOYED_FEATURE_DIM,), dtype=jnp.float32)

    invalid_descriptors = descriptors.at[0].set(jnp.asarray((-1, 0), dtype=jnp.int32))
    chex.assert_trees_all_equal(
        agent.build_chi(phi.at[0].set(jnp.nan), descriptors),
        jnp.zeros((DEPLOYED_FEATURE_DIM,), dtype=jnp.float32),
    )
    chex.assert_trees_all_equal(
        agent.build_chi(phi, invalid_descriptors),
        jnp.zeros((DEPLOYED_FEATURE_DIM,), dtype=jnp.float32),
    )
    chex.assert_trees_all_equal(
        agent.chain_chi_gradient_to_phi(
            phi,
            descriptors,
            gradient.at[0].set(jnp.inf),
        ),
        jnp.zeros((BASE_FEATURE_DIM,), dtype=jnp.float32),
    )


def test_all_integrated_long_lived_counters_saturate_without_replacement() -> None:
    agent = IntegratedHiddenPartnerAgent(IntegratedHiddenPartnerConfig(replacement_interval=0))
    start, transition, _ = _start_and_transition(agent, seed=83)
    maximum = jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=jnp.int32)
    interaction = start.state.interaction.replace(
        ages=jnp.full_like(start.state.interaction.ages, maximum),
        candidate_ages=jnp.full_like(start.state.interaction.candidate_ages, maximum),
        evidence_idle_steps=jnp.full_like(
            start.state.interaction.evidence_idle_steps,
            maximum,
        ),
        utility_evidence_streak=jnp.full_like(
            start.state.interaction.utility_evidence_streak,
            maximum,
        ),
        candidate_promotion_evidence_streak=jnp.full_like(
            start.state.interaction.candidate_promotion_evidence_streak,
            maximum,
        ),
        step_count=maximum,
    )
    state = start.state.replace(
        state_builder=start.state.state_builder.replace(
            step_count=maximum,
            update_count=maximum,
        ),
        interaction=interaction,
        behavior=start.state.behavior.replace(step_count=maximum),
        joint_world=start.state.joint_world.replace(
            visit_counts=jnp.full_like(
                start.state.joint_world.visit_counts,
                maximum,
            ),
            step_count=maximum,
        ),
        control=start.state.control.replace(step_count=maximum),
        router=dataclasses.replace(
            start.state.router,
            route_count=maximum,
            generation_count=maximum,
        ),
        consumer_evidence_streak=jnp.full_like(
            start.state.consumer_evidence_streak,
            maximum,
        ),
        consumer_read_idle_steps=jnp.full_like(
            start.state.consumer_read_idle_steps,
            maximum,
        ),
        step_count=maximum,
    )

    result = agent.update(state, transition)

    assert not bool(result.diagnostics.transition_rejected)
    assert int(result.state.step_count) == int(maximum)
    assert int(result.state.state_builder.step_count) == int(maximum)
    assert int(result.state.state_builder.update_count) == int(maximum)
    assert int(result.state.interaction.step_count) == int(maximum)
    assert bool(jnp.all(result.state.interaction.ages == maximum))
    assert bool(jnp.all(result.state.interaction.candidate_ages == maximum))
    assert int(result.state.behavior.step_count) == int(maximum)
    assert int(result.state.joint_world.step_count) == int(maximum)
    assert bool(jnp.all(result.state.joint_world.visit_counts == maximum))
    assert int(result.state.control.step_count) == int(maximum)
    assert int(result.state.router.route_count) == int(maximum)
    assert int(result.state.router.generation_count) == int(maximum)


def test_default_grounded_lane_is_absent_and_preserves_legacy_state_bytes() -> None:
    agent = IntegratedHiddenPartnerAgent()
    start, _, _ = _start_and_transition(agent, seed=8101)
    budget = agent.resource_budget(start.state)

    assert start.state.grounded_world is None
    assert start.state.current_evaluation.grounded_world is None
    assert agent.grounded_world_model is None
    assert agent.config.grounded_world_model is None
    assert agent.config.representation_gradient_mixer is None
    assert agent.config.grounded_world_learning_enabled
    assert not agent.config.grounded_world_planning_enabled
    assert budget.grounded_world_nbytes == 0
    assert budget.grounded_world_parameter_count == 0
    assert budget.grounded_world_parameters_touched_per_update == 0
    assert budget.grounded_world_update_counter_nbytes == 0
    assert budget.grounded_world_joint_cells_per_decision == 0
    assert budget.planner_cell_evaluations_per_decision == 4
    assert budget.decision_cache_nbytes == 303
    assert budget.total_state_nbytes == _tree_array_nbytes(start.state) == 6757


def test_grounded_nested_configs_roundtrip_and_reject_incomplete_or_wrong_shapes() -> None:
    config = _grounded_integrated_config(
        mode="world_only",
        grounded_planning=True,
        grounded_world_learning_enabled=False,
    )
    restored = IntegratedHiddenPartnerConfig.from_config(config.to_config())
    agent = IntegratedHiddenPartnerAgent(restored)

    assert restored == config
    assert not restored.grounded_world_learning_enabled
    assert agent.grounded_world_model is not None
    assert agent.grounded_world_model.config == config.grounded_world_model
    assert agent.to_config()["grounded_world"] == agent.grounded_world_model.to_config()
    assert (
        agent.to_config()["representation_gradient_mixer"]
        == config.representation_gradient_mixer.to_config()
    )

    with pytest.raises(ValueError, match="together"):
        IntegratedHiddenPartnerConfig(
            grounded_world_model=config.grounded_world_model,
        )
    with pytest.raises(ValueError, match="together"):
        IntegratedHiddenPartnerConfig(
            representation_gradient_mixer=config.representation_gradient_mixer,
        )
    with pytest.raises(ValueError, match="requires"):
        IntegratedHiddenPartnerConfig(grounded_world_planning_enabled=True)
    with pytest.raises(ValueError, match="representation_dim"):
        IntegratedHiddenPartnerConfig(
            grounded_world_model=dataclasses.replace(
                config.grounded_world_model,
                representation_dim=DEPLOYED_FEATURE_DIM - 1,
            ),
            representation_gradient_mixer=config.representation_gradient_mixer,
        )


@pytest.mark.parametrize(
    ("mode", "behavior_active", "world_active", "mix_applied"),
    [
        ("full", True, True, True),
        ("behavior_only", True, False, True),
        ("world_only", False, True, True),
        ("discard", False, False, False),
    ],
)
def test_grounded_update_and_all_gradient_modes_are_prequential_and_shape_matched(
    mode: str,
    behavior_active: bool,
    world_active: bool,
    mix_applied: bool,
) -> None:
    agent = IntegratedHiddenPartnerAgent(_grounded_integrated_config(mode=mode))
    start, transition, _ = _start_and_transition(agent, seed=8102)
    before = start.state
    assert before.grounded_world is not None
    assert agent.grounded_world_model is not None
    prediction_before = agent.grounded_world_model.predict(
        before.grounded_world,
        before.chi,
        transition.focal_action,
        transition.partner_action,
    )
    result = agent.update(before, transition)
    grounded_update = result.diagnostics.grounded_world_update
    gradient_mix = result.diagnostics.gradient_mix

    assert grounded_update is not None
    assert gradient_mix is not None
    assert result.state.grounded_world is not None
    chex.assert_trees_all_close(
        grounded_update.prediction,
        prediction_before,
        atol=0.0,
        rtol=0.0,
    )
    joint_index = 2 * int(transition.focal_action) + int(transition.partner_action)
    cached_grounded = before.current_evaluation.grounded_world
    assert cached_grounded is not None
    assert int(grounded_update.prediction.joint_action_index) == joint_index
    chex.assert_trees_all_equal(
        grounded_update.prediction.raw_predictions,
        cached_grounded.grounded_raw_predictions[joint_index],
    )
    chex.assert_trees_all_equal(
        grounded_update.prediction.raw_predictions,
        grounded_update.prediction.feature_contribution + grounded_update.prediction.row_bias,
    )
    assert bool(result.diagnostics.grounded_world_prediction_matches_decision)
    assert bool(grounded_update.diagnostics.applied)
    assert int(result.state.grounded_world.update_count) == 1
    assert int(result.diagnostics.grounded_world_step_delta) == 1
    assert not bool(
        jnp.array_equal(
            result.state.grounded_world.weights,
            before.grounded_world.weights,
        )
    )
    assert not bool(
        jnp.array_equal(
            result.state.grounded_world.bias,
            before.grounded_world.bias,
        )
    )
    assert bool(gradient_mix.valid)
    assert bool(gradient_mix.applied) is mix_applied
    assert bool(gradient_mix.diagnostics.behavior_active) is behavior_active
    assert bool(gradient_mix.diagnostics.grounded_world_active) is world_active
    chex.assert_trees_all_equal(
        result.diagnostics.mixed_gradient_chi,
        gradient_mix.gradient,
    )
    expected_phi_gradient = agent.chain_chi_gradient_to_phi(
        before.phi,
        before.router.descriptors,
        gradient_mix.gradient,
        agent._effective_pair_read_mask(  # noqa: SLF001
            before.interaction,
            before.consumer_active_mask,
        ),
    )
    chex.assert_trees_all_close(
        result.diagnostics.mixed_gradient_phi,
        expected_phi_gradient,
    )
    expected_behavior_phi = agent.chain_chi_gradient_to_phi(
        before.phi,
        before.router.descriptors,
        result.diagnostics.behavior_gradient_chi,
        agent._effective_pair_read_mask(  # noqa: SLF001
            before.interaction,
            before.consumer_active_mask,
        ),
    )
    chex.assert_trees_all_close(
        result.diagnostics.behavior_gradient_phi,
        expected_behavior_phi,
    )
    assert int(result.diagnostics.state_builder_learning_delta) == 1
    assert bool(result.diagnostics.transition_semantics_valid)
    assert bool(result.diagnostics.model_valid)


def test_world_gradient_has_a_causal_path_into_one_state_builder_learn_call() -> None:
    behavior_agent = IntegratedHiddenPartnerAgent(_grounded_integrated_config(mode="behavior_only"))
    world_agent = IntegratedHiddenPartnerAgent(_grounded_integrated_config(mode="world_only"))
    behavior_start, transition, _ = _start_and_transition(behavior_agent, seed=8103)
    world_start, world_transition, _ = _start_and_transition(world_agent, seed=8103)
    assert behavior_start.state.grounded_world is not None
    assert world_start.state.grounded_world is not None
    selected = int(transition.focal_action) * 2 + int(transition.partner_action)
    reward_head = RAW_OBSERVATION_DIM
    seeded_weights = (
        jnp.zeros_like(behavior_start.state.grounded_world.weights)
        .at[selected, reward_head, RAW_OBSERVATION_DIM]
        .set(1.0)
    )
    seeded_bias = (
        jnp.zeros_like(behavior_start.state.grounded_world.bias)
        .at[
            selected,
            reward_head,
        ]
        .set(2.0)
    )

    def seeded(agent: IntegratedHiddenPartnerAgent, state: Any) -> Any:
        assert state.grounded_world is not None
        behavior = state.behavior.replace(weights=jnp.zeros_like(state.behavior.weights))
        grounded = state.grounded_world.replace(
            weights=seeded_weights,
            bias=seeded_bias,
        )
        evaluation = agent.evaluate_models(
            behavior,
            state.joint_world,
            state.control,
            state.chi,
            grounded,
        )
        return state.replace(
            behavior=behavior,
            grounded_world=grounded,
            current_evaluation=evaluation,
        )

    behavior_state = seeded(behavior_agent, behavior_start.state)
    world_state = seeded(world_agent, world_start.state)
    behavior_result = behavior_agent.update(behavior_state, transition)
    world_result = world_agent.update(world_state, world_transition)

    chex.assert_trees_all_equal(
        behavior_result.diagnostics.behavior_gradient_chi,
        jnp.zeros((DEPLOYED_FEATURE_DIM,), dtype=jnp.float32),
    )
    assert (
        float(
            jnp.linalg.norm(world_result.diagnostics.grounded_world_update.representation_gradient)
        )
        > 0.0
    )
    assert float(jnp.linalg.norm(world_result.diagnostics.mixed_gradient_phi)) > 0.0
    assert float(world_result.diagnostics.state_learning.parameter_update_norm) > 0.0
    assert float(behavior_result.diagnostics.state_learning.parameter_update_norm) == 0.0
    assert not bool(
        jnp.array_equal(
            world_result.state.state_builder.parameters,
            behavior_result.state.state_builder.parameters,
        )
    )
    assert int(world_result.diagnostics.state_builder_learning_delta) == 1


def test_grounded_planner_evaluates_four_cells_and_static_mask_selects_reward_source() -> None:
    grounded_agent = IntegratedHiddenPartnerAgent(
        _grounded_integrated_config(
            grounded_planning=True,
            uniform_partner_belief=True,
            epsilon=0.0,
            planner_lambda=1.0,
        )
    )
    shadow_agent = IntegratedHiddenPartnerAgent(
        _grounded_integrated_config(
            grounded_planning=False,
            uniform_partner_belief=True,
            epsilon=0.0,
            planner_lambda=1.0,
        )
    )
    start, _, _ = _start_and_transition(grounded_agent, seed=8104)
    state = start.state
    assert state.grounded_world is not None
    reward_head = RAW_OBSERVATION_DIM
    grounded_rewards = jnp.asarray([[0.0, 0.0], [2.0, 2.0]], dtype=jnp.float32)
    grounded_state = state.grounded_world.replace(
        weights=jnp.zeros_like(state.grounded_world.weights),
        bias=state.grounded_world.bias.at[:, reward_head].set(grounded_rewards.reshape((-1,))),
    )
    table_rewards = jnp.asarray([[2.0, 2.0], [0.0, 0.0]], dtype=jnp.float32)
    table_state = state.joint_world.replace(reward_predictions=table_rewards)

    grounded = grounded_agent.evaluate_models(
        state.behavior,
        table_state,
        state.control,
        state.chi,
        grounded_state,
    )
    shadow = shadow_agent.evaluate_models(
        state.behavior,
        table_state,
        state.control,
        state.chi,
        grounded_state,
    )

    assert grounded.grounded_world is not None
    assert shadow.grounded_world is not None
    chex.assert_trees_all_equal(
        grounded.grounded_world.table_expected_rewards,
        jnp.asarray([2.0, 0.0], dtype=jnp.float32),
    )
    chex.assert_trees_all_equal(
        grounded.grounded_world.grounded_expected_rewards,
        jnp.asarray([0.0, 2.0], dtype=jnp.float32),
    )
    chex.assert_trees_all_equal(
        grounded.expected_rewards,
        grounded.grounded_world.grounded_expected_rewards,
    )
    chex.assert_trees_all_equal(
        shadow.expected_rewards,
        shadow.grounded_world.table_expected_rewards,
    )
    assert bool(grounded.grounded_world.predictions_valid)
    assert bool(grounded.grounded_world.planner_applied)
    assert not bool(shadow.grounded_world.planner_applied)
    assert int(grounded.grounded_world.cell_evaluations) == 4
    assert int(grounded.cell_evaluations) == 8
    assert int(shadow.cell_evaluations) == 8
    assert int(grounded.greedy_action) == 1
    assert int(shadow.greedy_action) == 0


def test_grounded_weights_join_identity_safe_route_while_bias_and_counter_do_not() -> None:
    agent = IntegratedHiddenPartnerAgent(_grounded_integrated_config())
    start, _, _ = _start_and_transition(agent, seed=8105)
    state = start.state
    assert state.grounded_world is not None
    old = state.router.descriptors
    proposed = (
        old.at[0].set(old[2]).at[1].set(old[0]).at[2].set(jnp.asarray([4, 5], dtype=jnp.int32))
    )
    grounded_weights = jnp.arange(
        state.grounded_world.weights.size,
        dtype=jnp.float32,
    ).reshape(state.grounded_world.weights.shape)
    grounded = state.grounded_world.replace(
        weights=grounded_weights,
        bias=jnp.arange(state.grounded_world.bias.size, dtype=jnp.float32).reshape(
            state.grounded_world.bias.shape
        ),
        update_count=jnp.asarray(7, dtype=jnp.int32),
    )
    routed_behavior, routed_control, routed_grounded, diagnostics, _ = (
        agent._route_feature_consumers_with_grounded(  # noqa: SLF001
            state.router,
            state.behavior,
            state.control,
            grounded,
            proposed,
        )
    )
    del routed_behavior, routed_control
    source = np.asarray([2, 0, -1] + list(range(3, ACTIVE_PAIR_SLOTS)))
    expected = np.asarray(grounded_weights).copy()
    expected_tail = np.zeros_like(expected[..., BASE_FEATURE_DIM:])
    survivor = source >= 0
    expected_tail[..., survivor] = np.asarray(grounded_weights)[
        ...,
        BASE_FEATURE_DIM + source[survivor],
    ]
    expected[..., BASE_FEATURE_DIM:] = expected_tail

    np.testing.assert_array_equal(routed_grounded.weights, expected)
    chex.assert_trees_all_equal(routed_grounded.bias, grounded.bias)
    chex.assert_trees_all_equal(routed_grounded.update_count, grounded.update_count)
    np.testing.assert_array_equal(diagnostics.source_slots, source)
    assert bool(diagnostics.valid)


def test_invalid_grounded_model_rejects_the_complete_integrated_transition() -> None:
    agent = IntegratedHiddenPartnerAgent(_grounded_integrated_config())
    start, transition, _ = _start_and_transition(agent, seed=8106)
    assert start.state.grounded_world is not None
    corrupt = start.state.replace(
        grounded_world=start.state.grounded_world.replace(
            weights=start.state.grounded_world.weights.at[0, 0, 0].set(jnp.nan)
        )
    )
    result = jax.jit(agent.update)(corrupt, transition)

    for before, after in zip(
        jax.tree_util.tree_leaves(corrupt),
        jax.tree_util.tree_leaves(result.state),
        strict=True,
    ):
        try:
            before_bytes = np.asarray(before).tobytes()
            after_bytes = np.asarray(after).tobytes()
        except TypeError:
            before_bytes = np.asarray(jr.key_data(before)).tobytes()
            after_bytes = np.asarray(jr.key_data(after)).tobytes()
        assert before_bytes == after_bytes
    assert result.diagnostics.grounded_world_update is not None
    assert result.diagnostics.gradient_mix is not None
    assert not bool(result.diagnostics.grounded_world_update.diagnostics.applied)
    assert bool(result.diagnostics.gradient_mix.rejected)
    assert bool(result.diagnostics.transition_rejected)
    assert not bool(result.diagnostics.transition_semantics_valid)
    assert int(result.diagnostics.grounded_world_step_delta) == 0
    for delta in (
        result.diagnostics.state_builder_step_delta,
        result.diagnostics.state_builder_learning_delta,
        result.diagnostics.behavior_step_delta,
        result.diagnostics.interaction_step_delta,
        result.diagnostics.world_step_delta,
        result.diagnostics.control_step_delta,
        result.diagnostics.router_route_delta,
        result.diagnostics.integrated_step_delta,
    ):
        assert int(delta) == 0


def test_invalid_unexecuted_grounded_joint_row_rejects_candidate_planner_atomically() -> None:
    agent = IntegratedHiddenPartnerAgent(_grounded_integrated_config())
    start, transition, _ = _start_and_transition(agent, seed=8107)
    assert start.state.grounded_world is not None
    executed = int(transition.focal_action) * 2 + int(transition.partner_action)
    unexecuted = (executed + 1) % 4
    adversarial_row = jnp.broadcast_to(
        jnp.where(start.state.chi >= 0.0, 100.0, -100.0),
        start.state.grounded_world.weights[unexecuted].shape,
    )
    adversarial = start.state.replace(
        grounded_world=start.state.grounded_world.replace(
            weights=start.state.grounded_world.weights.at[unexecuted].set(adversarial_row)
        )
    )

    result = agent.update(adversarial, transition)

    chex.assert_trees_all_equal(result.state, adversarial)
    assert result.diagnostics.grounded_world_update is not None
    assert result.diagnostics.gradient_mix is not None
    assert bool(result.diagnostics.grounded_world_update.diagnostics.applied)
    assert bool(result.diagnostics.gradient_mix.valid)
    assert result.diagnostics.current_evaluation.grounded_world is not None
    assert result.diagnostics.next_evaluation.grounded_world is not None
    assert bool(result.diagnostics.current_evaluation.grounded_world.predictions_valid)
    assert not bool(result.diagnostics.next_evaluation.grounded_world.predictions_valid)
    assert bool(result.diagnostics.transition_rejected)
    assert not bool(result.diagnostics.transition_semantics_valid)
    assert not bool(result.diagnostics.model_valid)
    assert int(result.diagnostics.grounded_world_step_delta) == 0


def test_enabled_gradient_modes_have_exact_resource_and_shape_parity_under_jit() -> None:
    totals = []
    states = []
    for index, mode in enumerate(("full", "behavior_only", "world_only", "discard")):
        agent = IntegratedHiddenPartnerAgent(_grounded_integrated_config(mode=mode))
        start, transition, _ = _start_and_transition(agent, seed=8200 + index)
        result = jax.jit(agent.update)(start.state, transition)
        budget = agent.resource_budget(start.state)
        assert start.state.grounded_world is not None
        assert budget.grounded_world_nbytes == _tree_array_nbytes(start.state.grounded_world)
        assert budget.grounded_world_parameter_count == (
            agent.grounded_world_model.resource_budget.trainable_float32_scalars
        )
        assert budget.grounded_world_parameters_touched_per_update == (
            agent.grounded_world_model.resource_budget.learned_float32_scalars_touched_per_update
        )
        assert budget.grounded_world_update_counter_nbytes == 4
        assert budget.grounded_world_joint_cells_per_decision == 4
        assert budget.planner_cell_evaluations_per_decision == 8
        assert budget.decision_cache_nbytes == 501
        assert budget.total_state_nbytes == 10_959
        assert budget.total_state_nbytes == _tree_array_nbytes(start.state)
        assert bool(result.diagnostics.all_finite)
        totals.append(budget.total_state_nbytes)
        states.append(jax.tree_util.tree_structure(start.state))

    assert len(set(totals)) == 1
    assert all(structure == states[0] for structure in states[1:])


def test_grounded_learning_freeze_preserves_compute_and_every_other_update() -> None:
    learning_agent = IntegratedHiddenPartnerAgent(
        _grounded_integrated_config(grounded_world_learning_enabled=True)
    )
    frozen_agent = IntegratedHiddenPartnerAgent(
        _grounded_integrated_config(grounded_world_learning_enabled=False)
    )
    learning_start, transition, _ = _start_and_transition(learning_agent, seed=8301)
    frozen_start, frozen_transition, _ = _start_and_transition(frozen_agent, seed=8301)
    chex.assert_trees_all_equal(learning_start.state, frozen_start.state)
    assert learning_start.state.grounded_world is not None

    learned = learning_agent.update(learning_start.state, transition)
    frozen = frozen_agent.update(frozen_start.state, frozen_transition)
    assert learned.diagnostics.grounded_world_update is not None
    assert frozen.diagnostics.grounded_world_update is not None
    assert learned.diagnostics.gradient_mix is not None
    assert frozen.diagnostics.gradient_mix is not None
    assert learned.state.grounded_world is not None
    assert frozen.state.grounded_world is not None

    assert bool(learned.diagnostics.grounded_world_learning_enabled)
    assert not bool(frozen.diagnostics.grounded_world_learning_enabled)
    assert bool(learned.diagnostics.grounded_world_update.diagnostics.applied)
    assert bool(frozen.diagnostics.grounded_world_update.diagnostics.applied)
    chex.assert_trees_all_equal(
        learned.diagnostics.grounded_world_update,
        frozen.diagnostics.grounded_world_update,
    )
    chex.assert_trees_all_equal(
        learned.diagnostics.gradient_mix,
        frozen.diagnostics.gradient_mix,
    )
    chex.assert_trees_all_equal(
        learned.diagnostics.mixed_gradient_phi,
        frozen.diagnostics.mixed_gradient_phi,
    )
    chex.assert_trees_all_equal(
        frozen.diagnostics.grounded_world_update.state,
        learned.state.grounded_world,
    )
    chex.assert_trees_all_equal(frozen.state.grounded_world, frozen_start.state.grounded_world)
    assert int(learned.diagnostics.grounded_world_step_delta) == 1
    assert int(frozen.diagnostics.grounded_world_step_delta) == 0
    assert int(frozen.diagnostics.grounded_world_update.state.update_count) == 1
    assert bool(frozen.diagnostics.transition_semantics_valid)

    learned_without_ground = learned.state.replace(
        grounded_world=None,
        current_evaluation=learned.state.current_evaluation.replace(grounded_world=None),
    )
    frozen_without_ground = frozen.state.replace(
        grounded_world=None,
        current_evaluation=frozen.state.current_evaluation.replace(grounded_world=None),
    )
    chex.assert_trees_all_equal(learned_without_ground, frozen_without_ground)
    assert (
        learning_agent.resource_budget(learning_start.state).total_state_nbytes
        == frozen_agent.resource_budget(frozen_start.state).total_state_nbytes
    )
    assert jax.tree_util.tree_structure(learning_start.state) == jax.tree_util.tree_structure(
        frozen_start.state
    )


def test_start_rejects_finite_but_invalid_grounded_planner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _grounded_integrated_config()
    assert config.grounded_world_model is not None
    config = dataclasses.replace(
        config,
        grounded_world_model=dataclasses.replace(
            config.grounded_world_model,
            max_input_magnitude=1.0,
            max_parameter_magnitude=2.0,
        ),
    )
    agent = IntegratedHiddenPartnerAgent(config)
    model = agent.grounded_world_model
    assert model is not None
    original_init = model.init

    def invalid_init(key: Any) -> Any:
        state = original_init(key)
        return state.replace(
            weights=jnp.zeros_like(state.weights),
            bias=jnp.full_like(state.bias, 2.0),
        )

    monkeypatch.setattr(model, "init", invalid_init)
    environment = _environment()
    environment_state = environment.init(jr.key(8302))
    with pytest.raises(ValueError, match="initial planner evaluation"):
        agent.start(environment.observe(environment_state), jr.key(18302))


def test_invalid_router_transaction_rejects_every_integrated_update() -> None:
    agent = IntegratedHiddenPartnerAgent(_grounded_integrated_config())
    start, transition, _ = _start_and_transition(agent, seed=8303)
    invalid = start.state.replace(
        router=dataclasses.replace(
            start.state.router,
            route_count=jnp.asarray(-1, dtype=jnp.int32),
        )
    )

    result = agent.update(invalid, transition)

    chex.assert_trees_all_equal(result.state, invalid)
    assert not bool(result.diagnostics.route.valid)
    assert bool(result.diagnostics.route.counter_invalid)
    assert bool(result.diagnostics.transition_rejected)
    assert not bool(result.diagnostics.transition_semantics_valid)
    assert int(result.diagnostics.integrated_step_delta) == 0
    assert int(result.diagnostics.grounded_world_step_delta) == 0


@pytest.mark.parametrize("checkpoint", ("behavior", "grounded"))
def test_replaced_checkpoint_must_match_the_cached_behavior_and_grounded_decision(
    checkpoint: str,
) -> None:
    agent = IntegratedHiddenPartnerAgent(_grounded_integrated_config())
    start, transition, _ = _start_and_transition(agent, seed=8304)
    assert start.state.grounded_world is not None
    if checkpoint == "behavior":
        replaced = start.state.replace(
            behavior=start.state.behavior.replace(bias=jnp.asarray([4.0, -4.0], dtype=jnp.float32))
        )
    else:
        executed = int(transition.focal_action) * 2 + int(transition.partner_action)
        unexecuted = (executed + 1) % 4
        reward_head = RAW_OBSERVATION_DIM
        replaced = start.state.replace(
            grounded_world=start.state.grounded_world.replace(
                bias=start.state.grounded_world.bias.at[unexecuted, reward_head].add(1.0)
            )
        )

    result = agent.update(replaced, transition)

    chex.assert_trees_all_equal(result.state, replaced)
    assert bool(result.diagnostics.transition_rejected)
    assert bool(result.diagnostics.behavior_prediction_matches_decision) is (
        checkpoint != "behavior"
    )
    assert bool(result.diagnostics.grounded_world_prediction_matches_decision) is (
        checkpoint != "grounded"
    )


def test_grounded_counter_saturates_without_stopping_continual_updates() -> None:
    agent = IntegratedHiddenPartnerAgent(_grounded_integrated_config())
    start, transition, _ = _start_and_transition(agent, seed=8305)
    assert start.state.grounded_world is not None
    maximum = jnp.asarray(2**31 - 1, dtype=jnp.int32)
    saturated = start.state.replace(
        grounded_world=start.state.grounded_world.replace(update_count=maximum)
    )

    result = agent.update(saturated, transition)

    assert result.state.grounded_world is not None
    assert result.diagnostics.grounded_world_update is not None
    assert bool(result.diagnostics.grounded_world_counter_saturated)
    assert bool(result.diagnostics.grounded_world_update.diagnostics.applied)
    assert bool(result.diagnostics.transition_semantics_valid)
    assert int(result.state.grounded_world.update_count) == int(maximum)
    assert int(result.diagnostics.grounded_world_step_delta) == 0
    assert not bool(
        jnp.array_equal(
            result.state.grounded_world.weights,
            saturated.grounded_world.weights,
        )
    )
    assert int(result.diagnostics.integrated_step_delta) == 1


def test_global_planning_ablation_reports_grounded_planner_not_applied() -> None:
    agent = IntegratedHiddenPartnerAgent(
        _grounded_integrated_config(
            grounded_planning=True,
            planning_enabled=False,
        )
    )
    start, _, _ = _start_and_transition(agent, seed=8306)
    grounded = start.state.current_evaluation.grounded_world

    assert grounded is not None
    assert not bool(grounded.planner_applied)
    chex.assert_trees_all_equal(
        start.state.current_evaluation.applied_model_term,
        jnp.zeros((2,), dtype=jnp.float32),
    )


def test_nonfinite_non_ground_candidate_is_an_atomic_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = IntegratedHiddenPartnerAgent(_grounded_integrated_config())
    start, transition, _ = _start_and_transition(agent, seed=8307)
    original_update = agent.interaction_learner.update

    def nonfinite_update(*args: Any, **kwargs: Any) -> Any:
        update = original_update(*args, **kwargs)
        return update.replace(
            state=update.state.replace(output_biases=update.state.output_biases.at[0].set(jnp.nan))
        )

    monkeypatch.setattr(agent.interaction_learner, "update", nonfinite_update)
    result = agent.update(start.state, transition)

    chex.assert_trees_all_equal(result.state, start.state)
    assert bool(result.diagnostics.transition_rejected)
    assert not bool(result.diagnostics.all_finite)
    assert int(result.diagnostics.integrated_step_delta) == 0


def test_current_invalid_grounded_cell_is_rejected_even_when_cache_is_coherent() -> None:
    agent = IntegratedHiddenPartnerAgent(_grounded_integrated_config())
    start, transition, _ = _start_and_transition(agent, seed=8308)
    assert start.state.grounded_world is not None
    executed = int(transition.focal_action) * 2 + int(transition.partner_action)
    unexecuted = (executed + 1) % 4
    invalid_grounded = start.state.grounded_world.replace(
        bias=start.state.grounded_world.bias.at[unexecuted].set(
            jnp.full_like(start.state.grounded_world.bias[unexecuted], 100.0)
        )
    )
    invalid_evaluation = agent.evaluate_models(
        start.state.behavior,
        start.state.joint_world,
        start.state.control,
        start.state.chi,
        invalid_grounded,
    )
    assert invalid_evaluation.grounded_world is not None
    assert not bool(invalid_evaluation.grounded_world.predictions_valid)
    coherent_invalid = start.state.replace(
        grounded_world=invalid_grounded,
        current_evaluation=invalid_evaluation,
    )

    result = agent.update(coherent_invalid, transition)

    chex.assert_trees_all_equal(result.state, coherent_invalid)
    assert bool(result.diagnostics.grounded_world_prediction_matches_decision)
    assert bool(result.diagnostics.transition_rejected)
    assert not bool(result.diagnostics.model_valid)


def test_grounded_evidence_gated_no_carry_route_is_one_atomic_transaction() -> None:
    agent = IntegratedHiddenPartnerAgent(
        _grounded_integrated_config(
            feature_lifecycle_enabled=True,
            replacement_interval=64,
            evidence_gated_consumer_memory=True,
            active_utility_retention_grace_steps=32,
            active_utility_evidence_threshold=0.01,
            carry_survivors=False,
        )
    )
    start, _, _ = _start_and_transition(agent, seed=8309)
    state = start.state
    assert state.grounded_world is not None
    behavior = state.behavior.replace(weights=jnp.ones_like(state.behavior.weights))
    control = state.control.replace(
        q_weights=jnp.full_like(state.control.q_weights, 2.0),
        q_trace_weights=jnp.full_like(state.control.q_trace_weights, 3.0),
        last_observation=jnp.full_like(state.control.last_observation, 4.0),
    )
    grounded = state.grounded_world.replace(
        weights=jnp.full_like(state.grounded_world.weights, 5.0),
        bias=jnp.full_like(state.grounded_world.bias, 6.0),
        update_count=jnp.asarray(7, dtype=jnp.int32),
    )
    old = state.router.descriptors
    proposed = (
        old.at[0].set(old[2]).at[1].set(old[0]).at[2].set(jnp.asarray([4, 5], dtype=jnp.int32))
    )

    routed_behavior, routed_control, routed_grounded, diagnostics, _ = (
        agent._route_feature_consumers_with_grounded(  # noqa: SLF001
            state.router,
            behavior,
            control,
            grounded,
            proposed,
        )
    )

    for routed in (
        routed_behavior.weights,
        routed_control.q_weights,
        routed_control.q_trace_weights,
        routed_control.last_observation,
        routed_grounded.weights,
    ):
        chex.assert_trees_all_equal(
            routed[..., BASE_FEATURE_DIM:],
            jnp.zeros_like(routed[..., BASE_FEATURE_DIM:]),
        )
    chex.assert_trees_all_equal(routed_grounded.bias, grounded.bias)
    chex.assert_trees_all_equal(routed_grounded.update_count, grounded.update_count)
    assert bool(diagnostics.valid)
    assert bool(diagnostics.descriptors_changed)
    assert not bool(diagnostics.carry_survivors)

    def audit(q_after: jax.Array) -> Any:
        return agent._audit_consumer_identity_route(  # noqa: SLF001
            old_descriptors=old,
            new_descriptors=proposed,
            route=diagnostics,
            behavior_before=behavior.weights,
            behavior_after=routed_behavior.weights,
            q_before=control.q_weights,
            q_after=q_after,
            trace_before=control.q_trace_weights,
            trace_after=routed_control.q_trace_weights,
            last_observation_before=control.last_observation,
            last_observation_after=routed_control.last_observation,
            grounded_before=grounded.weights,
            grounded_after=routed_grounded.weights,
            retired_slot=jnp.asarray(-1, dtype=jnp.int32),
            replaced_slot=jnp.asarray(2, dtype=jnp.int32),
        )

    corrupted_tail = routed_control.q_weights.at[..., BASE_FEATURE_DIM:].set(1.0)
    for evaluate in (audit, jax.jit(audit)):
        valid = evaluate(routed_control.q_weights)
        assert bool(valid.no_carry_reset_exact)
        assert bool(valid.values_exact)
        corrupted = evaluate(corrupted_tail)
        assert not bool(corrupted.no_carry_reset_exact)
        assert not bool(corrupted.q_values_exact)
        assert not bool(corrupted.values_exact)


def test_invalid_grounded_transition_has_eager_jit_and_scan_atomic_parity() -> None:
    agent = IntegratedHiddenPartnerAgent(_grounded_integrated_config())
    start, transition, _ = _start_and_transition(agent, seed=8310)
    invalid = transition.replace(reward=1.0 - transition.reward)
    eager = agent.update(start.state, invalid)
    compiled = jax.jit(agent.update)(start.state, invalid)

    chex.assert_trees_all_equal(eager, compiled)
    chex.assert_trees_all_equal(eager.state, start.state)

    def scan_step(state: Any, _: Any) -> tuple[Any, Any]:
        update = agent.update(state, invalid)
        return update.state, (
            update.diagnostics.transition_rejected,
            update.diagnostics.grounded_world_step_delta,
            update.diagnostics.integrated_step_delta,
        )

    final_state, (rejected, grounded_delta, integrated_delta) = jax.jit(
        lambda state: jax.lax.scan(scan_step, state, xs=None, length=2)
    )(start.state)
    chex.assert_trees_all_equal(final_state, start.state)
    chex.assert_trees_all_equal(rejected, jnp.ones((2,), dtype=jnp.bool_))
    chex.assert_trees_all_equal(grounded_delta, jnp.zeros((2,), dtype=jnp.int32))
    chex.assert_trees_all_equal(integrated_delta, jnp.zeros((2,), dtype=jnp.int32))


def test_complete_table_decision_cache_and_selection_mutations_fail_closed() -> None:
    agent = IntegratedHiddenPartnerAgent()
    start, transition, _ = _start_and_transition(agent, seed=8311)
    state = start.state
    evaluation = state.current_evaluation
    selection = state.current_selection
    one = jnp.asarray(1, dtype=jnp.int32)
    common_offset = jnp.full((2,), 0.125, dtype=jnp.float32)
    mutations = (
        state.replace(
            current_evaluation=evaluation.replace(
                predicted_partner_probabilities=(
                    evaluation.predicted_partner_probabilities.at[0].add(0.125)
                )
            )
        ),
        state.replace(
            current_evaluation=evaluation.replace(
                partner_probabilities=evaluation.partner_probabilities.at[0].add(0.125)
            )
        ),
        state.replace(
            current_evaluation=evaluation.replace(
                partner_probabilities_valid=~evaluation.partner_probabilities_valid
            )
        ),
        state.replace(
            current_evaluation=evaluation.replace(
                probability_violation=evaluation.probability_violation + 0.125
            )
        ),
        state.replace(
            current_evaluation=evaluation.replace(
                expected_rewards=evaluation.expected_rewards.at[0].set(jnp.nan)
            )
        ),
        state.replace(
            current_evaluation=evaluation.replace(
                expected_outcomes=evaluation.expected_outcomes.at[0, 0].add(0.125)
            )
        ),
        state.replace(
            current_evaluation=evaluation.replace(q_values=evaluation.q_values.at[0].add(0.125))
        ),
        state.replace(
            current_evaluation=evaluation.replace(
                q_values=evaluation.q_values + common_offset,
                planner_scores=evaluation.planner_scores + common_offset,
            )
        ),
        state.replace(
            current_q_value_delta=state.current_q_value_delta.at[0].add(0.125)
        ),
        state.replace(
            current_q_value_delta=state.current_q_value_delta.at[0].set(jnp.nan)
        ),
        state.replace(
            control=state.control.replace(
                last_observation=state.control.last_observation.at[0].add(0.125)
            )
        ),
        state.replace(
            control=state.control.replace(epsilon=state.control.epsilon + 0.125)
        ),
        state.replace(
            control=state.control.replace(q_bias=state.control.q_bias.at[0].add(0.125))
        ),
        state.replace(
            control=state.control.replace(
                q_trace_bias=state.control.q_trace_bias.at[0].add(0.125)
            )
        ),
        state.replace(control=state.control.replace(step_count=state.control.step_count + one)),
        state.replace(
            current_evaluation=evaluation.replace(
                centered_expected_rewards=evaluation.centered_expected_rewards.at[0].add(0.125)
            )
        ),
        state.replace(
            current_evaluation=evaluation.replace(model_term=evaluation.model_term.at[0].add(0.125))
        ),
        state.replace(
            current_evaluation=evaluation.replace(
                applied_model_term=evaluation.applied_model_term.at[0].add(0.125)
            )
        ),
        state.replace(
            current_evaluation=evaluation.replace(
                planner_scores=evaluation.planner_scores.at[0].add(0.125)
            )
        ),
        state.replace(
            current_evaluation=evaluation.replace(greedy_action=one - evaluation.greedy_action)
        ),
        state.replace(
            current_evaluation=evaluation.replace(
                cell_evaluations=evaluation.cell_evaluations + one
            )
        ),
        state.replace(current_selection=selection.replace(action=one - selection.action)),
        state.replace(
            current_selection=selection.replace(
                noisy_greedy_action=one - selection.noisy_greedy_action
            )
        ),
        state.replace(
            current_selection=selection.replace(random_action=one - selection.random_action)
        ),
        state.replace(current_selection=selection.replace(explored=~selection.explored)),
        state.replace(
            current_selection=selection.replace(
                externally_forced=~selection.externally_forced
            )
        ),
        state.replace(
            current_selection=selection.replace(
                rng_key_before=jr.fold_in(selection.rng_key_before, 1)
            )
        ),
        state.replace(
            current_selection=selection.replace(
                rng_key_after=jr.fold_in(selection.rng_key_after, 1)
            )
        ),
    )

    _assert_cache_mutations_reject_in_eager_jit_and_scan(agent, transition, mutations)


def test_current_q_value_delta_static_contract_rejects_shape_and_dtype() -> None:
    agent = IntegratedHiddenPartnerAgent()
    start, transition, _ = _start_and_transition(agent, seed=8313)

    with pytest.raises(ValueError, match="current_q_value_delta.*shape"):
        agent.update(
            start.state.replace(
                current_q_value_delta=jnp.zeros((1,), dtype=jnp.float32)
            ),
            transition,
        )
    with pytest.raises(TypeError, match="current_q_value_delta.*dtype"):
        agent.update(
            start.state.replace(
                current_q_value_delta=jnp.zeros((2,), dtype=jnp.int32)
            ),
            transition,
        )


def test_complete_grounded_decision_cache_mutations_fail_closed() -> None:
    agent = IntegratedHiddenPartnerAgent(_grounded_integrated_config(grounded_planning=True))
    start, transition, _ = _start_and_transition(agent, seed=8312)
    state = start.state
    evaluation = state.current_evaluation
    grounded = evaluation.grounded_world
    assert grounded is not None
    one = jnp.asarray(1, dtype=jnp.int32)
    mutations = (
        state.replace(
            current_evaluation=evaluation.replace(
                grounded_world=grounded.replace(
                    table_expected_rewards=grounded.table_expected_rewards.at[0].add(0.125)
                )
            )
        ),
        state.replace(
            current_evaluation=evaluation.replace(
                grounded_world=grounded.replace(
                    grounded_raw_predictions=grounded.grounded_raw_predictions.at[0, 0].set(
                        jnp.nan
                    )
                )
            )
        ),
        state.replace(
            current_evaluation=evaluation.replace(
                grounded_world=grounded.replace(
                    grounded_reward_cells=grounded.grounded_reward_cells.at[0, 0].add(0.125)
                )
            )
        ),
        state.replace(
            current_evaluation=evaluation.replace(
                grounded_world=grounded.replace(
                    grounded_expected_rewards=(grounded.grounded_expected_rewards.at[0].add(0.125))
                )
            )
        ),
        state.replace(
            current_evaluation=evaluation.replace(
                grounded_world=grounded.replace(predictions_valid=~grounded.predictions_valid)
            )
        ),
        state.replace(
            current_evaluation=evaluation.replace(
                grounded_world=grounded.replace(planner_applied=~grounded.planner_applied)
            )
        ),
        state.replace(
            current_evaluation=evaluation.replace(
                grounded_world=grounded.replace(cell_evaluations=grounded.cell_evaluations + one)
            )
        ),
    )

    _assert_cache_mutations_reject_in_eager_jit_and_scan(agent, transition, mutations)
