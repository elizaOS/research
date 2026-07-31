"""Focused L0 mechanism tests for the integrated hidden-partner kernel."""

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
from alberta_framework.streams.hidden_partner_mapping import (
    HiddenPartnerMappingConfig,
    HiddenPartnerMappingTransition,
    HiddenPartnerMappingWorld,
)


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


def _force_next_interaction_promotion(state: Any) -> Any:
    """Make candidate zero unambiguously replace active slot zero at step 64."""
    interaction = state.interaction
    candidate_utilities = (
        jnp.zeros(
            (CANDIDATE_PAIR_SLOTS,),
            dtype=jnp.float32,
        )
        .at[0]
        .set(10.0)
    )
    candidate_left = interaction.candidate_left.at[0].set(4)
    candidate_right = interaction.candidate_right.at[0].set(5)
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
            candidate_left=candidate_left,
            candidate_right=candidate_right,
        )
    )


def test_config_round_trip_and_exact_default_composition() -> None:
    config = IntegratedHiddenPartnerConfig()
    agent = IntegratedHiddenPartnerAgent(config)
    restored = IntegratedHiddenPartnerConfig.from_config(config.to_config())

    assert restored == config
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
    assert (
        agent.interaction_learner.to_config()["evidence_gated_active_output_memory"]
        is False
    )
    assert agent.interaction_learner.to_config()["utility_evidence_confirmation_steps"] == 1
    assert agent.interaction_learner.to_config()["independent_relevance_probe"] is False
    assert (
        agent.interaction_learner.to_config()["relevance_probe_mode"]
        == RELEVANCE_PROBE_MODE_CONDITIONAL_V1
    )
    assert agent.interaction_learner.to_config()["retire_stale_features"] is False
    assert (
        agent.interaction_learner.to_config()[
            "candidate_promotion_confirmation_steps"
        ]
        == 1
    )
    assert (
        agent.interaction_learner.to_config()[
            "candidate_reacquisition_confirmation_steps"
        ]
        == 1
    )
    assert agent.behavior_model.config.n_actions == 2
    assert agent.joint_world_model.resource_budget.joint_cells == 4
    assert agent.control_agent.config.n_actions == 2
    assert not agent.control_agent.config.use_bias
    assert agent.router.config.total_feature_dim == DEPLOYED_FEATURE_DIM == 24

    payload = config.to_config()
    assert payload["schema_version"] == "alberta.integrated-hidden-partner.l0.v10"
    assert payload["schema_version"] == INTEGRATED_HIDDEN_PARTNER_SCHEMA_VERSION
    assert payload["development_level"] == "L0"
    assert payload["accepted_scientific_evidence"] is False
    extra = copy.deepcopy(payload)
    extra["promoted_evidence"] = True
    with pytest.raises(ValueError, match="schema"):
        IntegratedHiddenPartnerConfig.from_config(extra)
    invalid_claim = copy.deepcopy(payload)
    invalid_claim["accepted_scientific_evidence"] = True
    with pytest.raises(ValueError, match="not accepted"):
        IntegratedHiddenPartnerConfig.from_config(invalid_claim)
    old_schema = copy.deepcopy(payload)
    old_schema["schema_version"] = "alberta.integrated-hidden-partner.l0.v9"
    with pytest.raises(ValueError, match="unsupported"):
        IntegratedHiddenPartnerConfig.from_config(old_schema)


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
    assert (
        agent.interaction_learner.to_config()[
            "candidate_reacquisition_confirmation_steps"
        ]
        == 4
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
        {"interaction_utility_decay": 0.0, "random_feature_curation": True},
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
            "evidence_gated_feature_memory": True,
            "feature_lifecycle_enabled": False,
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
        {
            "evidence_gated_consumer_memory": True,
            "feature_lifecycle_enabled": False,
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
        old.at[0]
        .set(old[2])
        .at[1]
        .set(old[0])
        .at[2]
        .set(jnp.asarray([4, 5], dtype=jnp.int32))
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
        evidence = jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.bool_).at[slot].set(
            has_evidence
        )
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
        idle_post = jnp.zeros((ACTIVE_PAIR_SLOTS,), dtype=jnp.int32).at[slot].set(
            idle_steps
        )
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
    products = (
        start.state.phi[descriptors[:, 0]]
        * start.state.phi[descriptors[:, 1]]
    )
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
    assert (
        agent.interaction_learner.to_config()[
            "candidate_promotion_confirmation_steps"
        ]
        == 3
    )
    slot = 0
    left, right = INITIAL_ACTIVE_DESCRIPTORS[slot]
    product = float(start.state.phi[left] * start.state.phi[right])
    partner_sign = 2.0 * float(transition.partner_action) - 1.0
    marginal_residual = partner_sign - float(
        start.state.interaction.relevance_probe_biases[0]
    )
    preupdate_probe = 0.5 * marginal_residual / product
    interaction = start.state.interaction.replace(
        output_weights=start.state.interaction.output_weights.at[0, slot].set(-0.0),
        relevance_probe_weights=(
            start.state.interaction.relevance_probe_weights.at[0, slot].set(
                preupdate_probe
            )
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
    assert int(
        np.asarray(first.state.interaction.output_weights[0, slot]).view(np.uint32)
    ) == 0

    environment = _environment()
    second_transition, _ = environment.step(next_environment_state, first.action)
    second_product = float(first.state.phi[left] * first.state.phi[right])
    second_sign = 2.0 * float(second_transition.partner_action) - 1.0
    second_residual = second_sign - float(
        first.state.interaction.relevance_probe_biases[0]
    )
    second_preupdate_probe = 0.5 * second_residual / second_product
    second_interaction = first.state.interaction.replace(
        relevance_probe_weights=(
            first.state.interaction.relevance_probe_weights.at[0, slot].set(
                second_preupdate_probe
            )
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
        jnp.all(
            second.diagnostics.interaction_candidate_promotion_evidence_streak_updated
            <= 3
        )
    )
    assert float(second.state.interaction.output_weights[0, slot]) == pytest.approx(
        second_preupdate_probe
    )
    assert bool(second.diagnostics.consumer_confirmed_write_pre[slot])
    expected_next_product = (
        second.state.phi[left] * second.state.phi[right]
    )
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
        output_weights=(
            recurrent_start.state.interaction.output_weights.at[0, slot].set(durable)
        ),
        relevance_probe_weights=(
            recurrent_start.state.interaction.relevance_probe_weights.at[0, slot].set(
                recurrent_probe
            )
        ),
        active_output_memory_committed=(
            recurrent_start.state.interaction.active_output_memory_committed.at[slot].set(
                True
            )
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
    expected_reacquired_product = (
        reacquired.state.phi[left] * reacquired.state.phi[right]
    )
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
    assert int(start.action) == int(selection.action)
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


def test_random_curation_priorities_replace_only_utility_ranking() -> None:
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
    )
    adversarial = interaction.replace(
        utilities=jnp.linspace(1e3, 2e3, ACTIVE_PAIR_SLOTS, dtype=jnp.float32),
        candidate_utilities=jnp.linspace(
            -2e3,
            -1e3,
            CANDIDATE_PAIR_SLOTS,
            dtype=jnp.float32,
        ),
    )
    ranked_a, active_priorities, candidate_priorities = agent._interaction_curation_input(
        interaction
    )  # noqa: SLF001
    ranked_b, active_priorities_b, candidate_priorities_b = agent._interaction_curation_input(
        adversarial
    )  # noqa: SLF001

    chex.assert_trees_all_equal(active_priorities, active_priorities_b)
    chex.assert_trees_all_equal(candidate_priorities, candidate_priorities_b)
    chex.assert_trees_all_equal(ranked_a.utilities, ranked_b.utilities)
    chex.assert_trees_all_equal(
        ranked_a.candidate_utilities,
        ranked_b.candidate_utilities,
    )
    assert len(np.unique(np.asarray(active_priorities))) == ACTIVE_PAIR_SLOTS
    assert len(np.unique(np.asarray(candidate_priorities))) == CANDIDATE_PAIR_SLOTS
    assert float(jnp.min(candidate_priorities)) > float(jnp.max(active_priorities)) * 0.0

    state_a = start.state.replace(interaction=interaction)
    state_b = start.state.replace(interaction=adversarial)
    result_a = agent.update(state_a, transition)
    result_b = agent.update(state_b, transition)
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
        int(result_a.diagnostics.interaction_replaced_slot)
        == int(result_b.diagnostics.interaction_replaced_slot)
        >= 0
    )
    assert (
        int(result_a.diagnostics.interaction_promoted_candidate)
        == int(result_b.diagnostics.interaction_promoted_candidate)
        >= 0
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


def test_feature_lifecycle_ablation_learns_shadow_without_deploying_replacement() -> None:
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
    enabled_state = _force_next_interaction_promotion(enabled_start.state)
    frozen_state = _force_next_interaction_promotion(frozen_start.state)

    enabled_result = enabled.update(enabled_state, transition)
    frozen_result = frozen.update(frozen_state, frozen_transition)

    assert int(enabled_result.diagnostics.interaction_replaced_slot) == 0
    assert int(frozen_result.diagnostics.interaction_replaced_slot) == 0
    assert bool(enabled_result.diagnostics.shadow_descriptors_changed)
    assert bool(frozen_result.diagnostics.shadow_descriptors_changed)
    assert bool(enabled_result.diagnostics.route.descriptors_changed)
    assert not bool(frozen_result.diagnostics.route.descriptors_changed)
    assert int(enabled_result.diagnostics.router_generation_delta) == 1
    assert int(frozen_result.diagnostics.router_generation_delta) == 0
    np.testing.assert_array_equal(
        enabled_result.state.router.descriptors[0],
        [4, 5],
    )
    np.testing.assert_array_equal(
        frozen_result.state.router.descriptors,
        frozen_start.state.router.descriptors,
    )
    np.testing.assert_array_equal(
        frozen_result.state.interaction.feature_left[0],
        4,
    )
    np.testing.assert_array_equal(
        frozen_result.state.interaction.feature_right[0],
        5,
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


def test_jitted_two_step_scan_preserves_causal_counters_and_finiteness() -> None:
    agent = IntegratedHiddenPartnerAgent()
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
        promoted_candidate = int(
            result.diagnostics.interaction_promoted_candidate
        )
        if promoted_candidate >= 0 and promoted_slot < 0:
            promoted_slot = int(result.diagnostics.interaction_replaced_slot)
            promoted_descriptor = (
                int(state.interaction.candidate_left[promoted_candidate]),
                int(state.interaction.candidate_right[promoted_candidate]),
            )
            assert abs(
                float(
                    state.interaction.candidate_output_weights[
                        0,
                        promoted_candidate,
                    ]
                )
            ) > 0.0
            assert bool(
                result.diagnostics.interaction_candidate_promotion_raw_evidence[
                    promoted_candidate
                ]
            )
            assert bool(
                result.diagnostics.interaction_candidate_promotion_confirmed[
                    promoted_candidate
                ]
            )
            assert int(
                result.diagnostics.interaction_candidate_promotion_evidence_streak_updated[
                    promoted_candidate
                ]
            ) >= 2
            assert bool(result.diagnostics.route.descriptors_changed)
            assert int(result.diagnostics.route.new_count) == 1
            assert tuple(map(int, result.state.router.descriptors[promoted_slot])) == (
                promoted_descriptor
            )

        state = result.state
        if promoted_slot >= 0:
            dynamic_column = BASE_FEATURE_DIM + promoted_slot
            descriptor_still_routed = tuple(
                map(int, state.router.descriptors[promoted_slot])
            ) == promoted_descriptor
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
    agent = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(replacement_interval=0)
    )
    start, transition, _ = _start_and_transition(agent, seed=80)
    replacement: dict[str, Any]
    if invalid_case == "observation_mismatch":
        replacement = {"observation": transition.observation.at[0].add(1.0)}
    elif invalid_case == "observation_nonfinite":
        replacement = {"observation": transition.observation.at[0].set(jnp.nan)}
    elif invalid_case == "next_observation_nonfinite":
        replacement = {
            "next_observation": transition.next_observation.at[0].set(jnp.inf)
        }
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
            transition.replace(
                focal_action=transition.focal_action.astype(jnp.float32)
            ),
        )
    with pytest.raises(TypeError, match="transition.terminated must have dtype bool"):
        agent.update(
            start.state,
            transition.replace(
                terminated=transition.terminated.astype(jnp.int32)
            ),
        )


def test_stateless_public_kernels_return_neutral_output_for_dynamic_invalidity() -> None:
    agent = IntegratedHiddenPartnerAgent()
    phi = jnp.ones((BASE_FEATURE_DIM,), dtype=jnp.float32)
    descriptors = jnp.asarray(INITIAL_ACTIVE_DESCRIPTORS, dtype=jnp.int32)
    gradient = jnp.ones((DEPLOYED_FEATURE_DIM,), dtype=jnp.float32)

    invalid_descriptors = descriptors.at[0].set(
        jnp.asarray((-1, 0), dtype=jnp.int32)
    )
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
    agent = IntegratedHiddenPartnerAgent(
        IntegratedHiddenPartnerConfig(replacement_interval=0)
    )
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
