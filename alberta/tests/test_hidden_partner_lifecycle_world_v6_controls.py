"""Static contracts for the nonexecuting noisy-world v6 control matrix."""

from __future__ import annotations

import json
import math

import pytest

from alberta_framework.core.integrated_hidden_partner import IntegratedHiddenPartnerConfig
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6 import (
    PRIMARY_CONDITION_ORDER,
    V6_INITIAL_ACTIVE_DESCRIPTORS,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_controls import (
    HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_CONTROL_SCHEMA,
    V6_DIAGNOSTIC_ORDER,
    V6_REPRESENTATION_LOSS_WEIGHTS,
    V6_TARGET_HEAD_ORDER,
    build_v6_control_matrix_config,
    build_v6_diagnostic_controls,
    build_v6_full_agent_config,
    build_v6_primary_controls,
    canonical_v6_control_matrix_json,
)

pytestmark = pytest.mark.unit


def _agent(control: object) -> IntegratedHiddenPartnerConfig:
    config = getattr(control, "agent_config")
    assert isinstance(config, IntegratedHiddenPartnerConfig)
    return config


def _leaf_differences(
    left: object,
    right: object,
    path: tuple[str, ...] = (),
) -> set[tuple[str, ...]]:
    if isinstance(left, dict) and isinstance(right, dict):
        assert set(left) == set(right)
        differences: set[tuple[str, ...]] = set()
        for key in left:
            differences.update(_leaf_differences(left[key], right[key], (*path, key)))
        return differences
    return set() if left == right else {path}


def test_v6_full_candidate_binds_world_state_heads_without_reward_duplication() -> None:
    config = build_v6_full_agent_config()
    grounded = config.grounded_world_model
    mixer = config.representation_gradient_mixer

    assert config.initial_active_descriptors == V6_INITIAL_ACTIVE_DESCRIPTORS
    assert config.grounded_world_learning_enabled
    assert config.grounded_world_planning_enabled
    assert config.epsilon == 0.20
    assert grounded is not None
    assert grounded.representation_loss_weights == V6_REPRESENTATION_LOSS_WEIGHTS
    assert grounded.feature_path_mode == "affine"
    assert V6_TARGET_HEAD_ORDER == (
        "x",
        "previous_contextual_outcome",
        "previous_partner_action",
        "has_partner_history",
        "u",
        "v",
        "world_cue_1",
        "world_cue_2",
        "reward",
        "discount",
    )
    positive_heads = {
        head
        for head, weight in zip(
            V6_TARGET_HEAD_ORDER,
            V6_REPRESENTATION_LOSS_WEIGHTS,
            strict=True,
        )
        if weight > 0.0
    }
    assert positive_heads == {
        "previous_contextual_outcome",
        "world_cue_1",
        "world_cue_2",
    }
    assert math.fsum(V6_REPRESENTATION_LOSS_WEIGHTS) == 10.0
    assert mixer is not None
    assert mixer.mode == "full"
    assert mixer.behavior_normalization == "unit_l2"
    assert mixer.grounded_world_normalization == "unit_l2"


def test_primary_control_order_and_matched_freeze_are_exact() -> None:
    controls = build_v6_primary_controls()
    assert tuple(control.name for control in controls) == PRIMARY_CONDITION_ORDER
    assert all(control.primary for control in controls)

    by_name = {control.name: control for control in controls}
    assert by_name["full"].execution_ready
    for name, expected in (
        ("world_credit_off", "behavior_only"),
        ("behavior_credit_off", "world_only"),
        ("all_representation_credit_off", "discard"),
    ):
        mixer = _agent(by_name[name]).representation_gradient_mixer
        assert mixer is not None
        assert mixer.mode == expected
    assert not _agent(by_name["grounded_model_frozen"]).grounded_world_learning_enabled
    assert not _agent(by_name["table_planner"]).grounded_world_planning_enabled
    assert not _agent(by_name["no_planning"]).planning_enabled
    assert _agent(by_name["uniform_partner"]).uniform_partner_belief
    assert not _agent(by_name["no_identity_carry"]).carry_survivors
    assert _agent(by_name["no_retention_floor"]).active_utility_retention_decay is None
    assert not _agent(by_name["retirement_disabled"]).retire_stale_features
    assert (
        _agent(by_name["retirement_disabled"]).candidate_reacquisition_confirmation_steps
        == 8
    )
    assert _agent(by_name["random_curation"]).random_feature_curation

    frozen = by_name["lifecycle_frozen"]
    assert frozen.execution_ready
    assert frozen.execution_blocker is None
    frozen_agent = _agent(frozen)
    assert not frozen_agent.feature_lifecycle_enabled
    assert frozen_agent.evidence_gated_feature_memory
    assert frozen_agent.evidence_gated_consumer_memory


def test_primary_arms_preserve_world_and_fixed_shape_mechanism_contracts() -> None:
    ready = [control for control in build_v6_primary_controls() if control.execution_ready]
    full = ready[0]
    full_grounded = _agent(full).grounded_world_model
    assert full_grounded is not None
    for control in ready:
        agent = _agent(control)
        assert control.world_config == full.world_config
        assert agent.initial_active_descriptors == V6_INITIAL_ACTIVE_DESCRIPTORS
        grounded = agent.grounded_world_model
        assert grounded is not None
        assert (
            grounded.representation_dim == full_grounded.representation_dim
        )
        assert grounded.target_dim == full_grounded.target_dim


def test_each_ready_primary_arm_changes_only_its_declared_static_intervention() -> None:
    controls = build_v6_primary_controls()
    by_name = {control.name: control for control in controls}
    full_payload = _agent(by_name["full"]).to_config()
    expected = {
        "full": set(),
        "grounded_model_frozen": {("grounded_world_learning_enabled",)},
        "world_credit_off": {("representation_gradient_mixer", "mode")},
        "behavior_credit_off": {("representation_gradient_mixer", "mode")},
        "all_representation_credit_off": {
            ("representation_gradient_mixer", "mode")
        },
        "state_frozen": {("state_learning_enabled",)},
        "recurrent_memory_masked": {("memory_masked",)},
        "table_planner": {("grounded_world_planning_enabled",)},
        "no_planning": {("planning_enabled",)},
        "uniform_partner": {("uniform_partner_belief",)},
        "lifecycle_frozen": {("feature_lifecycle_enabled",)},
        "no_identity_carry": {("carry_survivors",)},
        "no_retention_floor": {("active_utility_retention_decay",)},
        "retirement_disabled": {("retire_stale_features",)},
        "random_curation": {("random_feature_curation",)},
    }
    for name, paths in expected.items():
        assert _leaf_differences(full_payload, _agent(by_name[name]).to_config()) == paths


def test_diagnostics_bind_equal_cues_and_row_bias_but_block_uniform_actions() -> None:
    controls = build_v6_diagnostic_controls()
    assert tuple(control.name for control in controls) == V6_DIAGNOSTIC_ORDER
    by_name = {control.name: control for control in controls}

    uniform = by_name["uniform_action"]
    assert uniform.focal_action_policy == "balanced_external"
    assert not uniform.execution_ready
    assert uniform.execution_blocker is not None

    equal = by_name["equal_cue"]
    assert equal.execution_ready
    assert equal.world_config.cue_flip_probabilities == (0.30, 0.30)

    row_bias = by_name["row_bias"]
    assert row_bias.execution_ready
    grounded = _agent(row_bias).grounded_world_model
    assert grounded is not None
    assert grounded.feature_path_mode == "row_bias_only"


def test_control_matrix_is_json_only_nonauthorizing_and_has_no_seeds_or_outcomes() -> None:
    payload = build_v6_control_matrix_config()
    assert payload["schema"] == HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_CONTROL_SCHEMA
    assert payload["development_only"] is True
    assert payload["scientific_promotion_allowed"] is False
    assert payload["seed_namespace"] is None
    assert payload["thresholds"] is None
    assert payload["outcomes"] is None
    assert payload["status"] == "DEVELOPMENT_CONTROLS_PARTIALLY_BLOCKED"
    assert payload["primary_ready_count"] == 15
    assert payload["primary_required_count"] == 15
    assert payload["diagnostic_ready_count"] == 2
    assert payload["diagnostic_required_count"] == 3
    assert payload["all_controls_execution_ready"] is False
    assert len(payload["primary_controls"]) == 15
    assert len(payload["diagnostic_controls"]) == 3

    canonical = canonical_v6_control_matrix_json()
    assert json.loads(canonical) == payload
    assert "NaN" not in canonical
    assert "Infinity" not in canonical
