"""Static contracts for the nonexecuting noisy-world v6 control matrix."""

from __future__ import annotations

import dataclasses
import json
import math

import pytest

import alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_controls as controls_module
from alberta_framework.core.integrated_hidden_partner import IntegratedHiddenPartnerConfig
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6 import (
    OPEN_PREREQUISITES,
    PRIMARY_CONDITION_ORDER,
    V6_INITIAL_ACTIVE_DESCRIPTORS,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_controls import (
    HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_CONTROL_BINDING_SCHEMA,
    HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_CONTROL_SCHEMA,
    HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_DESIGN_SUCCESSOR_SCHEMA,
    V6_DESIGN_SUCCESSOR_CLOSED_BINDINGS,
    V6_DESIGN_SUCCESSOR_REMAINING_OPEN,
    V6_DIAGNOSTIC_ORDER,
    V6_MANIFEST_OVERRIDE_PATH_MAP,
    V6_REPRESENTATION_LOSS_WEIGHTS,
    V6_TARGET_HEAD_ORDER,
    HiddenPartnerLifecycleWorldV6Control,
    build_v6_control_bindings,
    build_v6_control_matrix_config,
    build_v6_design_successor_mapping_config,
    build_v6_diagnostic_controls,
    build_v6_full_agent_config,
    build_v6_primary_controls,
    canonical_v6_control_matrix_json,
    canonical_v6_control_matrix_sha256,
)
from alberta_framework.evaluation.hidden_partner_world_online_bridge import (
    HIDDEN_PARTNER_WORLD_ONLINE_BRIDGE_CONFIG_SCHEMA,
)
from alberta_framework.streams.hidden_partner_world_feedback import (
    HiddenPartnerWorldFeedbackConfig,
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
    assert config.action_selection_mode == "agent"
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


def test_diagnostics_bind_uniform_actions_equal_cues_and_row_bias_exactly() -> None:
    controls = build_v6_diagnostic_controls()
    assert tuple(control.name for control in controls) == V6_DIAGNOSTIC_ORDER
    by_name = {control.name: control for control in controls}
    full = build_v6_full_agent_config()

    uniform = by_name["uniform_action"]
    assert uniform.focal_action_policy == "balanced_external"
    assert uniform.initial_external_action == 0
    assert uniform.execution_ready
    assert uniform.execution_blocker is None
    assert _agent(uniform).action_selection_mode == "externally_forced"
    assert _leaf_differences(full.to_config(), _agent(uniform).to_config()) == {
        ("action_selection_mode",)
    }
    assert uniform.world_config == build_v6_primary_controls()[0].world_config

    equal = by_name["equal_cue"]
    assert equal.execution_ready
    assert equal.focal_action_policy == "agent"
    assert equal.initial_external_action == 0
    assert _agent(equal) == full
    assert equal.world_config.cue_flip_probabilities == (0.30, 0.30)

    row_bias = by_name["row_bias"]
    assert row_bias.execution_ready
    assert row_bias.focal_action_policy == "agent"
    assert row_bias.initial_external_action == 0
    grounded = _agent(row_bias).grounded_world_model
    assert grounded is not None
    assert grounded.feature_path_mode == "row_bias_only"
    assert _leaf_differences(full.to_config(), _agent(row_bias).to_config()) == {
        ("grounded_world_model", "feature_path_mode")
    }


def test_control_rejects_action_policy_and_agent_mode_mismatches() -> None:
    full = build_v6_full_agent_config()
    world = build_v6_primary_controls()[0].world_config
    forced = dataclasses.replace(full, action_selection_mode="externally_forced")

    with pytest.raises(ValueError, match="action_selection_mode must match exactly"):
        HiddenPartnerLifecycleWorldV6Control(
            name="uniform_action",
            primary=False,
            agent_config=full,
            world_config=world,
            focal_action_policy="balanced_external",
        )
    with pytest.raises(ValueError, match="action_selection_mode must match exactly"):
        HiddenPartnerLifecycleWorldV6Control(
            name="equal_cue",
            primary=False,
            agent_config=forced,
            world_config=world,
            focal_action_policy="agent",
        )


def test_control_matrix_is_json_only_nonauthorizing_and_has_no_seeds_or_outcomes() -> None:
    payload = build_v6_control_matrix_config()
    assert payload["schema"] == HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_CONTROL_SCHEMA
    assert (
        HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_CONTROL_SCHEMA
        == "alberta.hidden-partner-lifecycle-world.controls-development.v4"
    )
    assert payload["development_only"] is True
    assert payload["execution_authorized"] is False
    assert payload["evidence_authorized"] is False
    assert payload["scientific_promotion_allowed"] is False
    assert payload["seed_namespace"] is None
    assert payload["thresholds"] is None
    assert payload["outcomes"] is None
    assert payload["status"] == "DEVELOPMENT_CONTROLS_READY_FOR_RUNNER_BINDING"
    assert payload["primary_ready_count"] == 15
    assert payload["primary_required_count"] == 15
    assert payload["diagnostic_ready_count"] == 3
    assert payload["diagnostic_required_count"] == 3
    assert payload["all_controls_execution_ready"] is True
    primary_payloads = payload["primary_controls"]
    diagnostic_payloads = payload["diagnostic_controls"]
    assert isinstance(primary_payloads, list)
    assert isinstance(diagnostic_payloads, list)
    assert len(primary_payloads) == 15
    assert len(diagnostic_payloads) == 3
    for control in (*primary_payloads, *diagnostic_payloads):
        assert isinstance(control, dict)
        assert control["schema"] == HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_CONTROL_SCHEMA
        assert control["status"] == "READY_FOR_RUNNER_BINDING"
        assert control["execution_ready"] is True
        assert control["execution_blocker"] is None
        assert control["initial_external_action"] == 0
        assert control["execution_authorized"] is False
        assert control["evidence_authorized"] is False
        assert control["scientific_promotion_allowed"] is False

    canonical = canonical_v6_control_matrix_json()
    assert json.loads(canonical) == payload
    assert len(canonical_v6_control_matrix_sha256()) == 64
    assert "NaN" not in canonical
    assert "Infinity" not in canonical


def test_control_public_boundary_rejects_exact_type_laundering() -> None:
    uniform = build_v6_diagnostic_controls()[0]

    class StringSubclass(str):
        pass

    with pytest.raises(TypeError, match="focal_action_policy.*exact built-in str"):
        dataclasses.replace(
            uniform,
            focal_action_policy=StringSubclass("balanced_external"),
        )
    with pytest.raises(TypeError, match="control name.*exact built-in str"):
        dataclasses.replace(uniform, name=StringSubclass("uniform_action"))
    with pytest.raises(TypeError, match="primary.*exact built-in bool"):
        dataclasses.replace(uniform, primary=0)
    with pytest.raises(TypeError, match="initial_external_action.*exact built-in int"):
        dataclasses.replace(uniform, initial_external_action=False)
    with pytest.raises(ValueError, match="canonical initial_external_action 0"):
        dataclasses.replace(uniform, initial_external_action=1)


def test_control_public_boundary_rejects_config_and_control_subclasses() -> None:
    full = build_v6_full_agent_config()
    world = HiddenPartnerWorldFeedbackConfig()

    @dataclasses.dataclass(frozen=True)
    class AgentConfigSubclass(IntegratedHiddenPartnerConfig):
        pass

    agent_subclass = AgentConfigSubclass(
        **{
            field.name: getattr(full, field.name)
            for field in dataclasses.fields(IntegratedHiddenPartnerConfig)
        }
    )
    with pytest.raises(TypeError, match="exact IntegratedHiddenPartnerConfig"):
        HiddenPartnerLifecycleWorldV6Control(
            name="full",
            primary=True,
            agent_config=agent_subclass,
            world_config=world,
        )

    @dataclasses.dataclass(frozen=True)
    class WorldConfigSubclass(HiddenPartnerWorldFeedbackConfig):
        pass

    world_subclass = WorldConfigSubclass(
        **{
            field.name: getattr(world, field.name)
            for field in dataclasses.fields(HiddenPartnerWorldFeedbackConfig)
        }
    )
    with pytest.raises(TypeError, match="exact HiddenPartnerWorldFeedbackConfig"):
        HiddenPartnerLifecycleWorldV6Control(
            name="full",
            primary=True,
            agent_config=full,
            world_config=world_subclass,
        )

    @dataclasses.dataclass(frozen=True)
    class ControlSubclass(HiddenPartnerLifecycleWorldV6Control):
        pass

    with pytest.raises(TypeError, match="exact HiddenPartnerLifecycleWorldV6Control"):
        ControlSubclass(
            name="full",
            primary=True,
            agent_config=full,
            world_config=world,
        )


def test_named_controls_reject_semantically_compatible_but_wrong_compositions() -> None:
    full = build_v6_full_agent_config()
    diagnostics = {control.name: control for control in build_v6_diagnostic_controls()}

    with pytest.raises(ValueError, match="row_bias.*noncanonical agent semantics"):
        dataclasses.replace(diagnostics["row_bias"], agent_config=full)
    with pytest.raises(ValueError, match="equal_cue.*noncanonical world semantics"):
        dataclasses.replace(
            diagnostics["equal_cue"],
            world_config=HiddenPartnerWorldFeedbackConfig(),
        )
    with pytest.raises(ValueError, match="full.*noncanonical world semantics"):
        dataclasses.replace(
            build_v6_primary_controls()[0],
            world_config=HiddenPartnerWorldFeedbackConfig(
                cue_flip_probabilities=(0.30, 0.30)
            ),
        )


def test_primary_source_drift_fails_against_the_frozen_manifest_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = controls_module._primary_agent_config

    def drifted(
        name: str,
        full: IntegratedHiddenPartnerConfig,
    ) -> IntegratedHiddenPartnerConfig | None:
        if name == "state_frozen":
            return full
        return original(name, full)

    monkeypatch.setattr(controls_module, "_primary_agent_config", drifted)
    with pytest.raises(RuntimeError, match="frozen override mapping"):
        build_v6_primary_controls()

    assert V6_MANIFEST_OVERRIDE_PATH_MAP == (
        (
            "representation_gradient_mode",
            ("representation_gradient_mixer", "mode"),
        ),
    )


def test_every_control_has_a_versioned_constructible_bridge_binding() -> None:
    bindings = build_v6_control_bindings()
    expected_order = (
        *(("primary", name) for name in PRIMARY_CONDITION_ORDER),
        *(("diagnostic", name) for name in V6_DIAGNOSTIC_ORDER),
    )

    assert tuple((binding.family, binding.name) for binding in bindings) == expected_order
    assert len(bindings) == 18
    for binding in bindings:
        payload = binding.to_config()
        assert payload["schema"] == HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_CONTROL_BINDING_SCHEMA
        assert payload["control_schema"] == HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_CONTROL_SCHEMA
        assert payload["bridge_schema"] == HIDDEN_PARTNER_WORLD_ONLINE_BRIDGE_CONFIG_SCHEMA
        assert payload["initial_external_action"] == 0
        assert len(str(payload["control_config_sha256"])) == 64
        assert len(str(payload["bridge_config_sha256"])) == 64
        assert payload["execution_authorized"] is False
        assert payload["evidence_authorized"] is False
        assert payload["scientific_promotion_allowed"] is False

    first = bindings[0]
    with pytest.raises(ValueError, match="binding schema"):
        dataclasses.replace(first, binding_schema=first.binding_schema + ".tampered")
    with pytest.raises(ValueError, match="bridge schema"):
        dataclasses.replace(first, bridge_schema=first.bridge_schema + ".tampered")
    with pytest.raises(ValueError, match="canonical zero"):
        dataclasses.replace(first, initial_external_action=1)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        dataclasses.replace(first, bridge_config_sha256="0" * 63)


def test_design_successor_mapping_closes_only_historical_mechanism_entries() -> None:
    payload = build_v6_design_successor_mapping_config()
    assert payload["schema"] == HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_DESIGN_SUCCESSOR_SCHEMA
    assert payload["frozen_open_prerequisites_are_historical"] is True
    assert tuple(identifier for identifier, _ in OPEN_PREREQUISITES[:3]) == tuple(
        identifier for identifier, _ in V6_DESIGN_SUCCESSOR_CLOSED_BINDINGS
    )
    assert OPEN_PREREQUISITES[3:] == V6_DESIGN_SUCCESSOR_REMAINING_OPEN
    assert payload["execution_authorized"] is False
    assert payload["evidence_authorized"] is False
    assert payload["scientific_promotion_allowed"] is False
