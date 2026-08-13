"""Contracts for the HCCL one-factor arm panel."""

from __future__ import annotations

import dataclasses
import json

import pytest

from alberta_framework.core.hccl_causal_core_arms import (
    HCCL_CAUSAL_CORE_ARM_CONFIG_SCHEMA,
    HCCL_CAUSAL_CORE_ARM_PANEL,
    HCCL_CAUSAL_CORE_ARM_POLICY_SCHEMA,
    HCCL_CAUSAL_CORE_ARM_SCIENTIFIC_PROMOTION_ALLOWED,
    HCCL_CAUSAL_CORE_ARM_STATUS,
    HCCLCausalCoreArmConfig,
    HCCLCausalCoreArmName,
    HCCLCausalCoreArmRoutePolicy,
    hccl_causal_core_arm_intervention_fields,
    hccl_causal_core_arm_policy,
)

pytestmark = pytest.mark.unit


def test_panel_has_exact_names_order_and_one_factor_differences() -> None:
    assert tuple(arm.value for arm in HCCL_CAUSAL_CORE_ARM_PANEL) == (
        "full",
        "fast_state_unrouted",
        "slow_context_unrouted",
        "lineage_rescue_unrouted",
        "feature_random_rank",
        "feature_consumers_unrouted",
        "memory_dispatch_unrouted",
        "uniform_partner_belief",
        "planner_dispatch_unrouted",
    )
    assert hccl_causal_core_arm_intervention_fields(
        HCCLCausalCoreArmName.FULL
    ) == ()
    expected = {
        HCCLCausalCoreArmName.FAST_STATE_UNROUTED: "fast_state_routed",
        HCCLCausalCoreArmName.SLOW_CONTEXT_UNROUTED: "slow_context_routed",
        HCCLCausalCoreArmName.LINEAGE_RESCUE_UNROUTED: "lineage_rescue_routed",
        HCCLCausalCoreArmName.FEATURE_RANDOM_RANK: "feature_rank_selection",
        HCCLCausalCoreArmName.FEATURE_CONSUMERS_UNROUTED: (
            "feature_consumers_routed"
        ),
        HCCLCausalCoreArmName.MEMORY_DISPATCH_UNROUTED: (
            "memory_dispatch_selection"
        ),
        HCCLCausalCoreArmName.UNIFORM_PARTNER_BELIEF: (
            "partner_belief_selection"
        ),
        HCCLCausalCoreArmName.PLANNER_DISPATCH_UNROUTED: (
            "planner_dispatch_selection"
        ),
    }
    for arm, field in expected.items():
        assert hccl_causal_core_arm_intervention_fields(arm) == (field,)


def test_every_arm_declares_unbound_execution_requirements_not_runtime_facts() -> None:
    assert HCCL_CAUSAL_CORE_ARM_CONFIG_SCHEMA == "alberta.hccl-causal-core.arm-config.v2"
    assert HCCL_CAUSAL_CORE_ARM_POLICY_SCHEMA == (
        "alberta.hccl-causal-core.arm-route-policy.v2"
    )
    assert (
        HCCL_CAUSAL_CORE_ARM_STATUS
        == "l0-development-unbound-selection-requirement-not-executable"
    )

    full = hccl_causal_core_arm_policy(HCCLCausalCoreArmName.FULL)
    for arm in HCCL_CAUSAL_CORE_ARM_PANEL:
        policy = hccl_causal_core_arm_policy(arm)
        payload = policy.to_config()
        assert payload["schema"] == HCCL_CAUSAL_CORE_ARM_POLICY_SCHEMA
        assert payload["all_mechanisms_instantiated"] is False
        assert payload["all_routed_and_unrouted_alternatives_computed"] is False
        assert payload["all_mechanisms_instantiated_required_if_executed"] is True
        assert (
            payload["all_routed_and_unrouted_alternatives_computed_required_if_executed"]
            is True
        )
        assert payload["persistent_shapes_matched"] is False
        assert payload["persistent_shapes_matched_required_if_executed"] is True
        assert payload["paired_exogenous_key_roles"] is False
        assert payload["paired_exogenous_key_roles_required_if_executed"] is True
        assert payload["environment_proposal_calls_per_event"] == 0
        assert payload["environment_proposal_calls_per_event_required_if_executed"] == 8
        assert payload["scheduled_candidate_opportunities_preserved"] is False
        assert payload["scheduled_candidate_opportunities_preserved_required_if_executed"] is True
        assert payload["scheduled_curation_opportunities_preserved"] is False
        assert payload["scheduled_curation_opportunities_preserved_required_if_executed"] is True
        assert payload["mechanism_rng_static_rules_preserved"] is False
        assert payload["mechanism_rng_static_rules_preserved_required_if_executed"] is True
        assert payload["runtime_owner_bound"] is False
        assert payload["runtime_alternatives_validated"] is False
        assert payload["execution_implementation_available"] is False
        assert payload["matched_total_work_claimed"] is False
        assert payload["equal_flops_claimed"] is False
        assert payload["equal_wall_time_claimed"] is False
        assert payload["scientific_promotion_allowed"] is False
        differences = [
            field.name
            for field in dataclasses.fields(policy)
            if getattr(policy, field.name) != getattr(full, field.name)
        ]
        assert differences == list(hccl_causal_core_arm_intervention_fields(arm))


def test_each_intervention_selects_the_declared_neutral_alternative() -> None:
    assert not hccl_causal_core_arm_policy(
        HCCLCausalCoreArmName.FAST_STATE_UNROUTED
    ).fast_state_routed
    assert not hccl_causal_core_arm_policy(
        HCCLCausalCoreArmName.SLOW_CONTEXT_UNROUTED
    ).slow_context_routed
    assert not hccl_causal_core_arm_policy(
        HCCLCausalCoreArmName.LINEAGE_RESCUE_UNROUTED
    ).lineage_rescue_routed
    assert (
        hccl_causal_core_arm_policy(
            HCCLCausalCoreArmName.FEATURE_RANDOM_RANK
        ).feature_rank_selection
        == "random"
    )
    assert not hccl_causal_core_arm_policy(
        HCCLCausalCoreArmName.FEATURE_CONSUMERS_UNROUTED
    ).feature_consumers_routed
    assert (
        hccl_causal_core_arm_policy(
            HCCLCausalCoreArmName.MEMORY_DISPATCH_UNROUTED
        ).memory_dispatch_selection
        == "base"
    )
    assert (
        hccl_causal_core_arm_policy(
            HCCLCausalCoreArmName.UNIFORM_PARTNER_BELIEF
        ).partner_belief_selection
        == "uniform"
    )
    assert (
        hccl_causal_core_arm_policy(
            HCCLCausalCoreArmName.PLANNER_DISPATCH_UNROUTED
        ).planner_dispatch_selection
        == "memory"
    )


@pytest.mark.parametrize("arm", HCCL_CAUSAL_CORE_ARM_PANEL)
def test_config_round_trip_is_strict_and_grants_no_authority(
    arm: HCCLCausalCoreArmName,
) -> None:
    config = HCCLCausalCoreArmConfig(arm=arm)
    payload = json.loads(json.dumps(config.to_config()))
    restored = HCCLCausalCoreArmConfig.from_config(payload)

    assert restored == config
    assert payload["schema"] == HCCL_CAUSAL_CORE_ARM_CONFIG_SCHEMA
    assert payload["selection_contract_only"] is True
    assert payload["execution_authorized"] is False
    assert payload["all_mechanisms_instantiated"] is False
    assert payload["all_routed_and_unrouted_alternatives_computed"] is False
    assert payload["all_mechanisms_instantiated_required_if_executed"] is True
    assert payload["all_routed_and_unrouted_alternatives_computed_required_if_executed"] is True
    assert payload["runtime_owner_bound"] is False
    assert payload["runtime_alternatives_validated"] is False
    assert payload["execution_implementation_available"] is False
    assert payload["seed_reservation_or_consumption_authorized"] is False
    assert payload["artifact_writes_authorized"] is False
    assert payload["thresholds_defined"] is False
    assert payload["evidence_claimed"] is False
    assert payload["scientific_promotion_allowed"] is False
    assert HCCL_CAUSAL_CORE_ARM_SCIENTIFIC_PROMOTION_ALLOWED is False

    with pytest.raises(ValueError, match="noncanonical or unsupported"):
        HCCLCausalCoreArmConfig.from_config({**payload, "extra": True})
    with pytest.raises(ValueError, match="noncanonical or unsupported"):
        HCCLCausalCoreArmConfig.from_config(
            {**payload, "intervention_fields": ["forged"]}
        )


def test_legacy_or_runtime_fact_conflation_is_noncanonical() -> None:
    payload = HCCLCausalCoreArmConfig().to_config()

    legacy = json.loads(json.dumps(payload))
    legacy["schema"] = "alberta.hccl-causal-core.arm-config.v1"
    legacy_policy = legacy["policy"]
    assert isinstance(legacy_policy, dict)
    legacy_policy["schema"] = "alberta.hccl-causal-core.arm-route-policy.v1"
    with pytest.raises(ValueError, match="noncanonical or unsupported"):
        HCCLCausalCoreArmConfig.from_config(legacy)

    for field in (
        "all_mechanisms_instantiated",
        "all_routed_and_unrouted_alternatives_computed",
        "runtime_owner_bound",
        "runtime_alternatives_validated",
        "execution_implementation_available",
    ):
        conflated = json.loads(json.dumps(payload))
        conflated[field] = True
        with pytest.raises(ValueError, match="noncanonical or unsupported"):
            HCCLCausalCoreArmConfig.from_config(conflated)

        nested_conflated = json.loads(json.dumps(payload))
        nested_policy = nested_conflated["policy"]
        assert isinstance(nested_policy, dict)
        nested_policy[field] = True
        with pytest.raises(ValueError, match="noncanonical or unsupported"):
            HCCLCausalCoreArmConfig.from_config(nested_conflated)


def test_wrong_runtime_types_and_invalid_policy_values_are_rejected() -> None:
    with pytest.raises(TypeError, match="exact HCCLCausalCoreArmName"):
        HCCLCausalCoreArmConfig(arm="full")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact HCCLCausalCoreArmName"):
        hccl_causal_core_arm_policy("full")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="fast_state_routed"):
        HCCLCausalCoreArmRoutePolicy(fast_state_routed=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="feature_rank_selection"):
        HCCLCausalCoreArmRoutePolicy(
            feature_rank_selection="oracle"  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="fixed causal-core arms"):
        HCCLCausalCoreArmConfig.from_config({"arm": "unknown"})


def test_config_and_policy_are_frozen() -> None:
    config = HCCLCausalCoreArmConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.arm = HCCLCausalCoreArmName.FEATURE_RANDOM_RANK  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.policy.fast_state_routed = False  # type: ignore[misc]
