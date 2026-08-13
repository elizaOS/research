# mypy: disable-error-code="arg-type,untyped-decorator"
"""Cheap contracts for the canonical routed-R35 continual-dyad factory."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest

from alberta_framework.core.hccl_continual_dyad_factory import (
    HCCLContinualDyadFactoryConfig,
    build_hccl_continual_dyad_config,
)
from alberta_framework.core.hccl_routed_continual_dyad import (
    HCCL_ROUTED_CONTINUAL_DYAD_CONFIG_SCHEMA,
    HCCLRoutedContinualDyadConfig,
)
from alberta_framework.core.hccl_routed_continual_dyad_factory import (
    HCCL_ROUTED_CONTINUAL_DYAD_BINDING_OWNER_DIGEST,
    HCCL_ROUTED_CONTINUAL_DYAD_BMP_AGENT_0_OWNER_DIGEST,
    HCCL_ROUTED_CONTINUAL_DYAD_BMP_AGENT_1_OWNER_DIGEST,
    HCCL_ROUTED_CONTINUAL_DYAD_FACTORY_CONFIG_SCHEMA,
    HCCL_ROUTED_CONTINUAL_DYAD_FACTORY_EVIDENCE_LEVEL,
    HCCL_ROUTED_CONTINUAL_DYAD_FACTORY_STATUS,
    HCCLRoutedContinualDyadFactory,
    HCCLRoutedContinualDyadFactoryConfig,
    build_hccl_routed_continual_dyad_config,
)
from alberta_framework.streams.hccl_causal_core import (
    HCCL_CAUSAL_CORE_CANONICAL_PROFILE,
    HCCL_CAUSAL_CORE_L2_PROFILE,
    HCCL_CAUSAL_CORE_L3_PROFILE,
    HCCL_CAUSAL_CORE_SMOKE_PROFILE,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("factory_config", "profile", "world_events", "agent_events", "is_smoke"),
    (
        (
            HCCLRoutedContinualDyadFactoryConfig(),
            HCCL_CAUSAL_CORE_CANONICAL_PROFILE,
            8_998,
            8_998,
            False,
        ),
        (
            HCCLRoutedContinualDyadFactoryConfig.mechanics_smoke(),
            HCCL_CAUSAL_CORE_SMOKE_PROFILE,
            420,
            8_998,
            True,
        ),
        (
            HCCLRoutedContinualDyadFactoryConfig.core_l2(),
            HCCL_CAUSAL_CORE_L2_PROFILE,
            71_984,
            71_984,
            False,
        ),
        (
            HCCLRoutedContinualDyadFactoryConfig.core_l3(),
            HCCL_CAUSAL_CORE_L3_PROFILE,
            1_007_776,
            1_007_776,
            False,
        ),
    ),
)
def test_profiles_derive_exact_world_and_agent_horizons(
    factory_config: HCCLRoutedContinualDyadFactoryConfig,
    profile: str,
    world_events: int,
    agent_events: int,
    is_smoke: bool,
) -> None:
    assert factory_config.schedule_profile == profile
    assert factory_config.maximum_committed_transitions == world_events
    assert factory_config.agent_lifetime_events == agent_events
    assert factory_config.mechanics_smoke_enabled is is_smoke


@pytest.mark.unit
@pytest.mark.parametrize(
    "factory_config",
    (
        HCCLRoutedContinualDyadFactoryConfig(),
        HCCLRoutedContinualDyadFactoryConfig.mechanics_smoke(),
        HCCLRoutedContinualDyadFactoryConfig.core_l2(),
        HCCLRoutedContinualDyadFactoryConfig.core_l3(),
    ),
)
def test_factory_manifest_round_trip_is_strict_and_grants_no_authority(
    factory_config: HCCLRoutedContinualDyadFactoryConfig,
) -> None:
    payload = factory_config.to_config()
    encoded = factory_config.to_json()
    assert encoded == json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert HCCLRoutedContinualDyadFactoryConfig.from_config(payload) == factory_config
    assert HCCLRoutedContinualDyadFactoryConfig.from_json(encoded) == factory_config
    assert payload["schema"] == HCCL_ROUTED_CONTINUAL_DYAD_FACTORY_CONFIG_SCHEMA
    assert payload["transaction_config_schema"] == HCCL_ROUTED_CONTINUAL_DYAD_CONFIG_SCHEMA
    assert payload["mechanism_status"] == HCCL_ROUTED_CONTINUAL_DYAD_FACTORY_STATUS
    assert payload["evidence_level"] == HCCL_ROUTED_CONTINUAL_DYAD_FACTORY_EVIDENCE_LEVEL
    assert payload["deterministic_build"] is True
    assert payload["deterministic_initialization"] is True
    assert payload["configuration_and_initialization_only"] is True
    for name in (
        "execution_authorized",
        "schedule_execution_authorized",
        "transaction_execution_authorized",
        "benchmark_execution_authorized",
        "dispatch_authorized",
        "seed_reservation_or_consumption_authorized",
        "artifact_authorized",
        "output_writes_authorized",
        "evidence_authorized",
        "promotion_authorized",
        "scientific_promotion_allowed",
    ):
        assert payload[name] is False
    assert payload["artifact_write_calls"] == 0
    assert payload["artifact_bytes_written"] == 0

    changed = dict(payload)
    changed["evidence_authorized"] = True
    with pytest.raises(ValueError, match="noncanonical"):
        HCCLRoutedContinualDyadFactoryConfig.from_config(changed)
    with pytest.raises(ValueError, match="non-strict"):
        HCCLRoutedContinualDyadFactoryConfig.from_json(
            '{"schedule_profile":"a","schedule_profile":"b"}'
        )
    with pytest.raises(ValueError, match="non-strict"):
        HCCLRoutedContinualDyadFactoryConfig.from_json('{"value":NaN}')
    with pytest.raises(ValueError, match="encode one object"):
        HCCLRoutedContinualDyadFactoryConfig.from_json("[]")
    with pytest.raises(TypeError, match="exact string"):
        HCCLRoutedContinualDyadFactoryConfig.from_json(b"{}")


@pytest.mark.unit
@pytest.mark.parametrize(
    "factory_config",
    (
        HCCLRoutedContinualDyadFactoryConfig(),
        HCCLRoutedContinualDyadFactoryConfig.mechanics_smoke(),
        HCCLRoutedContinualDyadFactoryConfig.core_l2(),
        HCCLRoutedContinualDyadFactoryConfig.core_l3(),
    ),
)
def test_build_config_reuses_primitive_core_and_adds_exact_r35_owners(
    factory_config: HCCLRoutedContinualDyadFactoryConfig,
) -> None:
    routed = build_hccl_routed_continual_dyad_config(factory_config)
    primitive = build_hccl_continual_dyad_config(
        HCCLContinualDyadFactoryConfig(
            schedule_profile=factory_config.schedule_profile,
        )
    )
    assert type(routed) is HCCLRoutedContinualDyadConfig
    assert routed.hccl == primitive.hccl
    assert routed.coordinator.to_config() == primitive.agent_0.coordinator.to_config()
    assert routed.context.to_config() == primitive.context.to_config()
    assert primitive.agent_0.coordinator == primitive.agent_1.coordinator

    assert routed.hccl.world_config.maximum_committed_transitions == (
        factory_config.maximum_committed_transitions
    )
    assert routed.coordinator.max_events == factory_config.agent_lifetime_events
    for index, memory_config in enumerate(
        (routed.memory_agent_0, routed.memory_agent_1)
    ):
        memory = memory_config.controller.memory
        assert memory_config.agent_index == index
        assert memory.capacity == 64
        assert memory.observation_dim == 35
        assert memory.key_dim == 35
        assert memory.action_dim == 2
        assert memory.outcome_dim == 35
        assert memory.max_age == factory_config.agent_lifetime_events
        assert memory.staleness_scale == float(factory_config.agent_lifetime_events)

    assert routed.planner.to_config()["representation_dim"] == 35
    assert routed.planner.to_config()["n_actions"] == 2
    assert routed.bmp_agent_0.owner_digest == (
        HCCL_ROUTED_CONTINUAL_DYAD_BMP_AGENT_0_OWNER_DIGEST
    )
    assert routed.bmp_agent_1.owner_digest == (
        HCCL_ROUTED_CONTINUAL_DYAD_BMP_AGENT_1_OWNER_DIGEST
    )
    assert routed.binding_owner_digest == HCCL_ROUTED_CONTINUAL_DYAD_BINDING_OWNER_DIGEST
    assert len(
        {
            routed.bmp_agent_0.owner_digest,
            routed.bmp_agent_1.owner_digest,
            routed.binding_owner_digest,
        }
    ) == 3
    assert routed.to_config()["caller_supplied_pair_admission"] is False
    assert routed.to_config()["dispatch_authority"] is False
    assert routed.to_config()["artifact_authority"] is False
    assert routed.to_config()["evidence_authority"] is False
    assert routed.to_config()["promotion_authority"] is False


@pytest.mark.unit
def test_default_build_is_canonical_and_exact_types_are_required() -> None:
    expected = build_hccl_routed_continual_dyad_config(
        HCCLRoutedContinualDyadFactoryConfig()
    )
    assert build_hccl_routed_continual_dyad_config().to_config() == expected.to_config()
    factory = HCCLRoutedContinualDyadFactory()
    assert factory.config == HCCLRoutedContinualDyadFactoryConfig()
    assert factory.transaction_config.to_config() == expected.to_config()
    assert factory.dyad_config is factory.transaction_config
    assert factory.to_config() == factory.config.to_config()
    with pytest.raises(TypeError, match="factory_config"):
        build_hccl_routed_continual_dyad_config(
            HCCLContinualDyadFactoryConfig()
        )
    with pytest.raises(TypeError, match="config"):
        HCCLRoutedContinualDyadFactory(HCCLContinualDyadFactoryConfig())
    with pytest.raises(ValueError, match="fixed versioned HCCL profile"):
        HCCLRoutedContinualDyadFactoryConfig(schedule_profile="unsupported")


@pytest.mark.unit
def test_build_and_init_surfaces_delegate_without_running_a_real_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import alberta_framework.core.hccl_routed_continual_dyad_factory as factory_module

    calls: list[tuple[object, object]] = []
    state = object()

    class FakeDyad:
        def __init__(self, config: HCCLRoutedContinualDyadConfig) -> None:
            self.config = config

        def init(
            self,
            key: object,
            *,
            initial_hard_action_masks: object = None,
        ) -> object:
            calls.append((key, initial_hard_action_masks))
            return state

    monkeypatch.setattr(factory_module, "HCCLRoutedContinualDyad", FakeDyad)
    factory = HCCLRoutedContinualDyadFactory.mechanics_smoke()
    built = factory.build()
    assert type(built) is FakeDyad
    assert built.config is factory.transaction_config
    key = object()
    masks = object()
    initialized = factory.init(key, initial_hard_action_masks=masks)
    assert initialized.factory_config is factory.config
    assert type(initialized.dyad) is FakeDyad
    assert initialized.dyad.config is factory.transaction_config
    assert initialized.state is state
    assert calls == [(key, masks)]
    assert dataclasses.is_dataclass(initialized)


def test_module_declares_only_additive_factory_authority() -> None:
    import alberta_framework.core.hccl_routed_continual_dyad_factory as factory_module

    exports: Any = factory_module.__all__
    assert "HCCLRoutedContinualDyadFactory" in exports
    assert "HCCLRoutedContinualDyadFactoryConfig" in exports
    assert "build_hccl_routed_continual_dyad_config" in exports
    assert not hasattr(HCCLRoutedContinualDyadFactory, "run")
    assert not hasattr(HCCLRoutedContinualDyadFactory, "step")
