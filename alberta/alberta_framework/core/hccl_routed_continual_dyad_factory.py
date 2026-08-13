"""Canonical construction for the routed-R35 HCCL continual dyad.

The factory projects the world, coordinator, and context configurations from
the public primitive-dyad configuration builder.  It then replaces the legacy
raw-observation memory and planner layer with two feature-bound R35 memories,
planner-v2, and three distinct persistent owner identities.  No private helper
from the primitive factory is imported.

This module owns configuration, construction, and deterministic initialization
only.  It does not execute an event or schedule, reserve a seed, write an
artifact, or grant dispatch, benchmark, evidence, threshold, or promotion
authority.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from typing import Final, cast

from jax import Array

from alberta_framework.core.experiential_memory import ExperientialMemoryConfig
from alberta_framework.core.hccl_authenticated_bmp_projection import (
    HCCLAuthenticatedBMPProjectionConfig,
)
from alberta_framework.core.hccl_continual_dyad_factory import (
    HCCLContinualDyadFactoryConfig,
    build_hccl_continual_dyad_config,
)
from alberta_framework.core.hccl_feature_bound_memory import (
    HCCLFeatureBoundMemoryConfig,
)
from alberta_framework.core.hccl_routed_continual_dyad import (
    HCCL_ROUTED_CONTINUAL_DYAD_CONFIG_SCHEMA,
    HCCLRoutedContinualDyad,
    HCCLRoutedContinualDyadConfig,
    HCCLRoutedContinualDyadState,
)
from alberta_framework.core.learned_experiential_memory_controller import (
    LearnedExperientialMemoryControllerConfig,
)
from alberta_framework.core.prototype_factorized_partner_planner_v2 import (
    PrototypeFactorizedPartnerPlannerV2Config,
)
from alberta_framework.streams.hccl_causal_core import (
    HCCL_CAUSAL_CORE_CANONICAL_PROFILE,
    HCCL_CAUSAL_CORE_L2_PROFILE,
    HCCL_CAUSAL_CORE_L3_PROFILE,
    HCCL_CAUSAL_CORE_SMOKE_PROFILE,
    hccl_causal_core_lifetime_for_profile,
)

HCCL_ROUTED_CONTINUAL_DYAD_FACTORY_CONFIG_SCHEMA: Final = (
    "alberta.hccl-routed-continual-dyad-factory.config.v1"
)
HCCL_ROUTED_CONTINUAL_DYAD_FACTORY_STATUS: Final = (
    "l0-development-r35-config-build-and-deterministic-initialization-only"
)
HCCL_ROUTED_CONTINUAL_DYAD_FACTORY_EVIDENCE_LEVEL: Final = "L0"

HCCL_ROUTED_CONTINUAL_DYAD_BMP_AGENT_0_OWNER_DIGEST: Final = (
    0xB101B101,
    0xB202B202,
    0xB303B303,
    0xB404B404,
    0xB505B505,
    0xB606B606,
    0xB707B707,
    0xB808B808,
)
HCCL_ROUTED_CONTINUAL_DYAD_BMP_AGENT_1_OWNER_DIGEST: Final = (
    0xE101E101,
    0xE202E202,
    0xE303E303,
    0xE404E404,
    0xE505E505,
    0xE606E606,
    0xE707E707,
    0xE808E808,
)
HCCL_ROUTED_CONTINUAL_DYAD_BINDING_OWNER_DIGEST: Final = (
    0xF101F101,
    0xF202F202,
    0xF303F303,
    0xF404F404,
    0xF505F505,
    0xF606F606,
    0xF707F707,
    0xF808F808,
)

_CANONICAL_AGENT_LIFETIME = 8_998
_MEMORY_CAPACITY = 64
_ROUTED_REPRESENTATION_DIM = 35
_N_ACTIONS = 2


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"nonfinite JSON constant {value!r} is forbidden")


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLRoutedContinualDyadFactoryConfig:
    """Select one fixed HCCL schedule for the exact routed-R35 dyad."""

    schedule_profile: str = HCCL_CAUSAL_CORE_CANONICAL_PROFILE

    def __post_init__(self) -> None:
        if type(self.schedule_profile) is not str or self.schedule_profile not in (
            HCCL_CAUSAL_CORE_CANONICAL_PROFILE,
            HCCL_CAUSAL_CORE_SMOKE_PROFILE,
            HCCL_CAUSAL_CORE_L2_PROFILE,
            HCCL_CAUSAL_CORE_L3_PROFILE,
        ):
            raise ValueError("schedule_profile must select one fixed versioned HCCL profile")

    @classmethod
    def mechanics_smoke(cls) -> HCCLRoutedContinualDyadFactoryConfig:
        """Select the opt-in authenticated 420-event mechanics schedule."""

        return cls(schedule_profile=HCCL_CAUSAL_CORE_SMOKE_PROFILE)

    @classmethod
    def core_l2(cls) -> HCCLRoutedContinualDyadFactoryConfig:
        """Select the uninterrupted eight-cycle 71,984-event Core-L2 life."""

        return cls(schedule_profile=HCCL_CAUSAL_CORE_L2_PROFILE)

    @classmethod
    def core_l3(cls) -> HCCLRoutedContinualDyadFactoryConfig:
        """Select the uninterrupted 112-cycle 1,007,776-event Core-L3 life."""

        return cls(schedule_profile=HCCL_CAUSAL_CORE_L3_PROFILE)

    @property
    def maximum_committed_transitions(self) -> int:
        """Return the exact world horizon selected by the versioned profile."""

        return cast(
            int,
            hccl_causal_core_lifetime_for_profile(self.schedule_profile),
        )

    @property
    def agent_lifetime_events(self) -> int:
        """Close every agent guard over a full life; smoke retains 8,998."""

        return max(_CANONICAL_AGENT_LIFETIME, self.maximum_committed_transitions)

    @property
    def mechanics_smoke_enabled(self) -> bool:
        return bool(self.schedule_profile == HCCL_CAUSAL_CORE_SMOKE_PROFILE)

    def to_config(self) -> dict[str, object]:
        """Return the strict JSON-compatible side-effect and authority contract."""

        return {
            "type": type(self).__name__,
            "schema": HCCL_ROUTED_CONTINUAL_DYAD_FACTORY_CONFIG_SCHEMA,
            "transaction_config_schema": HCCL_ROUTED_CONTINUAL_DYAD_CONFIG_SCHEMA,
            "mechanism_status": HCCL_ROUTED_CONTINUAL_DYAD_FACTORY_STATUS,
            "evidence_level": HCCL_ROUTED_CONTINUAL_DYAD_FACTORY_EVIDENCE_LEVEL,
            "schedule_profile": self.schedule_profile,
            "maximum_committed_transitions": self.maximum_committed_transitions,
            "mechanics_smoke_enabled": self.mechanics_smoke_enabled,
            "agent_lifetime_events": self.agent_lifetime_events,
            "memory_capacity": _MEMORY_CAPACITY,
            "memory_observation_dim": _ROUTED_REPRESENTATION_DIM,
            "memory_key_dim": _ROUTED_REPRESENTATION_DIM,
            "memory_action_dim": _N_ACTIONS,
            "memory_outcome_dim": _ROUTED_REPRESENTATION_DIM,
            "planner_version": "v2",
            "deterministic_build": True,
            "deterministic_initialization": True,
            "configuration_and_initialization_only": True,
            "execution_authorized": False,
            "schedule_execution_authorized": False,
            "transaction_execution_authorized": False,
            "benchmark_execution_authorized": False,
            "dispatch_authorized": False,
            "seed_reservation_or_consumption_authorized": False,
            "artifact_authorized": False,
            "output_writes_authorized": False,
            "evidence_authorized": False,
            "promotion_authorized": False,
            "scientific_promotion_allowed": False,
            "artifact_write_calls": 0,
            "artifact_bytes_written": 0,
        }

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, object],
    ) -> HCCLRoutedContinualDyadFactoryConfig:
        """Reconstruct only one exact current routed-factory manifest."""

        if type(payload) is not dict:
            raise TypeError("routed continual-dyad factory config must be an exact dict")
        schedule_profile = payload.get("schedule_profile")
        if type(schedule_profile) is not str:
            raise ValueError(
                "routed continual-dyad factory schedule_profile must be an exact string"
            )
        candidate = cls(schedule_profile=schedule_profile)
        if _canonical_json_bytes(payload) != _canonical_json_bytes(candidate.to_config()):
            raise ValueError("routed continual-dyad factory config is noncanonical or unsupported")
        return candidate

    def to_json(self) -> str:
        """Serialize the complete factory declaration as canonical strict JSON."""

        return _canonical_json_bytes(self.to_config()).decode("utf-8")

    @classmethod
    def from_json(cls, payload: str) -> HCCLRoutedContinualDyadFactoryConfig:
        """Parse a declaration while rejecting duplicate and nonfinite values."""

        if type(payload) is not str:
            raise TypeError("routed continual-dyad factory JSON must be an exact string")
        try:
            decoded = json.loads(
                payload,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                "routed continual-dyad factory JSON is invalid or non-strict"
            ) from error
        if type(decoded) is not dict:
            raise ValueError("routed continual-dyad factory JSON must encode one object")
        return cls.from_config(decoded)


def _memory_config(
    *,
    agent_index: int,
    agent_lifetime_events: int,
) -> HCCLFeatureBoundMemoryConfig:
    memory = ExperientialMemoryConfig(
        capacity=_MEMORY_CAPACITY,
        observation_dim=_ROUTED_REPRESENTATION_DIM,
        key_dim=_ROUTED_REPRESENTATION_DIM,
        action_dim=_N_ACTIONS,
        outcome_dim=_ROUTED_REPRESENTATION_DIM,
        top_k=1,
        min_neighbors=1,
        distance_scale=1.0,
        min_similarity=0.0,
        min_effective_reliability=0.01,
        max_uncertainty=1.0,
        max_safety_cost=1.0,
        max_age=agent_lifetime_events,
        staleness_scale=float(agent_lifetime_events),
        utility_decay=1.0,
        eviction_utility_weight=1.0,
        eviction_recency_weight=1.0,
        recency_scale=10.0,
    )
    return HCCLFeatureBoundMemoryConfig(
        agent_index=agent_index,
        controller=LearnedExperientialMemoryControllerConfig(
            memory=memory,
            admission_step_size=0.05,
            retention_step_size=0.1,
            admission_threshold=0.0,
            initial_admission_bias=0.0,
            max_abs_admission_weight=8.0,
            max_abs_counterfactual_delta=1.0,
            retention_prior=0.5,
        ),
    )


def build_hccl_routed_continual_dyad_config(
    factory_config: HCCLRoutedContinualDyadFactoryConfig | None = None,
) -> HCCLRoutedContinualDyadConfig:
    """Build the exact routed owner config without initializing or executing."""

    selected = (
        HCCLRoutedContinualDyadFactoryConfig()
        if factory_config is None
        else factory_config
    )
    if type(selected) is not HCCLRoutedContinualDyadFactoryConfig:
        raise TypeError(
            "factory_config must be an exact HCCLRoutedContinualDyadFactoryConfig"
        )
    primitive = build_hccl_continual_dyad_config(
        HCCLContinualDyadFactoryConfig(
            schedule_profile=selected.schedule_profile,
        )
    )
    if primitive.agent_0.coordinator != primitive.agent_1.coordinator:
        raise ValueError("primitive factory returned asymmetric canonical coordinators")
    return HCCLRoutedContinualDyadConfig(
        hccl=primitive.hccl,
        coordinator=primitive.agent_0.coordinator,
        memory_agent_0=_memory_config(
            agent_index=0,
            agent_lifetime_events=selected.agent_lifetime_events,
        ),
        memory_agent_1=_memory_config(
            agent_index=1,
            agent_lifetime_events=selected.agent_lifetime_events,
        ),
        planner=PrototypeFactorizedPartnerPlannerV2Config(),
        context=primitive.context,
        bmp_agent_0=HCCLAuthenticatedBMPProjectionConfig(
            owner_digest=HCCL_ROUTED_CONTINUAL_DYAD_BMP_AGENT_0_OWNER_DIGEST,
        ),
        bmp_agent_1=HCCLAuthenticatedBMPProjectionConfig(
            owner_digest=HCCL_ROUTED_CONTINUAL_DYAD_BMP_AGENT_1_OWNER_DIGEST,
        ),
        binding_owner_digest=HCCL_ROUTED_CONTINUAL_DYAD_BINDING_OWNER_DIGEST,
        discount=0.99,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLRoutedContinualDyadFactoryInitialization:
    """One deterministic initialization result with no event authority."""

    factory_config: HCCLRoutedContinualDyadFactoryConfig
    dyad: HCCLRoutedContinualDyad
    state: HCCLRoutedContinualDyadState


class HCCLRoutedContinualDyadFactory:
    """Build and initialize the exact production-owned routed-R35 dyad."""

    def __init__(
        self,
        config: HCCLRoutedContinualDyadFactoryConfig | None = None,
    ) -> None:
        selected = HCCLRoutedContinualDyadFactoryConfig() if config is None else config
        if type(selected) is not HCCLRoutedContinualDyadFactoryConfig:
            raise TypeError("config must be an exact HCCLRoutedContinualDyadFactoryConfig")
        self._config = selected
        self._transaction_config = build_hccl_routed_continual_dyad_config(selected)

    @classmethod
    def mechanics_smoke(cls) -> HCCLRoutedContinualDyadFactory:
        return cls(HCCLRoutedContinualDyadFactoryConfig.mechanics_smoke())

    @classmethod
    def core_l2(cls) -> HCCLRoutedContinualDyadFactory:
        return cls(HCCLRoutedContinualDyadFactoryConfig.core_l2())

    @classmethod
    def core_l3(cls) -> HCCLRoutedContinualDyadFactory:
        return cls(HCCLRoutedContinualDyadFactoryConfig.core_l3())

    @property
    def config(self) -> HCCLRoutedContinualDyadFactoryConfig:
        return self._config

    @property
    def transaction_config(self) -> HCCLRoutedContinualDyadConfig:
        return self._transaction_config

    @property
    def dyad_config(self) -> HCCLRoutedContinualDyadConfig:
        return self._transaction_config

    def to_config(self) -> dict[str, object]:
        return self._config.to_config()

    def build(self) -> HCCLRoutedContinualDyad:
        """Construct one routed owner without initializing or executing an event."""

        return HCCLRoutedContinualDyad(self._transaction_config)

    def init(
        self,
        key: Array,
        *,
        initial_hard_action_masks: Array | None = None,
    ) -> HCCLRoutedContinualDyadFactoryInitialization:
        """Deterministically initialize all owners without advancing the schedule."""

        dyad = self.build()
        state = dyad.init(key, initial_hard_action_masks=initial_hard_action_masks)
        return HCCLRoutedContinualDyadFactoryInitialization(
            factory_config=self._config,
            dyad=dyad,
            state=state,
        )


__all__ = (
    "HCCL_ROUTED_CONTINUAL_DYAD_BINDING_OWNER_DIGEST",
    "HCCL_ROUTED_CONTINUAL_DYAD_BMP_AGENT_0_OWNER_DIGEST",
    "HCCL_ROUTED_CONTINUAL_DYAD_BMP_AGENT_1_OWNER_DIGEST",
    "HCCL_ROUTED_CONTINUAL_DYAD_FACTORY_CONFIG_SCHEMA",
    "HCCL_ROUTED_CONTINUAL_DYAD_FACTORY_EVIDENCE_LEVEL",
    "HCCL_ROUTED_CONTINUAL_DYAD_FACTORY_STATUS",
    "HCCLRoutedContinualDyadFactory",
    "HCCLRoutedContinualDyadFactoryConfig",
    "HCCLRoutedContinualDyadFactoryInitialization",
    "build_hccl_routed_continual_dyad_config",
)
