"""Unbound one-factor selection requirements for the HCCL causal core.

The nine arms in this module are requirements for a future runtime owner, not
benchmark runs or an executable runtime.  Exactly one requested selection
differs from ``full`` in each ablation.  A conforming executor would have to
instantiate every mechanism, compute every named routed and unrouted
alternative, match persistent shapes and exogenous key roles, make eight
same-receipt world proposals, and preserve scheduled opportunities.  None of
those execution preconditions is currently implemented or runtime-validated.

No class here binds an owner, executes an event, reserves a seed, writes an
artifact, sets a threshold, or grants evidence or promotion authority.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from enum import StrEnum
from typing import Final, Literal, cast

HCCL_CAUSAL_CORE_ARM_CONFIG_SCHEMA: Final = (
    "alberta.hccl-causal-core.arm-config.v2"
)
HCCL_CAUSAL_CORE_ARM_POLICY_SCHEMA: Final = (
    "alberta.hccl-causal-core.arm-route-policy.v2"
)
HCCL_CAUSAL_CORE_ARM_STATUS: Final = (
    "l0-development-unbound-selection-requirement-not-executable"
)
HCCL_CAUSAL_CORE_ARM_SCIENTIFIC_PROMOTION_ALLOWED: Final = False

FeatureRankSelection = Literal["learned", "random"]
MemoryDispatchSelection = Literal["memory", "base"]
PartnerBeliefSelection = Literal["learned", "uniform"]
PlannerDispatchSelection = Literal["planner", "memory"]

_POLICY_INTERVENTION_FIELDS: Final = (
    "fast_state_routed",
    "slow_context_routed",
    "lineage_rescue_routed",
    "feature_rank_selection",
    "feature_consumers_routed",
    "memory_dispatch_selection",
    "partner_belief_selection",
    "planner_dispatch_selection",
)


class HCCLCausalCoreArmName(StrEnum):
    """The fixed one-factor causal-core panel."""

    FULL = "full"
    FAST_STATE_UNROUTED = "fast_state_unrouted"
    SLOW_CONTEXT_UNROUTED = "slow_context_unrouted"
    LINEAGE_RESCUE_UNROUTED = "lineage_rescue_unrouted"
    FEATURE_RANDOM_RANK = "feature_random_rank"
    FEATURE_CONSUMERS_UNROUTED = "feature_consumers_unrouted"
    MEMORY_DISPATCH_UNROUTED = "memory_dispatch_unrouted"
    UNIFORM_PARTNER_BELIEF = "uniform_partner_belief"
    PLANNER_DISPATCH_UNROUTED = "planner_dispatch_unrouted"


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLCausalCoreArmRoutePolicy:
    """One exact selection requirement for a future matched-alternative owner."""

    fast_state_routed: bool = True
    slow_context_routed: bool = True
    lineage_rescue_routed: bool = True
    feature_rank_selection: FeatureRankSelection = "learned"
    feature_consumers_routed: bool = True
    memory_dispatch_selection: MemoryDispatchSelection = "memory"
    partner_belief_selection: PartnerBeliefSelection = "learned"
    planner_dispatch_selection: PlannerDispatchSelection = "planner"

    def __post_init__(self) -> None:
        for name in (
            "fast_state_routed",
            "slow_context_routed",
            "lineage_rescue_routed",
            "feature_consumers_routed",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact bool")
        if self.feature_rank_selection not in ("learned", "random"):
            raise ValueError("feature_rank_selection must be learned or random")
        if self.memory_dispatch_selection not in ("memory", "base"):
            raise ValueError("memory_dispatch_selection must be memory or base")
        if self.partner_belief_selection not in ("learned", "uniform"):
            raise ValueError("partner_belief_selection must be learned or uniform")
        if self.planner_dispatch_selection not in ("planner", "memory"):
            raise ValueError("planner_dispatch_selection must be planner or memory")

    def to_config(self) -> dict[str, object]:
        """Return the strict JSON requirement and current runtime nonclaims."""

        return {
            "schema": HCCL_CAUSAL_CORE_ARM_POLICY_SCHEMA,
            "type": type(self).__name__,
            **dataclasses.asdict(self),
            "all_mechanisms_instantiated": False,
            "all_mechanisms_instantiated_required_if_executed": True,
            "all_routed_and_unrouted_alternatives_computed": False,
            "all_routed_and_unrouted_alternatives_computed_required_if_executed": True,
            "persistent_shapes_matched": False,
            "persistent_shapes_matched_required_if_executed": True,
            "paired_exogenous_key_roles": False,
            "paired_exogenous_key_roles_required_if_executed": True,
            "environment_proposal_calls_per_event": 0,
            "environment_proposal_calls_per_event_required_if_executed": 8,
            "scheduled_candidate_opportunities_preserved": False,
            "scheduled_candidate_opportunities_preserved_required_if_executed": True,
            "scheduled_curation_opportunities_preserved": False,
            "scheduled_curation_opportunities_preserved_required_if_executed": True,
            "mechanism_rng_static_rules_preserved": False,
            "mechanism_rng_static_rules_preserved_required_if_executed": True,
            "runtime_owner_bound": False,
            "runtime_alternatives_validated": False,
            "execution_implementation_available": False,
            "matched_total_work_claimed": False,
            "equal_flops_claimed": False,
            "equal_wall_time_claimed": False,
            "benchmark_execution_authorized": False,
            "artifact_writes_authorized": False,
            "scientific_promotion_allowed": False,
        }


_FULL_POLICY: Final = HCCLCausalCoreArmRoutePolicy()

_POLICIES: Final[dict[HCCLCausalCoreArmName, HCCLCausalCoreArmRoutePolicy]] = {
    HCCLCausalCoreArmName.FULL: _FULL_POLICY,
    HCCLCausalCoreArmName.FAST_STATE_UNROUTED: dataclasses.replace(
        _FULL_POLICY, fast_state_routed=False
    ),
    HCCLCausalCoreArmName.SLOW_CONTEXT_UNROUTED: dataclasses.replace(
        _FULL_POLICY, slow_context_routed=False
    ),
    HCCLCausalCoreArmName.LINEAGE_RESCUE_UNROUTED: dataclasses.replace(
        _FULL_POLICY, lineage_rescue_routed=False
    ),
    HCCLCausalCoreArmName.FEATURE_RANDOM_RANK: dataclasses.replace(
        _FULL_POLICY, feature_rank_selection="random"
    ),
    HCCLCausalCoreArmName.FEATURE_CONSUMERS_UNROUTED: dataclasses.replace(
        _FULL_POLICY, feature_consumers_routed=False
    ),
    HCCLCausalCoreArmName.MEMORY_DISPATCH_UNROUTED: dataclasses.replace(
        _FULL_POLICY, memory_dispatch_selection="base"
    ),
    HCCLCausalCoreArmName.UNIFORM_PARTNER_BELIEF: dataclasses.replace(
        _FULL_POLICY, partner_belief_selection="uniform"
    ),
    HCCLCausalCoreArmName.PLANNER_DISPATCH_UNROUTED: dataclasses.replace(
        _FULL_POLICY, planner_dispatch_selection="memory"
    ),
}


def hccl_causal_core_arm_policy(
    arm: HCCLCausalCoreArmName,
) -> HCCLCausalCoreArmRoutePolicy:
    """Return the immutable policy for one exact arm enum."""

    if type(arm) is not HCCLCausalCoreArmName:
        raise TypeError("arm must be an exact HCCLCausalCoreArmName")
    return _POLICIES[arm]


def hccl_causal_core_arm_intervention_fields(
    arm: HCCLCausalCoreArmName,
) -> tuple[str, ...]:
    """Return policy fields whose selection differs from ``full``."""

    policy = hccl_causal_core_arm_policy(arm)
    return tuple(
        name
        for name in _POLICY_INTERVENTION_FIELDS
        if getattr(policy, name) != getattr(_FULL_POLICY, name)
    )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLCausalCoreArmConfig:
    """Strict unbound selection requirement for one causal-core arm."""

    arm: HCCLCausalCoreArmName = HCCLCausalCoreArmName.FULL

    def __post_init__(self) -> None:
        if type(self.arm) is not HCCLCausalCoreArmName:
            raise TypeError("arm must be an exact HCCLCausalCoreArmName")
        differences = hccl_causal_core_arm_intervention_fields(self.arm)
        if self.arm is HCCLCausalCoreArmName.FULL:
            if differences:
                raise RuntimeError("full arm must have no routed intervention")
        elif len(differences) != 1:
            raise RuntimeError("each ablation must differ from full in exactly one selection")

    @property
    def policy(self) -> HCCLCausalCoreArmRoutePolicy:
        return hccl_causal_core_arm_policy(self.arm)

    @property
    def intervention_fields(self) -> tuple[str, ...]:
        return hccl_causal_core_arm_intervention_fields(self.arm)

    def to_config(self) -> dict[str, object]:
        """Return a strict, unbound, nonexecuting arm requirement manifest."""

        return {
            "schema": HCCL_CAUSAL_CORE_ARM_CONFIG_SCHEMA,
            "type": type(self).__name__,
            "mechanism_status": HCCL_CAUSAL_CORE_ARM_STATUS,
            "arm": self.arm.value,
            "intervention_fields": list(self.intervention_fields),
            "policy": self.policy.to_config(),
            "selection_contract_only": True,
            "unbound_selection_requirement": True,
            "all_mechanisms_instantiated": False,
            "all_mechanisms_instantiated_required_if_executed": True,
            "all_routed_and_unrouted_alternatives_computed": False,
            "all_routed_and_unrouted_alternatives_computed_required_if_executed": True,
            "runtime_owner_bound": False,
            "runtime_alternatives_validated": False,
            "execution_implementation_available": False,
            "execution_authorized": False,
            "seed_reservation_or_consumption_authorized": False,
            "artifact_writes_authorized": False,
            "thresholds_defined": False,
            "evidence_claimed": False,
            "scientific_promotion_allowed": False,
        }

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, object],
    ) -> HCCLCausalCoreArmConfig:
        """Reconstruct only the exact current selection manifest."""

        if type(payload) is not dict:
            raise TypeError("arm config must be an exact dict")
        raw_arm = payload.get("arm")
        if type(raw_arm) is not str:
            raise ValueError("arm must be an exact string")
        try:
            arm = HCCLCausalCoreArmName(raw_arm)
        except ValueError as error:
            raise ValueError("arm is not one of the fixed causal-core arms") from error
        candidate = cls(arm=arm)
        if _canonical_json_bytes(cast(object, payload)) != _canonical_json_bytes(
            candidate.to_config()
        ):
            raise ValueError("arm config is noncanonical or unsupported")
        return candidate


HCCL_CAUSAL_CORE_ARM_PANEL: Final = tuple(HCCLCausalCoreArmName)

if len(HCCL_CAUSAL_CORE_ARM_PANEL) != 9:  # pragma: no cover - import-time invariant
    raise RuntimeError("the causal-core panel must contain exactly nine arms")


__all__ = (
    "HCCL_CAUSAL_CORE_ARM_CONFIG_SCHEMA",
    "HCCL_CAUSAL_CORE_ARM_PANEL",
    "HCCL_CAUSAL_CORE_ARM_POLICY_SCHEMA",
    "HCCL_CAUSAL_CORE_ARM_SCIENTIFIC_PROMOTION_ALLOWED",
    "HCCL_CAUSAL_CORE_ARM_STATUS",
    "HCCLCausalCoreArmConfig",
    "HCCLCausalCoreArmName",
    "HCCLCausalCoreArmRoutePolicy",
    "hccl_causal_core_arm_intervention_fields",
    "hccl_causal_core_arm_policy",
)
