"""Inert protocol declaration for cadence-separated future-utility calibration.

This pure-stdlib module freezes identity, schedule geometry, intervention arms,
and lifecycle limits.  It cannot construct experience, enter the operational
lifecycle, run an arm, write an artifact, select an outcome, or authorize an
evidence claim.  Any later operational declaration must be a separate object.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import struct
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

PROTOCOL_SCHEMA: Final = (
    "alberta.compositional-future-utility-calibration-v3-cadence-separated."
    "protocol.full-base.v1"
)
PROTOCOL_NAMESPACE: Final = "alberta.compositional-future-utility-calibration-v3-cadence-separated"
PROTOCOL_NAMESPACE_SHA256: Final = (
    "12efd48b6159117b40a887ccbc2fad0a37a72b045746198999942321242766a2"
)
DEVELOPMENT_ROOT: Final = 317_707_403
DEVELOPMENT_ROOT_HEX: Final = "0x12EFD48B"

DECLARED_CONSUMED_DEVELOPMENT_ROOTS: Final = (329_631_721, 1_924_178_934)
DECLARED_CONSUMED_DEVELOPMENT_ROOT_HEXES: Final = ("0x13A5C7E9", "0x72B0A3F6")

PHASE_ORDER: Final = ("A", "B", "A", "D", "A", "C", "A", "B", "C", "A")
PHASE_LENGTHS: Final = (773, 811, 839, 877, 907, 937, 967, 999, 1020, 868)
PHASE_BOUNDARIES: Final = (0, 773, 1584, 2423, 3300, 4207, 5144, 6111, 7110, 8130, 8998)
CURATION_INTERVAL: Final = 32
CURATION_OPPORTUNITIES_PER_PHASE: Final = (24, 25, 26, 28, 28, 29, 30, 32, 32, 27)
TOTAL_STEPS: Final = 8_998
TOTAL_CURATION_OPPORTUNITIES: Final = 281

LEFT_PACK_SOURCE_ARM: Final = "dovetail_coverage_ancestor_headroom_leftpack"
EPSILON: Final = 0.1
ENTRY_WINDOW: Final = 64
TAIL_WINDOW: Final = 64
RAW_DIM: Final = 6
ACTIVE_SLOTS: Final = 11
CANDIDATE_SLOTS: Final = 8
ACTION_HEADS: Final = 2
ALLOCATED_MAX_DEPTH: Final = 3
TARGET_NAMES: Final = ("A", "B", "C")
LEARNER_OBSERVATION_FIELDS: Final = ("raw_rademacher_values",)
LEARNER_FEEDBACK_FIELDS: Final = ("selected_action_reward",)
RESETS_ALLOWED: Final = False

SOURCE_ARM_CONFIG: Final[Mapping[str, object]] = MappingProxyType(
    {
        "name": LEFT_PACK_SOURCE_ARM,
        "role": (
            "matched headroom arm with lowest-index margin-eligible destination "
            "placement"
        ),
        "composed_readout_enabled": True,
        "effective_max_depth": 3,
        "generation_strategy": "dovetail_product_coverage",
        "retention_slow_utility_decay": 0.999,
        "ancestor_utility_backup_decay": 0.95,
        "candidate_novelty_admission_bonus": 1.0,
        "topology_headroom_reserve": True,
        "topology_left_pack_destinations": True,
    }
)

SIGNATURE_NAMES: Final = ("A", "B", "C", "D", "shared_p45", "obsolete_p12")
SIGNATURE_RAW_INDICES: Final = (
    (1, 4, 5),
    (2, 4, 5),
    (3, 4, 5),
    (1, 2, 3),
    (4, 5),
    (1, 2),
)
SIGNATURE_ROLES: Final = (
    "recurring_root",
    "recurring_root",
    "recurring_root",
    "one_exposure_obsolete_root",
    "shared_recurring_intermediate",
    "one_exposure_obsolete_intermediate",
)
PHASE_TARGET_RAW_INDICES: Final = (
    (1, 4, 5),
    (2, 4, 5),
    (1, 4, 5),
    (1, 2, 3),
    (1, 4, 5),
    (3, 4, 5),
    (1, 4, 5),
    (2, 4, 5),
    (3, 4, 5),
    (1, 4, 5),
)

CANDIDATE_SCORING_MODE: Final = "legacy"
CANDIDATE_NOVELTY_ADMISSION_BONUS: Final = 0.0
FUTURE_UTILITY_TRACE_MODE: Final = "contribution"
FUTURE_UTILITY_NORMALIZATION_DECAY: Final = 0.99
FUTURE_UTILITY_RARE_TASK_POWER: Final = 0.0

CORRECTED_COMMON_BASE_CONFIG: Final[Mapping[str, object]] = MappingProxyType(
    {
        "ancestor_utility_backup_decay": 0.95,
        "candidate_count": 8,
        "candidate_imprint_scale": 0.1,
        "candidate_min_age": 16,
        "candidate_novelty_admission_bonus": 0.0,
        "candidate_novelty_floor": 0.05,
        "candidate_novelty_power": 1.0,
        "candidate_novelty_weight": 0.25,
        "candidate_score_energy_epsilon": 1e-6,
        "candidate_score_trace_decay": 0.95,
        "candidate_scoring_mode": "legacy",
        "candidate_selector": "legacy",
        "candidate_selector_exploration": 0.0,
        "candidate_selector_learning_rate": 1.0,
        "future_utility_normalization_decay": 0.99,
        "future_utility_rare_task_power": 0.0,
        "future_utility_task_activity_decay": 0.995,
        "future_utility_trace_mode": "contribution",
        "generation_strategy": "dovetail_product_coverage",
        "generator_resource_advantage_clip": 10.0,
        "generator_resource_contexts": 1,
        "generator_resource_cost_weight": 0.0,
        "generator_resource_discount": 0.995,
        "generator_resource_exploration": 0.01,
        "generator_resource_initial_preferences": None,
        "generator_resource_learning_rate": 1.0,
        "generator_resource_promotion_credit": 0.0,
        "generator_resource_update_rule": "hedge",
        "learn_generator_resources": False,
        "max_depth": 3,
        "min_feature_age": 16,
        "n_features": 11,
        "n_tasks": 2,
        "obgd_kappa": 2.0,
        "operation_prior": None,
        "parent_depth_prior": 0.1,
        "parent_novelty_weight": 0.1,
        "parent_temperature": 1.0,
        "promotion_blend": 1.0,
        "promotion_margin": 1.0,
        "promotion_output_mode": "scaled_candidate",
        "replacement_interval": 32,
        "residual_guidance": 1.0,
        "retention_depth_bonus": 0.05,
        "retention_product_min_count": 0,
        "retention_slow_utility_decay": 0.999,
        "retention_tanh_min_count": 0,
        "signed_tanh_scaffold_count": 0,
        "step_size_output": 0.01,
        "step_size_theta": 0.001,
        "topology_headroom_reserve": True,
        "topology_left_pack_destinations": True,
        "train_candidate_theta": False,
        "type": "CompositionalFeatureLearner",
        "use_obgd": True,
        "utility_decay": 0.995,
    }
)
CORRECTED_COMMON_BASE_CONFIG_SHA256: Final = (
    "a6ac2b6bdd6d37ca34a7a9ec08e582256fbdfaae08cf923f56695b4909525763"
)

LONG_TRACE_DECAY: Final = 0.999215304851532
LONG_TRACE_DECAY_F32_BITS: Final = "3f7fcc93"
INTERVENTION_FIELDS: Final = (
    "future_utility_mix",
    "future_utility_trace_decay",
    "future_utility_normalization",
)
ARM_NAMES: Final = (
    "current_mix0_decay095_none",
    "future_mix1_decay095_none",
    "calibrated_mix05_decay095_none",
    "normalized_mix1_decay095_uncertainty_age",
    "horizon_mix1_decay883_uncertainty_age",
)
ARM_ROLES: Final = (
    "current-utility reference with contribution traces retained",
    "unscaled future-utility endpoint",
    "equal current/future mixture calibration",
    "causally age-and-uncertainty-normalized future utility",
    "long-horizon normalized future utility (about 883-step half-life)",
)
ARM_PARAMETERS: Final = (
    (0.0, 0.95, "none"),
    (1.0, 0.95, "none"),
    (0.5, 0.95, "none"),
    (1.0, 0.95, "uncertainty_age"),
    (1.0, LONG_TRACE_DECAY, "uncertainty_age"),
)

LIFECYCLE_STATE: Final = "issued-unused"
MAXIMUM_OPERATIONAL_ENTRIES: Final = 1
OPERATIONAL_ENTRIES_CONSUMED: Final = 0
OPERATIONAL_ENTRIES_REMAINING: Final = 1
ENTRY_CAPABILITY_PROVIDED: Final = False

DEVELOPMENT_ONLY: Final = True
PANEL_EXECUTED: Final = False
RESULT_AVAILABLE: Final = False
EXECUTION_AUTHORIZED_BY_PROTOCOL: Final = False
OUTPUT_WRITES_ALLOWED: Final = False
ARTIFACT_AUTHORIZED: Final = False
THRESHOLD_AUTHORIZED: Final = False
WINNER_OR_DEFAULT_SELECTION_ALLOWED: Final = False
SEARCH_OR_TUNING_ALLOWED: Final = False
RETRY_ALLOWED: Final = False
RECOVERY_ALLOWED: Final = False
EVIDENCE_AUTHORIZED: Final = False
SCIENTIFIC_PROMOTION_ALLOWED: Final = False


def derive_namespace_sha256(namespace: str) -> str:
    """Derive the lowercase SHA-256 identity of an ASCII namespace."""

    if type(namespace) is not str or not namespace.isascii():
        raise ValueError("namespace must be an exact ASCII string")
    return hashlib.sha256(namespace.encode("ascii")).hexdigest()


def derive_root_from_namespace_sha256(namespace_sha256: str) -> int:
    """Derive the unsigned 32-bit root from the digest's first eight digits."""

    if (
        type(namespace_sha256) is not str
        or len(namespace_sha256) != 64
        or namespace_sha256 != namespace_sha256.lower()
        or any(character not in "0123456789abcdef" for character in namespace_sha256)
    ):
        raise ValueError("namespace digest must be 64 lowercase hexadecimal digits")
    return int(namespace_sha256[:8], 16)


def format_root_hex(root: int) -> str:
    """Return the exact eight-digit uppercase hexadecimal root form."""

    if type(root) is not int or not 0 <= root <= 0xFFFFFFFF:
        raise ValueError("root must be an unsigned 32-bit integer")
    return f"0x{root:08X}"


def _strict_integer_sequence(values: Sequence[int], *, label: str) -> tuple[int, ...]:
    if type(values) not in (list, tuple):
        raise ValueError(f"{label} must be an exact list or tuple")
    if not values or any(type(value) is not int for value in values):
        raise ValueError(f"{label} must contain exact integers")
    return tuple(values)


def reconstruct_phase_boundaries(phase_lengths: Sequence[int]) -> tuple[int, ...]:
    """Reconstruct half-open phase boundaries from positive phase lengths."""

    lengths = _strict_integer_sequence(phase_lengths, label="phase_lengths")
    if any(length <= 0 for length in lengths):
        raise ValueError("phase lengths must be positive")
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    return tuple(boundaries)


def reconstruct_boundary_residues(
    phase_boundaries: Sequence[int], curation_interval: int
) -> tuple[int, ...]:
    """Reconstruct every boundary's global-cadence residue."""

    boundaries = _strict_integer_sequence(phase_boundaries, label="phase_boundaries")
    if boundaries[0] != 0 or any(right <= left for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("phase boundaries must start at zero and strictly increase")
    if type(curation_interval) is not int or curation_interval <= 0:
        raise ValueError("curation interval must be a positive exact integer")
    return tuple(boundary % curation_interval for boundary in boundaries)


def reconstruct_curation_opportunities(
    phase_boundaries: Sequence[int], curation_interval: int
) -> tuple[int, ...]:
    """Count cadence-triggering updates in each half-open zero-based phase."""

    boundaries = _strict_integer_sequence(phase_boundaries, label="phase_boundaries")
    reconstruct_boundary_residues(boundaries, curation_interval)
    return tuple(
        right // curation_interval - left // curation_interval
        for left, right in zip(boundaries, boundaries[1:])
    )


@dataclasses.dataclass(frozen=True, slots=True)
class FutureUtilityArm:
    """One strict, immutable intervention arm description."""

    name: str
    role: str
    future_utility_mix: float
    future_utility_trace_decay: float
    future_utility_normalization: str

    def __post_init__(self) -> None:
        if (
            type(self.name) is not str
            or not self.name
            or type(self.role) is not str
            or not self.role
            or type(self.future_utility_mix) is not float
            or type(self.future_utility_trace_decay) is not float
            or type(self.future_utility_normalization) is not str
        ):
            raise ValueError("future-utility arm fields have strict types")
        if not 0.0 <= self.future_utility_mix <= 1.0:
            raise ValueError("future-utility mix must be in [0, 1]")
        if not 0.0 <= self.future_utility_trace_decay <= 1.0:
            raise ValueError("future-utility trace decay must be in [0, 1]")
        if self.future_utility_normalization not in ("none", "uncertainty_age"):
            raise ValueError("future-utility normalization is not declared")

    def to_config(self) -> dict[str, object]:
        """Return a fresh strict-JSON arm configuration."""

        interventions: dict[str, object] = {
            "future_utility_mix": self.future_utility_mix,
            "future_utility_trace_decay": self.future_utility_trace_decay,
            "future_utility_normalization": self.future_utility_normalization,
        }
        return {"name": self.name, "role": self.role, "interventions": interventions}


ARMS: Final = tuple(
    FutureUtilityArm(
        name=name,
        role=role,
        future_utility_mix=parameters[0],
        future_utility_trace_decay=parameters[1],
        future_utility_normalization=parameters[2],
    )
    for name, role, parameters in zip(ARM_NAMES, ARM_ROLES, ARM_PARAMETERS, strict=True)
)


def reconstruct_arm_learner_config(arm: FutureUtilityArm) -> dict[str, object]:
    """Reconstruct one complete 59-field learner config from frozen literals."""

    if type(arm) is not FutureUtilityArm or arm not in ARMS:
        raise ValueError("arm must be one of the exact frozen arm records")
    config = dict(CORRECTED_COMMON_BASE_CONFIG)
    config.update(
        {
            "future_utility_mix": arm.future_utility_mix,
            "future_utility_trace_decay": arm.future_utility_trace_decay,
            "future_utility_normalization": arm.future_utility_normalization,
        }
    )
    if len(config) != 59 or set(config) != {
        *CORRECTED_COMMON_BASE_CONFIG,
        *INTERVENTION_FIELDS,
    }:
        raise RuntimeError("frozen learner config does not reconstruct exactly")
    return config


def reconstruct_varying_intervention_fields(
    arms: Sequence[FutureUtilityArm],
) -> tuple[str, ...]:
    """Return declared intervention fields whose values differ across arms."""

    if type(arms) not in (list, tuple) or not arms:
        raise ValueError("arms must be a nonempty exact list or tuple")
    if any(type(arm) is not FutureUtilityArm for arm in arms):
        raise ValueError("arms must contain exact FutureUtilityArm records")
    return tuple(
        field for field in INTERVENTION_FIELDS if len({getattr(arm, field) for arm in arms}) > 1
    )


@dataclasses.dataclass(frozen=True, slots=True)
class ProtocolLifecycle:
    """Inert issued-root accounting; this record cannot consume an entry."""

    state: str = LIFECYCLE_STATE
    maximum_operational_entries: int = MAXIMUM_OPERATIONAL_ENTRIES
    entries_consumed: int = OPERATIONAL_ENTRIES_CONSUMED
    entries_remaining: int = OPERATIONAL_ENTRIES_REMAINING
    entry_capability_provided: bool = ENTRY_CAPABILITY_PROVIDED

    def __post_init__(self) -> None:
        if not (
            type(self.state) is str
            and self.state == LIFECYCLE_STATE
            and type(self.maximum_operational_entries) is int
            and self.maximum_operational_entries == MAXIMUM_OPERATIONAL_ENTRIES
            and type(self.entries_consumed) is int
            and self.entries_consumed == OPERATIONAL_ENTRIES_CONSUMED
            and type(self.entries_remaining) is int
            and self.entries_remaining == OPERATIONAL_ENTRIES_REMAINING
            and type(self.entry_capability_provided) is bool
            and self.entry_capability_provided is ENTRY_CAPABILITY_PROVIDED
        ):
            raise ValueError("protocol lifecycle is frozen")

    def to_config(self) -> dict[str, object]:
        return {
            "state": self.state,
            "maximum_operational_entries": self.maximum_operational_entries,
            "entries_consumed": self.entries_consumed,
            "entries_remaining": self.entries_remaining,
            "entry_capability_provided": self.entry_capability_provided,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class ProtocolAuthority:
    """Fail-closed statement of authority absent from this protocol slice."""

    development_only: bool = DEVELOPMENT_ONLY
    panel_executed: bool = PANEL_EXECUTED
    result_available: bool = RESULT_AVAILABLE
    execution_authorized_by_protocol: bool = EXECUTION_AUTHORIZED_BY_PROTOCOL
    output_writes_allowed: bool = OUTPUT_WRITES_ALLOWED
    artifact_authorized: bool = ARTIFACT_AUTHORIZED
    threshold_authorized: bool = THRESHOLD_AUTHORIZED
    winner_or_default_selection_allowed: bool = WINNER_OR_DEFAULT_SELECTION_ALLOWED
    search_or_tuning_allowed: bool = SEARCH_OR_TUNING_ALLOWED
    retry_allowed: bool = RETRY_ALLOWED
    recovery_allowed: bool = RECOVERY_ALLOWED
    evidence_authorized: bool = EVIDENCE_AUTHORIZED
    scientific_promotion_allowed: bool = SCIENTIFIC_PROMOTION_ALLOWED

    def __post_init__(self) -> None:
        values = dataclasses.astuple(self)
        if (
            any(type(value) is not bool for value in values)
            or self.development_only is not True
            or any(values[1:])
        ):
            raise ValueError("protocol authority is frozen and fail-closed")

    def to_config(self) -> dict[str, object]:
        return {
            "development_only": self.development_only,
            "panel_executed": self.panel_executed,
            "result_available": self.result_available,
            "execution_authorized_by_protocol": self.execution_authorized_by_protocol,
            "output_writes_allowed": self.output_writes_allowed,
            "artifact_authorized": self.artifact_authorized,
            "threshold_authorized": self.threshold_authorized,
            "winner_or_default_selection_allowed": self.winner_or_default_selection_allowed,
            "search_or_tuning_allowed": self.search_or_tuning_allowed,
            "retry_allowed": self.retry_allowed,
            "recovery_allowed": self.recovery_allowed,
            "evidence_authorized": self.evidence_authorized,
            "scientific_promotion_allowed": self.scientific_promotion_allowed,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class CompositionalFutureUtilityCalibrationV3Protocol:
    """The exact immutable cadence-separated protocol specification."""

    schema: str = PROTOCOL_SCHEMA
    namespace: str = PROTOCOL_NAMESPACE
    namespace_sha256: str = PROTOCOL_NAMESPACE_SHA256
    development_root: int = DEVELOPMENT_ROOT
    development_root_hex: str = DEVELOPMENT_ROOT_HEX
    declared_consumed_development_roots: tuple[int, ...] = DECLARED_CONSUMED_DEVELOPMENT_ROOTS
    phase_order: tuple[str, ...] = PHASE_ORDER
    phase_lengths: tuple[int, ...] = PHASE_LENGTHS
    phase_boundaries: tuple[int, ...] = PHASE_BOUNDARIES
    curation_interval: int = CURATION_INTERVAL
    curation_opportunities_per_phase: tuple[int, ...] = CURATION_OPPORTUNITIES_PER_PHASE
    left_pack_source_arm: str = LEFT_PACK_SOURCE_ARM
    epsilon: float = EPSILON
    entry_window: int = ENTRY_WINDOW
    tail_window: int = TAIL_WINDOW
    target_names: tuple[str, ...] = TARGET_NAMES
    arms: tuple[FutureUtilityArm, ...] = ARMS
    lifecycle: ProtocolLifecycle = ProtocolLifecycle()
    authority: ProtocolAuthority = ProtocolAuthority()

    def __post_init__(self) -> None:
        exact = (
            type(self.schema) is str
            and self.schema == PROTOCOL_SCHEMA
            and type(self.namespace) is str
            and self.namespace == PROTOCOL_NAMESPACE
            and type(self.namespace_sha256) is str
            and self.namespace_sha256 == PROTOCOL_NAMESPACE_SHA256
            and type(self.development_root) is int
            and self.development_root == DEVELOPMENT_ROOT
            and type(self.development_root_hex) is str
            and self.development_root_hex == DEVELOPMENT_ROOT_HEX
            and type(self.declared_consumed_development_roots) is tuple
            and self.declared_consumed_development_roots == DECLARED_CONSUMED_DEVELOPMENT_ROOTS
            and type(self.phase_order) is tuple
            and self.phase_order == PHASE_ORDER
            and type(self.phase_lengths) is tuple
            and self.phase_lengths == PHASE_LENGTHS
            and type(self.phase_boundaries) is tuple
            and self.phase_boundaries == PHASE_BOUNDARIES
            and type(self.curation_interval) is int
            and self.curation_interval == CURATION_INTERVAL
            and type(self.curation_opportunities_per_phase) is tuple
            and self.curation_opportunities_per_phase == CURATION_OPPORTUNITIES_PER_PHASE
            and type(self.left_pack_source_arm) is str
            and self.left_pack_source_arm == LEFT_PACK_SOURCE_ARM
            and type(self.epsilon) is float
            and self.epsilon == EPSILON
            and type(self.entry_window) is int
            and self.entry_window == ENTRY_WINDOW
            and type(self.tail_window) is int
            and self.tail_window == TAIL_WINDOW
            and type(self.target_names) is tuple
            and self.target_names == TARGET_NAMES
            and type(self.arms) is tuple
            and all(type(arm) is FutureUtilityArm for arm in self.arms)
            and self.arms == ARMS
            and type(self.lifecycle) is ProtocolLifecycle
            and self.lifecycle == ProtocolLifecycle()
            and type(self.authority) is ProtocolAuthority
            and self.authority == ProtocolAuthority()
        )
        derived = (
            derive_namespace_sha256(self.namespace) == self.namespace_sha256
            and derive_root_from_namespace_sha256(self.namespace_sha256) == self.development_root
            and format_root_hex(self.development_root) == self.development_root_hex
            and self.development_root not in self.declared_consumed_development_roots
            and tuple(format_root_hex(root) for root in self.declared_consumed_development_roots)
            == DECLARED_CONSUMED_DEVELOPMENT_ROOT_HEXES
            and len(self.phase_order)
            == len(self.phase_lengths)
            == len(self.curation_opportunities_per_phase)
            and reconstruct_phase_boundaries(self.phase_lengths) == self.phase_boundaries
            and reconstruct_curation_opportunities(self.phase_boundaries, self.curation_interval)
            == self.curation_opportunities_per_phase
            and self.phase_boundaries[-1] == TOTAL_STEPS
            and sum(self.curation_opportunities_per_phase) == TOTAL_CURATION_OPPORTUNITIES
            and struct.pack(">f", LONG_TRACE_DECAY).hex() == LONG_TRACE_DECAY_F32_BITS
            and reconstruct_varying_intervention_fields(self.arms) == INTERVENTION_FIELDS
            and len(CORRECTED_COMMON_BASE_CONFIG) == 56
            and canonical_json_sha256(dict(CORRECTED_COMMON_BASE_CONFIG))
            == CORRECTED_COMMON_BASE_CONFIG_SHA256
            and all(len(reconstruct_arm_learner_config(arm)) == 59 for arm in self.arms)
            and len(SIGNATURE_NAMES)
            == len(SIGNATURE_RAW_INDICES)
            == len(SIGNATURE_ROLES)
            and self.target_names == SIGNATURE_NAMES[:3]
            and PHASE_TARGET_RAW_INDICES
            == tuple(
                SIGNATURE_RAW_INDICES[SIGNATURE_NAMES.index(name)]
                for name in self.phase_order
            )
            and SOURCE_ARM_CONFIG["name"] == self.left_pack_source_arm
            and SOURCE_ARM_CONFIG["effective_max_depth"] == ALLOCATED_MAX_DEPTH
        )
        if not exact or not derived:
            raise ValueError("the cadence-separated protocol is frozen")

    def to_config(self) -> dict[str, object]:
        """Return a fresh strict-JSON representation of the frozen protocol."""

        identity: dict[str, object] = {
            "namespace": self.namespace,
            "namespace_sha256": self.namespace_sha256,
            "development_root": self.development_root,
            "development_root_hex": self.development_root_hex,
            "declared_consumed_development_roots": list(self.declared_consumed_development_roots),
            "declared_consumed_development_root_hexes": list(
                DECLARED_CONSUMED_DEVELOPMENT_ROOT_HEXES
            ),
        }
        schedule: dict[str, object] = {
            "phase_order": list(self.phase_order),
            "phase_lengths": list(self.phase_lengths),
            "phase_boundaries": list(self.phase_boundaries),
            "boundary_residues": list(
                reconstruct_boundary_residues(self.phase_boundaries, self.curation_interval)
            ),
            "curation_interval": self.curation_interval,
            "curation_opportunities_per_phase": list(self.curation_opportunities_per_phase),
            "total_steps": TOTAL_STEPS,
            "total_curation_opportunities": TOTAL_CURATION_OPPORTUNITIES,
        }
        source_geometry: dict[str, object] = {
            "left_pack_source_arm": self.left_pack_source_arm,
            "epsilon": self.epsilon,
            "entry_window": self.entry_window,
            "tail_window": self.tail_window,
            "raw_dim": RAW_DIM,
            "active_slots": ACTIVE_SLOTS,
            "candidate_slots": CANDIDATE_SLOTS,
            "action_heads": ACTION_HEADS,
            "allocated_max_depth": ALLOCATED_MAX_DEPTH,
            "target_names": list(self.target_names),
            "learner_observation_fields": list(LEARNER_OBSERVATION_FIELDS),
            "learner_feedback_fields": list(LEARNER_FEEDBACK_FIELDS),
            "resets_allowed": RESETS_ALLOWED,
            "source_arm_config": dict(SOURCE_ARM_CONFIG),
        }
        corrected_common_base: dict[str, object] = {
            "invariant_field_count": len(CORRECTED_COMMON_BASE_CONFIG),
            "invariant_fields_sha256": CORRECTED_COMMON_BASE_CONFIG_SHA256,
            "invariant_fields": dict(CORRECTED_COMMON_BASE_CONFIG),
        }
        task_semantics: dict[str, object] = {
            "signature_names": list(SIGNATURE_NAMES),
            "signature_raw_indices": [list(indices) for indices in SIGNATURE_RAW_INDICES],
            "signature_roles": list(SIGNATURE_ROLES),
            "phase_target_raw_indices": [
                list(indices) for indices in PHASE_TARGET_RAW_INDICES
            ],
            "observation_values": [-1.0, 1.0],
            "observation_probabilities": [0.5, 0.5],
            "observation_coordinates_independent": True,
            "target_value_operation": "product",
            "action_values": [0, 1],
            "action_reward_multipliers": [-1.0, 1.0],
            "action_sign_equation": "2 * action - 1",
            "executed_reward_equation": (
                "action_reward_multiplier * target_value"
            ),
            "greedy_action_rule": "first_argmax_of_composed_full_q",
            "exploration_rule": (
                "epsilon_mask_selects_pinned_uniform_random_action"
            ),
            "learner_target_rule": "selected_head_reward_other_head_nan",
            "counterfactual_action_reward_is_learner_visible": False,
            "phase_identity_is_learner_visible": False,
        }
        return {
            "schema": self.schema,
            "identity": identity,
            "schedule": schedule,
            "source_geometry": source_geometry,
            "corrected_common_base": corrected_common_base,
            "task_semantics": task_semantics,
            "arms": [arm.to_config() for arm in self.arms],
            "lifecycle": self.lifecycle.to_config(),
            "authority": self.authority.to_config(),
        }


def canonical_json(value: object) -> str:
    """Encode a value with the protocol's exact canonical JSON rules."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_json_sha256(value: object) -> str:
    """Hash the ASCII bytes of the exact canonical JSON encoding."""

    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def canonical_protocol_config() -> dict[str, object]:
    """Return the one canonical protocol configuration."""

    return CompositionalFutureUtilityCalibrationV3Protocol().to_config()


def _validate_strict_json(value: object) -> None:
    if type(value) is dict:
        for key, nested in value.items():
            if type(key) is not str:
                raise ValueError("protocol config must use strict JSON string keys")
            _validate_strict_json(nested)
        return
    if type(value) is list:
        for nested in value:
            _validate_strict_json(nested)
        return
    if type(value) not in (str, int, float, bool, type(None)):
        raise ValueError("protocol config must use strict JSON container and scalar types")


def reconstruct_protocol(config: object) -> CompositionalFutureUtilityCalibrationV3Protocol:
    """Strictly reconstruct the protocol, rejecting any shape or value drift."""

    _validate_strict_json(config)
    if type(config) is not dict or canonical_json(config) != canonical_json(
        canonical_protocol_config()
    ):
        raise ValueError("configuration does not match the canonical frozen protocol")
    return CompositionalFutureUtilityCalibrationV3Protocol()


def protocol_config_sha256(protocol: CompositionalFutureUtilityCalibrationV3Protocol) -> str:
    """Hash one exact protocol configuration."""

    if type(protocol) is not CompositionalFutureUtilityCalibrationV3Protocol:
        raise ValueError("protocol must be the exact frozen protocol type")
    return canonical_json_sha256(protocol.to_config())


PROTOCOL_CONFIG_SHA256: Final = "09b7d06ae720f1a2aeb167ae10e4dbde46dff5437659e431bfff79a8445dc16c"

__all__ = [
    "ACTION_HEADS",
    "ACTIVE_SLOTS",
    "ALLOCATED_MAX_DEPTH",
    "ARMS",
    "ARM_NAMES",
    "ARM_PARAMETERS",
    "ARM_ROLES",
    "ARTIFACT_AUTHORIZED",
    "CANDIDATE_NOVELTY_ADMISSION_BONUS",
    "CANDIDATE_SCORING_MODE",
    "CANDIDATE_SLOTS",
    "CORRECTED_COMMON_BASE_CONFIG",
    "CORRECTED_COMMON_BASE_CONFIG_SHA256",
    "CompositionalFutureUtilityCalibrationV3Protocol",
    "CURATION_INTERVAL",
    "CURATION_OPPORTUNITIES_PER_PHASE",
    "DECLARED_CONSUMED_DEVELOPMENT_ROOTS",
    "DECLARED_CONSUMED_DEVELOPMENT_ROOT_HEXES",
    "DEVELOPMENT_ONLY",
    "DEVELOPMENT_ROOT",
    "DEVELOPMENT_ROOT_HEX",
    "ENTRY_CAPABILITY_PROVIDED",
    "ENTRY_WINDOW",
    "EPSILON",
    "EVIDENCE_AUTHORIZED",
    "EXECUTION_AUTHORIZED_BY_PROTOCOL",
    "FUTURE_UTILITY_NORMALIZATION_DECAY",
    "FUTURE_UTILITY_RARE_TASK_POWER",
    "FUTURE_UTILITY_TRACE_MODE",
    "FutureUtilityArm",
    "INTERVENTION_FIELDS",
    "LIFECYCLE_STATE",
    "LEFT_PACK_SOURCE_ARM",
    "LONG_TRACE_DECAY",
    "LONG_TRACE_DECAY_F32_BITS",
    "MAXIMUM_OPERATIONAL_ENTRIES",
    "OPERATIONAL_ENTRIES_CONSUMED",
    "OPERATIONAL_ENTRIES_REMAINING",
    "OUTPUT_WRITES_ALLOWED",
    "PANEL_EXECUTED",
    "PHASE_BOUNDARIES",
    "PHASE_LENGTHS",
    "PHASE_ORDER",
    "PHASE_TARGET_RAW_INDICES",
    "PROTOCOL_CONFIG_SHA256",
    "PROTOCOL_NAMESPACE",
    "PROTOCOL_NAMESPACE_SHA256",
    "PROTOCOL_SCHEMA",
    "ProtocolAuthority",
    "ProtocolLifecycle",
    "RAW_DIM",
    "RECOVERY_ALLOWED",
    "RESULT_AVAILABLE",
    "RESETS_ALLOWED",
    "RETRY_ALLOWED",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "SEARCH_OR_TUNING_ALLOWED",
    "SIGNATURE_NAMES",
    "SIGNATURE_RAW_INDICES",
    "SIGNATURE_ROLES",
    "SOURCE_ARM_CONFIG",
    "TAIL_WINDOW",
    "TARGET_NAMES",
    "THRESHOLD_AUTHORIZED",
    "TOTAL_CURATION_OPPORTUNITIES",
    "TOTAL_STEPS",
    "WINNER_OR_DEFAULT_SELECTION_ALLOWED",
    "canonical_json",
    "canonical_json_sha256",
    "canonical_protocol_config",
    "derive_namespace_sha256",
    "derive_root_from_namespace_sha256",
    "format_root_hex",
    "protocol_config_sha256",
    "reconstruct_boundary_residues",
    "reconstruct_arm_learner_config",
    "reconstruct_curation_opportunities",
    "reconstruct_phase_boundaries",
    "reconstruct_protocol",
    "reconstruct_varying_intervention_fields",
]
