"""Identity-free mechanism specifications for future-utility calibration.

This module projects already-declared arm mechanisms onto a caller-supplied
control-life base.  It cannot issue a root, build a stream, execute a panel,
write output, select a setting, or authorize evidence or promotion.
"""

from __future__ import annotations

import dataclasses
import json
import struct
from collections.abc import Mapping, Sequence
from typing import Any, Final, cast

import numpy as np

from alberta_framework.evaluation import compositional_control_life_development as control
from alberta_framework.evaluation import compositional_future_utility_panel_core as cadence

OpportunityPartition = cadence.OpportunityPartition
REQUIRED_CADENCE_MUTATION_MASK_NAMES: Final = (
    cadence.REQUIRED_CADENCE_MUTATION_MASK_NAMES
)

DEVELOPMENT_ONLY: Final = True
PANEL_EXECUTION_AUTHORIZED: Final = False
ROOT_ISSUANCE_AUTHORIZED: Final = False
OUTPUT_WRITES_ALLOWED: Final = False
EVIDENCE_AUTHORIZED: Final = False
SCIENTIFIC_PROMOTION_ALLOWED: Final = False

INTERVENTION_FIELDS: Final = (
    "future_utility_mix",
    "future_utility_trace_decay",
    "future_utility_normalization",
)
COMMON_DEPARTURE_FIELDS: Final = (
    "candidate_scoring_mode",
    "candidate_novelty_admission_bonus",
    "future_utility_trace_mode",
    "future_utility_rare_task_power",
)
NORMALIZATION_DECAY: Final = 0.99
RESOURCE_ACCOUNTING_SCOPE: Final = (
    "exact persistent learner-state bytes, matched shared-base logical cell/update "
    "counts, intervention-specific named cell counts, and measured curation-audit "
    "arrays; excludes behavior-dependent branch work, source arrays, full scan "
    "telemetry, compiler workspaces, and compiled FLOPs"
)
PRIMARY_ENDPOINT_NAMES: Final = (
    "margin_passes",
    "promotions",
    "candidate_refreshes",
    "cascade_losses",
    "target_admission_loss_end",
    "pre_recurrence_presence",
    "target_occupancy",
    "pre_recurrence_ranks",
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _exact_positive_int(value: object, *, field: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field} must be an exact integer")
    if value < 1:
        raise ValueError(f"{field} must be positive")
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class FutureUtilityArmSpec:
    """One mechanism-only arm with no protocol or execution identity."""

    name: str
    role: str
    mix: float
    trace_decay: float
    normalization: str

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError("arm name must be a non-empty exact string")
        if type(self.role) is not str or not self.role:
            raise ValueError("arm role must be a non-empty exact string")
        if type(self.mix) is not float:
            raise TypeError("arm mix must be an exact float")
        if not 0.0 <= self.mix <= 1.0:
            raise ValueError("arm mix must be in [0, 1]")
        if type(self.trace_decay) is not float:
            raise TypeError("arm trace_decay must be an exact float")
        if not 0.0 <= self.trace_decay <= 1.0:
            raise ValueError("arm trace_decay must be in [0, 1]")
        if self.normalization not in {"none", "uncertainty_age"}:
            raise ValueError("arm normalization is unsupported")


@dataclasses.dataclass(frozen=True, slots=True)
class FutureUtilityWorkGeometry:
    """Fixed shapes used for logical work accounting."""

    steps: int
    curation_interval: int
    active_slots: int
    candidate_slots: int
    action_heads: int

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            _exact_positive_int(getattr(self, field.name), field=field.name)


@dataclasses.dataclass(frozen=True, slots=True)
class FutureUtilityEndpointGeometry:
    """Schedule geometry needed for recurrence and cadence endpoints."""

    phase_order: tuple[str, ...]
    phase_lengths: tuple[int, ...]
    target_names: tuple[str, ...]
    curation_interval: int

    def __post_init__(self) -> None:
        if (
            type(self.phase_order) is not tuple
            or not self.phase_order
            or any(type(name) is not str or not name for name in self.phase_order)
        ):
            raise ValueError("phase_order must be a non-empty exact string tuple")
        if (
            type(self.phase_lengths) is not tuple
            or len(self.phase_lengths) != len(self.phase_order)
        ):
            raise ValueError("phase_lengths must align exactly with phase_order")
        for length in self.phase_lengths:
            _exact_positive_int(length, field="phase length")
        if len(set(self.phase_lengths)) != len(self.phase_lengths):
            raise ValueError("phase lengths must be unique")
        if (
            type(self.target_names) is not tuple
            or not self.target_names
            or len(set(self.target_names)) != len(self.target_names)
            or any(name not in control.SIGNATURE_NAMES for name in self.target_names)
        ):
            raise ValueError("target_names must be unique known signatures")
        if any(self.phase_order.count(name) < 2 for name in self.target_names):
            raise ValueError("every target must have at least one recurrence")
        _exact_positive_int(self.curation_interval, field="curation_interval")

    @property
    def total_steps(self) -> int:
        return sum(self.phase_lengths)

    @property
    def phase_boundaries(self) -> tuple[int, ...]:
        boundaries = [0]
        for length in self.phase_lengths:
            boundaries.append(boundaries[-1] + length)
        return tuple(boundaries)


def build_future_utility_learner_config(
    historical_base: Mapping[str, Any],
    arm: FutureUtilityArmSpec,
) -> dict[str, Any]:
    """Apply one exact mechanism arm to an externally selected common base."""

    if not isinstance(historical_base, Mapping) or not historical_base:
        raise TypeError("historical_base must be a non-empty mapping")
    if type(arm) is not FutureUtilityArmSpec:
        raise TypeError("arm must be an exact FutureUtilityArmSpec")
    config = dict(historical_base)
    required = {
        "candidate_scoring_mode",
        "candidate_novelty_admission_bonus",
        "future_utility_trace_mode",
        "future_utility_mix",
        "future_utility_trace_decay",
        "future_utility_normalization",
        "future_utility_normalization_decay",
        "future_utility_rare_task_power",
    }
    if not required <= set(config):
        raise ValueError("historical_base lacks required future-utility fields")
    config.update(
        {
            "candidate_scoring_mode": "legacy",
            "candidate_novelty_admission_bonus": 0.0,
            "future_utility_trace_mode": "contribution",
            "future_utility_mix": arm.mix,
            "future_utility_trace_decay": arm.trace_decay,
            "future_utility_normalization": arm.normalization,
            "future_utility_normalization_decay": NORMALIZATION_DECAY,
            "future_utility_rare_task_power": 0.0,
        }
    )
    return config


def arm_definition(arm: FutureUtilityArmSpec) -> dict[str, object]:
    """Return the canonical mechanism record for one arm."""

    if type(arm) is not FutureUtilityArmSpec:
        raise TypeError("arm must be an exact FutureUtilityArmSpec")
    return {
        "name": arm.name,
        "role": arm.role,
        "future_utility_mix": arm.mix,
        "future_utility_trace_decay": arm.trace_decay,
        "future_utility_trace_decay_f32_bits": struct.pack(">f", arm.trace_decay).hex(),
        "future_utility_trace_mode": "contribution",
        "future_utility_normalization": arm.normalization,
        "future_utility_normalization_decay": NORMALIZATION_DECAY,
        "future_utility_rare_task_power": 0.0,
        "candidate_scoring_mode": "legacy",
        "candidate_novelty_admission_bonus": 0.0,
    }


def _f32_uint_bits(value: float) -> int:
    return int(struct.unpack(">I", struct.pack(">f", value))[0])


def _bank_descending_rank(
    mask: object,
    scores: object,
    *,
    start: int,
    stop: int,
    matching_slots_field: str,
) -> dict[str, object]:
    """Return one tie-aware binary32 rank over an exact bank slice."""

    selected_all = np.asarray(mask)
    values_all = np.asarray(scores)
    if selected_all.dtype != np.dtype(np.bool_):
        raise TypeError("rank mask must have exact boolean dtype")
    if values_all.dtype != np.dtype(np.float32):
        raise TypeError("rank scores must have exact binary32 dtype")
    if (
        selected_all.ndim != 1
        or values_all.ndim != 1
        or selected_all.shape != values_all.shape
        or start < 0
        or stop <= start
        or stop > selected_all.shape[0]
    ):
        raise RuntimeError("rank mask, scores, or bank slice is invalid")
    if not np.all(np.isfinite(values_all)):
        raise RuntimeError("rank scores must be finite")
    selected = selected_all[start:stop]
    values = values_all[start:stop]
    matching_slots = [start + int(index) for index in np.flatnonzero(selected)]
    if not np.any(selected):
        return {
            "present": False,
            matching_slots_field: matching_slots,
            "matching_score_f32_bits": [],
            "best_score_f32_bits": None,
            "descending_rank_interval": None,
        }
    best_value = np.max(values[selected])
    strictly_greater = int(np.count_nonzero(values > best_value))
    equal = int(np.count_nonzero(values == best_value))
    return {
        "present": True,
        matching_slots_field: matching_slots,
        "matching_score_f32_bits": [
            _f32_uint_bits(float(value)) for value in values[selected]
        ],
        "best_score_f32_bits": _f32_uint_bits(float(best_value)),
        "descending_rank_interval": [
            1 + strictly_greater,
            strictly_greater + equal,
        ],
    }


def active_bank_descending_rank(mask: object, scores: object) -> dict[str, object]:
    """Rank matching signatures among composed active slots only."""

    return _bank_descending_rank(
        mask,
        scores,
        start=control.RAW_DIM,
        stop=control.ACTIVE_SLOTS,
        matching_slots_field="matching_composed_slots",
    )


def candidate_bank_descending_rank(
    mask: object,
    scores: object,
) -> dict[str, object]:
    """Rank matching signatures over the complete candidate bank."""

    return _bank_descending_rank(
        mask,
        scores,
        start=0,
        stop=control.CANDIDATE_SLOTS,
        matching_slots_field="matching_candidate_slots",
    )


def pre_recurrence_records(
    geometry: FutureUtilityEndpointGeometry,
    events: object,
) -> list[dict[str, object]]:
    """Project exact pre-recurrence presence and ranks from caller-bound geometry."""

    if type(geometry) is not FutureUtilityEndpointGeometry:
        raise TypeError("geometry must be an exact FutureUtilityEndpointGeometry")
    scan = cast(Any, events)
    post_slots = np.asarray(scan.post_active_signature_slots)
    post_candidate_slots = np.asarray(scan.post_candidate_signature_slots)
    direct = np.asarray(scan.direct_active_scores)
    backed = np.asarray(scan.backed_active_scores)
    candidate_direct = np.asarray(scan.direct_candidate_scores)
    candidate_augmented = np.asarray(scan.augmented_candidate_scores)
    if (
        post_slots.dtype != np.dtype(np.bool_)
        or post_candidate_slots.dtype != np.dtype(np.bool_)
    ):
        raise TypeError("pre-recurrence signature masks must have exact boolean dtype")
    if any(
        array.dtype != np.dtype(np.float32)
        for array in (direct, backed, candidate_direct, candidate_augmented)
    ):
        raise TypeError("pre-recurrence rank scores must have exact binary32 dtype")
    expected_shapes = (
        (
            post_slots,
            (
                geometry.total_steps,
                control.ACTIVE_SLOTS,
                len(control.SIGNATURE_NAMES),
            ),
        ),
        (
            post_candidate_slots,
            (
                geometry.total_steps,
                control.CANDIDATE_SLOTS,
                len(control.SIGNATURE_NAMES),
            ),
        ),
        (direct, (geometry.total_steps, control.ACTIVE_SLOTS)),
        (backed, (geometry.total_steps, control.ACTIVE_SLOTS)),
        (candidate_direct, (geometry.total_steps, control.CANDIDATE_SLOTS)),
        (candidate_augmented, (geometry.total_steps, control.CANDIDATE_SLOTS)),
    )
    if any(array.shape != shape for array, shape in expected_shapes):
        raise RuntimeError("pre-recurrence telemetry shapes do not match geometry")
    if not all(
        np.all(np.isfinite(array))
        for array in (
            direct,
            backed,
            candidate_direct,
            candidate_augmented,
        )
    ):
        raise RuntimeError("pre-recurrence rank telemetry is not finite")
    if np.any(post_slots[:, : control.RAW_DIM, :]):
        raise RuntimeError("reserved raw active slots cannot match product signatures")
    if np.any(np.sum(post_slots, axis=2, dtype=np.int32) > 1) or np.any(
        np.sum(post_candidate_slots, axis=2, dtype=np.int32) > 1
    ):
        raise RuntimeError("one feature slot cannot match multiple distinct signatures")

    seen: dict[str, int] = {}
    records: list[dict[str, object]] = []
    for phase_index, (name, start) in enumerate(
        zip(
            geometry.phase_order,
            geometry.phase_boundaries[:-1],
            strict=True,
        )
    ):
        occurrence = seen.get(name, 0) + 1
        seen[name] = occurrence
        if name not in geometry.target_names or occurrence == 1:
            continue
        event_index = start - 1
        signature_index = control.SIGNATURE_NAMES.index(name)
        signature_mask = post_slots[event_index, :, signature_index]
        candidate_mask = post_candidate_slots[event_index, :, signature_index]
        active_present = bool(np.any(signature_mask))
        candidate_present = bool(np.any(candidate_mask))
        direct_rank = active_bank_descending_rank(
            signature_mask,
            direct[event_index],
        )
        backed_rank = active_bank_descending_rank(
            signature_mask,
            backed[event_index],
        )
        candidate_direct_rank = candidate_bank_descending_rank(
            candidate_mask,
            candidate_direct[event_index],
        )
        candidate_augmented_rank = candidate_bank_descending_rank(
            candidate_mask,
            candidate_augmented[event_index],
        )
        if (
            direct_rank.get("present") is not active_present
            or backed_rank.get("present") is not active_present
            or candidate_direct_rank.get("present") is not candidate_present
            or candidate_augmented_rank.get("present") is not candidate_present
        ):
            raise RuntimeError(
                "pre-recurrence rank presence does not match structural presence"
            )
        records.append(
            {
                "target": name,
                "occurrence": occurrence,
                "recurrence_phase_index": phase_index,
                "pre_recurrence_post_step": start,
                "active_present": active_present,
                "candidate_present": candidate_present,
                "active_slot_count": int(np.count_nonzero(signature_mask)),
                "candidate_slot_count": int(np.count_nonzero(candidate_mask)),
                "matching_active_slots": [
                    int(slot) for slot in np.flatnonzero(signature_mask)
                ],
                "matching_candidate_slots": [
                    int(slot) for slot in np.flatnonzero(candidate_mask)
                ],
                "direct_rank": direct_rank,
                "ancestor_backed_rank": backed_rank,
                "candidate_direct_rank": candidate_direct_rank,
                "candidate_augmented_rank": candidate_augmented_rank,
            }
        )
    return records


def validate_future_utility_trace_shapes(
    geometry: FutureUtilityEndpointGeometry,
    events: object,
) -> dict[str, list[int]]:
    """Require the exact diagnostic and mutation trailing shapes."""

    if type(geometry) is not FutureUtilityEndpointGeometry:
        raise TypeError("geometry must be an exact FutureUtilityEndpointGeometry")
    trace = cast(Any, events).curation_trace
    steps = geometry.total_steps
    scalar_names = (
        "decision_margin_passed",
        "decision_should_promote",
        "decision_should_refresh",
        "proposal_formed",
        "has_event",
        "promotion_applied",
        "root_change_applied",
    )
    active_names = (
        "root_change_mask",
        "cascade_refill_mask",
        "active_change_mask",
    )
    candidate_names = (
        "ordinary_candidate_refresh_mask",
        "post_promotion_candidate_refresh_mask",
        "candidate_refresh_mask",
        "candidate_rebound_mask",
        "candidate_overdepth_regeneration_mask",
    )
    expected_shapes = {
        **{name: (steps,) for name in scalar_names},
        "decision_candidate_margin_eligible": (
            steps,
            control.CANDIDATE_SLOTS,
            control.ACTIVE_SLOTS,
        ),
        **{name: (steps, control.ACTIVE_SLOTS) for name in active_names},
        **{name: (steps, control.CANDIDATE_SLOTS) for name in candidate_names},
    }
    observed_shapes: dict[str, list[int]] = {}
    for name, expected_shape in expected_shapes.items():
        value = np.asarray(getattr(trace, name))
        if value.dtype != np.dtype(np.bool_):
            raise TypeError(f"future-utility trace field {name} must be boolean")
        if value.shape != expected_shape:
            raise RuntimeError(
                f"future-utility trace field {name} has shape {value.shape}, "
                f"expected {expected_shape}"
            )
        observed_shapes[name] = list(value.shape)
    return observed_shapes


def validate_future_utility_experience_semantics(
    geometry: FutureUtilityEndpointGeometry,
    events: object,
    *,
    observations: object,
    phase_indices: object,
    exploration_mask: object,
    random_actions: object,
    phase_target_raw_indices: Sequence[Sequence[int]],
    action_reward_multipliers: Sequence[float],
    composed_readout_enabled: bool,
) -> dict[str, object]:
    """Validate exact observation, action, target, and reward trace semantics."""

    if type(geometry) is not FutureUtilityEndpointGeometry:
        raise TypeError("geometry must be an exact FutureUtilityEndpointGeometry")
    if type(composed_readout_enabled) is not bool:
        raise TypeError("composed_readout_enabled must be an exact bool")
    targets = tuple(tuple(indices) for indices in phase_target_raw_indices)
    if (
        len(targets) != len(geometry.phase_order)
        or any(
            not indices
            or any(type(index) is not int or not 0 <= index < control.RAW_DIM for index in indices)
            or len(set(indices)) != len(indices)
            for indices in targets
        )
    ):
        raise ValueError("phase target raw indices are invalid")
    multipliers = tuple(action_reward_multipliers)
    if multipliers != (-1.0, 1.0) or any(type(value) is not float for value in multipliers):
        raise ValueError("action reward multipliers must be the exact two-action mapping")

    steps = geometry.total_steps

    def exact_array(
        value: object,
        *,
        name: str,
        shape: tuple[int, ...],
        dtype: np.dtype[Any],
    ) -> np.ndarray[Any, Any]:
        array = np.asarray(value)
        if array.shape != shape or array.dtype != dtype:
            raise RuntimeError(
                f"experience field {name} has shape/dtype "
                f"{array.shape}/{array.dtype}, expected {shape}/{dtype}"
            )
        return array

    observation_values = exact_array(
        observations,
        name="observations",
        shape=(steps, control.RAW_DIM),
        dtype=np.dtype(np.float32),
    )
    phase_values = exact_array(
        phase_indices,
        name="phase_indices",
        shape=(steps,),
        dtype=np.dtype(np.int32),
    )
    exploration_values = exact_array(
        exploration_mask,
        name="exploration_mask",
        shape=(steps,),
        dtype=np.dtype(np.bool_),
    )
    random_action_values = exact_array(
        random_actions,
        name="random_actions",
        shape=(steps,),
        dtype=np.dtype(np.int32),
    )
    if (
        not np.all(np.isfinite(observation_values))
        or not np.all(np.isin(observation_values, (-1.0, 1.0)))
        or not np.all(np.isin(random_action_values, (0, 1)))
    ):
        raise RuntimeError("source observation or random-action domain is invalid")
    expected_phase_values = np.repeat(
        np.arange(len(geometry.phase_order), dtype=np.int32),
        np.asarray(geometry.phase_lengths, dtype=np.int32),
    )
    if not np.array_equal(phase_values, expected_phase_values):
        raise RuntimeError("source phase indices do not match endpoint geometry")

    scan = cast(Any, events)
    action = exact_array(
        scan.action,
        name="action",
        shape=(steps,),
        dtype=np.dtype(np.int32),
    )
    greedy_action = exact_array(
        scan.greedy_action,
        name="greedy_action",
        shape=(steps,),
        dtype=np.dtype(np.int32),
    )
    explored = exact_array(
        scan.explored,
        name="explored",
        shape=(steps,),
        dtype=np.dtype(np.bool_),
    )
    target_value = exact_array(
        scan.target_value,
        name="target_value",
        shape=(steps,),
        dtype=np.dtype(np.float32),
    )
    executed_reward = exact_array(
        scan.executed_reward,
        name="executed_reward",
        shape=(steps,),
        dtype=np.dtype(np.float32),
    )
    greedy_reward = exact_array(
        scan.greedy_reward,
        name="greedy_reward",
        shape=(steps,),
        dtype=np.dtype(np.float32),
    )
    executed_regret = exact_array(
        scan.executed_regret,
        name="executed_regret",
        shape=(steps,),
        dtype=np.dtype(np.float32),
    )
    greedy_regret = exact_array(
        scan.greedy_regret,
        name="greedy_regret",
        shape=(steps,),
        dtype=np.dtype(np.float32),
    )
    q_shape = (steps, control.ACTION_HEADS)
    full_q = exact_array(
        scan.full_q,
        name="full_q",
        shape=q_shape,
        dtype=np.dtype(np.float32),
    )
    raw_q = exact_array(
        scan.raw_q,
        name="raw_q",
        shape=q_shape,
        dtype=np.dtype(np.float32),
    )
    behavior_q = exact_array(
        scan.behavior_q,
        name="behavior_q",
        shape=q_shape,
        dtype=np.dtype(np.float32),
    )
    if not all(
        np.all(np.isfinite(array))
        for array in (
            target_value,
            executed_reward,
            greedy_reward,
            executed_regret,
            greedy_regret,
            full_q,
            raw_q,
            behavior_q,
        )
    ):
        raise RuntimeError("experience rewards or action values are not finite")

    expected_behavior_q = full_q if composed_readout_enabled else raw_q
    expected_greedy_action = np.argmax(expected_behavior_q, axis=1).astype(np.int32)
    expected_action = np.where(
        exploration_values,
        random_action_values,
        expected_greedy_action,
    ).astype(np.int32)
    expected_target = np.empty((steps,), dtype=np.float32)
    for phase, indices in enumerate(targets):
        mask = phase_values == phase
        expected_target[mask] = np.prod(
            observation_values[mask][:, indices],
            axis=1,
            dtype=np.float32,
        )
    multiplier_values = np.asarray(multipliers, dtype=np.float32)
    expected_executed_reward = multiplier_values[expected_action] * expected_target
    expected_greedy_reward = multiplier_values[expected_greedy_action] * expected_target
    if not np.array_equal(explored, exploration_values):
        raise RuntimeError("exploration trace does not match the pinned mask")
    if not np.array_equal(behavior_q, expected_behavior_q):
        raise RuntimeError("behavior action values do not match the readout rule")
    if not np.array_equal(greedy_action, expected_greedy_action):
        raise RuntimeError("greedy action trace does not match first-argmax")
    if not np.array_equal(action, expected_action):
        raise RuntimeError("executed action trace does not match exploration selection")
    if not np.array_equal(target_value, expected_target):
        raise RuntimeError("target trace does not match phase product semantics")
    if (
        not np.array_equal(executed_reward, expected_executed_reward)
        or not np.array_equal(greedy_reward, expected_greedy_reward)
    ):
        raise RuntimeError("reward trace does not match action-sign target semantics")
    if (
        not np.array_equal(
            executed_regret,
            np.float32(1.0) - expected_executed_reward,
        )
        or not np.array_equal(
            greedy_regret,
            np.float32(1.0) - expected_greedy_reward,
        )
    ):
        raise RuntimeError("regret trace does not match one-minus-reward semantics")
    return {
        "all_experience_semantics_match": True,
        "steps": steps,
        "composed_readout_enabled": composed_readout_enabled,
        "explored_step_count": int(np.count_nonzero(exploration_values)),
        "executed_action_counts": [
            int(value) for value in np.bincount(action, minlength=control.ACTION_HEADS)
        ],
        "greedy_action_counts": [
            int(value)
            for value in np.bincount(greedy_action, minlength=control.ACTION_HEADS)
        ],
    }


def validate_future_utility_eventwise_curation_semantics(
    geometry: FutureUtilityEndpointGeometry,
    events: object,
) -> dict[str, object]:
    """Require exact same-step relationships among decisions and mutations."""

    validate_future_utility_trace_shapes(geometry, events)
    trace = cast(Any, events).curation_trace

    def boolean(name: str) -> np.ndarray[Any, Any]:
        return np.asarray(getattr(trace, name), dtype=np.bool_)

    promotion = boolean("promotion_applied")
    should_promote = boolean("decision_should_promote")
    root_applied = boolean("root_change_applied")
    root_mask = boolean("root_change_mask")
    cascade_mask = boolean("cascade_refill_mask")
    active_change_mask = boolean("active_change_mask")
    should_refresh = boolean("decision_should_refresh")
    ordinary_refresh_mask = boolean("ordinary_candidate_refresh_mask")
    post_promotion_refresh_mask = boolean(
        "post_promotion_candidate_refresh_mask"
    )
    candidate_refresh_mask = boolean("candidate_refresh_mask")
    rebound_mask = boolean("candidate_rebound_mask")
    overdepth_mask = boolean("candidate_overdepth_regeneration_mask")
    proposal = boolean("proposal_formed")
    has_event = boolean("has_event")
    margin = boolean("decision_margin_passed")
    candidate_margin = boolean("decision_candidate_margin_eligible")
    selected_candidate = np.asarray(trace.decision_selected_candidate)
    selected_destination = np.asarray(trace.decision_selected_destination)
    promotion_source = np.asarray(trace.promotion_source_candidate)
    promotion_destination = np.asarray(trace.promotion_destination_active)
    expected_index_shape = (geometry.total_steps,)
    if (
        selected_candidate.shape != expected_index_shape
        or selected_destination.shape != expected_index_shape
        or promotion_source.shape != expected_index_shape
        or promotion_destination.shape != expected_index_shape
        or selected_candidate.dtype != np.dtype(np.int32)
        or selected_destination.dtype != np.dtype(np.int32)
        or promotion_source.dtype != np.dtype(np.int32)
        or promotion_destination.dtype != np.dtype(np.int32)
    ):
        raise RuntimeError("selected curation indices have an invalid shape or dtype")

    root_event = np.any(root_mask, axis=1)
    cascade_event = np.any(cascade_mask, axis=1)
    ordinary_refresh_event = np.any(ordinary_refresh_mask, axis=1)
    post_promotion_refresh_event = np.any(post_promotion_refresh_mask, axis=1)
    candidate_refresh_event = np.any(candidate_refresh_mask, axis=1)
    rebound_event = np.any(rebound_mask, axis=1)
    overdepth_event = np.any(overdepth_mask, axis=1)
    candidate_margin_event = np.any(candidate_margin, axis=(1, 2))
    expected_has_event = (
        root_event
        | cascade_event
        | candidate_refresh_event
        | rebound_event
        | overdepth_event
    )
    relationships = (
        ("promotion/should-promote", promotion, should_promote),
        ("promotion/root-applied", promotion, root_applied),
        ("promotion/root-mask", promotion, root_event),
        (
            "promotion/post-promotion-refresh",
            promotion,
            post_promotion_refresh_event,
        ),
        ("ordinary-refresh decision/mask", should_refresh, ordinary_refresh_event),
        (
            "candidate-refresh union",
            candidate_refresh_mask,
            ordinary_refresh_mask | post_promotion_refresh_mask,
        ),
        ("active-change union", active_change_mask, root_mask | cascade_mask),
        ("proposal decision", proposal, promotion | should_refresh),
        ("event identity", has_event, expected_has_event),
    )
    for label, observed, expected in relationships:
        if not np.array_equal(observed, expected):
            raise RuntimeError(f"same-step curation relationship failed: {label}")
    if np.any(promotion & ~margin) or np.any(promotion & ~candidate_margin_event):
        raise RuntimeError("same-step promotion lacks its strict-margin diagnostics")
    promotion_rows = np.flatnonzero(promotion)
    if promotion_rows.size:
        promotion_candidates = selected_candidate[promotion_rows]
        promotion_destinations = selected_destination[promotion_rows]
        if (
            np.any(promotion_candidates < 0)
            or np.any(promotion_candidates >= control.CANDIDATE_SLOTS)
            or np.any(promotion_destinations < 0)
            or np.any(promotion_destinations >= control.ACTIVE_SLOTS)
            or not np.all(
                candidate_margin[
                    promotion_rows,
                    promotion_candidates,
                    promotion_destinations,
                ]
            )
            or not np.all(
                root_mask[promotion_rows, promotion_destinations]
            )
            or not np.all(
                post_promotion_refresh_mask[
                    promotion_rows,
                    promotion_candidates,
                ]
            )
            or not np.array_equal(
                promotion_source[promotion_rows],
                promotion_candidates,
            )
            or not np.array_equal(
                promotion_destination[promotion_rows],
                promotion_destinations,
            )
        ):
            raise RuntimeError(
                "promotion does not bind its selected candidate/destination cell"
            )

    curation_counts = np.asarray(cast(Any, events).curation_counts)
    expected_count_shape = (geometry.total_steps, len(control.CURATION_COUNT_NAMES))
    if (
        curation_counts.shape != expected_count_shape
        or curation_counts.dtype != np.dtype(np.int32)
        or np.any(curation_counts < 0)
    ):
        raise RuntimeError("per-step curation counts have an invalid shape or dtype")
    expected_curation_counts = np.stack(
        (
            boolean("should_try_replace").astype(np.int64),
            proposal.astype(np.int64),
            np.sum(root_mask, axis=1, dtype=np.int64),
            promotion.astype(np.int64),
            np.sum(cascade_mask, axis=1, dtype=np.int64),
            np.sum(ordinary_refresh_mask, axis=1, dtype=np.int64),
            np.sum(post_promotion_refresh_mask, axis=1, dtype=np.int64),
            np.sum(candidate_refresh_mask, axis=1, dtype=np.int64),
            np.sum(rebound_mask, axis=1, dtype=np.int64),
            np.sum(overdepth_mask, axis=1, dtype=np.int64),
            (
                np.sum(root_mask, axis=1, dtype=np.int64)
                + np.sum(candidate_refresh_mask, axis=1, dtype=np.int64)
                + np.sum(cascade_mask, axis=1, dtype=np.int64)
                + np.sum(rebound_mask, axis=1, dtype=np.int64)
                + np.sum(overdepth_mask, axis=1, dtype=np.int64)
            ),
        ),
        axis=1,
    )
    if not np.array_equal(curation_counts, expected_curation_counts):
        raise RuntimeError("per-step curation counts do not match mutation telemetry")
    trace_count_fields = (
        "proposal_count",
        "root_change_count",
        "promotion_count",
        "cascade_refill_count",
        "ordinary_candidate_refresh_count",
        "post_promotion_candidate_refresh_count",
        "candidate_refresh_count",
        "candidate_rebound_count",
        "candidate_overdepth_regeneration_count",
        "logical_event_count",
    )
    if not np.array_equal(
        curation_counts[:, 0],
        boolean("should_try_replace").astype(np.int32),
    ):
        raise RuntimeError("curation-due count column does not match the production trace")
    for column, field in enumerate(trace_count_fields, start=1):
        values = np.asarray(getattr(trace, field))
        if values.shape != expected_index_shape or values.dtype != np.dtype(np.int32):
            raise RuntimeError(f"production trace count {field} has an invalid shape or dtype")
        if not np.array_equal(values, curation_counts[:, column]):
            raise RuntimeError(
                f"production trace count {field} disagrees with curation_counts"
            )
    return {
        "all_eventwise_curation_semantics_match": True,
        "promotion_event_count": int(np.count_nonzero(promotion)),
        "ordinary_refresh_event_count": int(np.count_nonzero(should_refresh)),
        "event_bearing_opportunity_count": int(np.count_nonzero(has_event)),
    }


def future_utility_cadence_audit_from_events(
    geometry: FutureUtilityEndpointGeometry,
    events: object,
    *,
    pinned_due_mask: object,
) -> cadence.FutureUtilityCadenceAudit:
    """Bind public production-trace fields to the neutral cadence audit."""

    if type(geometry) is not FutureUtilityEndpointGeometry:
        raise TypeError("geometry must be an exact FutureUtilityEndpointGeometry")
    validate_future_utility_trace_shapes(geometry, events)
    scan = cast(Any, events)
    trace = scan.curation_trace
    mutation_masks = {
        name: getattr(trace, name) for name in REQUIRED_CADENCE_MUTATION_MASK_NAMES
    }
    domain = cadence.build_fixed_curation_opportunity_domain(
        post_step=trace.post_step,
        decision_update_available=trace.decision_update_available,
        pre_replacement_phase=trace.pre_replacement_phase,
        post_replacement_phase=trace.post_replacement_phase,
        should_try_replace=trace.should_try_replace,
        pinned_due_mask=pinned_due_mask,
        replacement_interval=geometry.curation_interval,
    )
    if domain.steps != geometry.total_steps:
        raise RuntimeError("cadence domain does not match endpoint geometry")
    return cadence.build_future_utility_cadence_audit(
        domain,
        decision_margin_passed=trace.decision_margin_passed,
        decision_candidate_margin_eligible=(
            trace.decision_candidate_margin_eligible
        ),
        mutation_masks=mutation_masks,
    )


def validate_future_utility_curation_count_closure(
    audit: cadence.FutureUtilityCadenceAudit,
    curation_totals: Mapping[str, int],
) -> dict[str, object]:
    """Require production count totals to close against exact mutation masks."""

    if type(audit) is not cadence.FutureUtilityCadenceAudit:
        raise TypeError("audit must be an exact FutureUtilityCadenceAudit")
    if not isinstance(curation_totals, Mapping):
        raise TypeError("curation_totals must be a mapping")
    if set(curation_totals) != set(control.CURATION_COUNT_NAMES):
        raise ValueError("curation_totals must declare the exact production count set")
    if any(type(value) is not int or value < 0 for value in curation_totals.values()):
        raise ValueError("curation totals must be nonnegative exact integers")
    partitions = audit.mutation_partitions
    expected_by_total = {
        "proposal": "proposal_formed",
        "root_change": "root_change_mask",
        "promotion": "promotion_applied",
        "cascade_refill": "cascade_refill_mask",
        "ordinary_candidate_refresh": "ordinary_candidate_refresh_mask",
        "post_promotion_candidate_refresh": (
            "post_promotion_candidate_refresh_mask"
        ),
        "candidate_refresh": "candidate_refresh_mask",
        "candidate_rebound": "candidate_rebound_mask",
        "candidate_overdepth_regeneration": (
            "candidate_overdepth_regeneration_mask"
        ),
    }
    observed = {
        total_name: partitions[mask_name].due_opportunity_count
        for total_name, mask_name in expected_by_total.items()
    }
    expected = {name: curation_totals[name] for name in expected_by_total}
    if observed != expected:
        raise RuntimeError("curation mutation masks and production totals disagree")
    if curation_totals["curation_due"] != audit.due_opportunity_count:
        raise RuntimeError("curation due count does not match the cadence domain")
    if (
        partitions["root_change_applied"].due_opportunity_count
        != curation_totals["root_change"]
        or partitions["decision_should_promote"].due_opportunity_count
        != curation_totals["promotion"]
        or partitions["decision_should_refresh"].due_opportunity_count
        != curation_totals["ordinary_candidate_refresh"]
        or partitions["active_change_mask"].due_opportunity_count
        != curation_totals["root_change"] + curation_totals["cascade_refill"]
        or curation_totals["candidate_refresh"]
        != curation_totals["ordinary_candidate_refresh"]
        + curation_totals["post_promotion_candidate_refresh"]
    ):
        raise RuntimeError("derived curation event counts do not close")
    if curation_totals["proposal"] != (
        curation_totals["promotion"]
        + curation_totals["ordinary_candidate_refresh"]
    ):
        raise RuntimeError("proposal count does not equal promotion plus ordinary refresh")
    if curation_totals["post_promotion_candidate_refresh"] != curation_totals[
        "promotion"
    ]:
        raise RuntimeError("post-promotion refresh count does not equal promotion count")
    reconstructed_logical_events = (
        curation_totals["root_change"]
        + curation_totals["candidate_refresh"]
        + curation_totals["cascade_refill"]
        + curation_totals["candidate_rebound"]
        + curation_totals["candidate_overdepth_regeneration"]
    )
    if reconstructed_logical_events != curation_totals["logical_event"]:
        raise RuntimeError("logical event count does not reconstruct")
    if partitions["has_event"].due_opportunity_count > reconstructed_logical_events:
        raise RuntimeError("event-bearing opportunities exceed logical events")
    return {
        "all_checked_counts_close": True,
        "curation_due_count": audit.due_opportunity_count,
        "mutation_counts": observed,
        "logical_event_count": reconstructed_logical_events,
        "event_bearing_opportunity_count": partitions[
            "has_event"
        ].due_opportunity_count,
    }


def _opportunity_partition_record(
    partition: cadence.OpportunityPartition,
) -> dict[str, int]:
    if type(partition) is not cadence.OpportunityPartition:
        raise TypeError("partition must be an exact OpportunityPartition")
    return {
        "all_step_count": partition.all_step_count,
        "due_opportunity_count": partition.due_opportunity_count,
        "off_opportunity_count": partition.off_opportunity_count,
    }


def _target_coexistence_record(
    geometry: FutureUtilityEndpointGeometry,
    active_signature_counts: np.ndarray[Any, Any],
) -> dict[str, object]:
    target_indices = tuple(
        control.SIGNATURE_NAMES.index(name) for name in geometry.target_names
    )
    present = active_signature_counts[:, target_indices] > 0
    active_count = np.sum(present, axis=1, dtype=np.int64)
    histogram = np.bincount(active_count, minlength=len(target_indices) + 1)
    all_targets = active_count == len(target_indices)
    all_target_indices = np.flatnonzero(all_targets)
    return {
        "target_order": list(geometry.target_names),
        "steps": geometry.total_steps,
        "steps_by_active_target_count": [int(value) for value in histogram],
        "maximum_active_target_count": int(np.max(active_count, initial=0)),
        "all_targets_present_steps": int(np.count_nonzero(all_targets)),
        "all_targets_presence_fraction": float(
            np.mean(all_targets, dtype=np.float64)
        ),
        "first_all_targets_post_step": (
            None
            if all_target_indices.size == 0
            else int(all_target_indices[0]) + 1
        ),
        "last_all_targets_post_step": (
            None
            if all_target_indices.size == 0
            else int(all_target_indices[-1]) + 1
        ),
        "active_targets_at_end": [
            name
            for name, flag in zip(geometry.target_names, present[-1], strict=True)
            if flag
        ],
    }


def build_future_utility_primary_endpoints(
    geometry: FutureUtilityEndpointGeometry,
    events: object,
    *,
    active_trajectories: Mapping[str, Mapping[str, object]],
    curation_totals: Mapping[str, int],
    curation_audit: Mapping[str, object],
    pinned_due_mask: object,
) -> dict[str, object]:
    """Project cadence-safe structural endpoints from one completed arm scan.

    Raw margin diagnostics are explicitly partitioned into due and off-cadence
    cells.  In contrast, every state-changing mask must be false off cadence,
    and its due-cell count must close against the production curation totals.
    """

    if type(geometry) is not FutureUtilityEndpointGeometry:
        raise TypeError("geometry must be an exact FutureUtilityEndpointGeometry")
    if not isinstance(active_trajectories, Mapping) or set(active_trajectories) != set(
        geometry.target_names
    ):
        raise ValueError("active trajectories must declare the exact target set")
    if not isinstance(curation_audit, Mapping):
        raise TypeError("curation_audit must be a mapping")

    cadence_audit = future_utility_cadence_audit_from_events(
        geometry,
        events,
        pinned_due_mask=pinned_due_mask,
    )
    count_closure = validate_future_utility_curation_count_closure(
        cadence_audit,
        curation_totals,
    )
    eventwise_closure = validate_future_utility_eventwise_curation_semantics(
        geometry,
        events,
    )
    due_audit_count = curation_audit.get("due_curation_event_count")
    if type(due_audit_count) is not int or due_audit_count != (
        cadence_audit.due_opportunity_count
    ):
        raise RuntimeError("curation audit and cadence-domain due counts disagree")

    scan = cast(Any, events)
    active_counts_raw = np.asarray(scan.active_signature_counts)
    candidate_counts_raw = np.asarray(scan.candidate_signature_counts)
    expected_count_shape = (geometry.total_steps, len(control.SIGNATURE_NAMES))
    if (
        active_counts_raw.shape != expected_count_shape
        or candidate_counts_raw.shape != expected_count_shape
        or active_counts_raw.dtype != np.dtype(np.int32)
        or candidate_counts_raw.dtype != np.dtype(np.int32)
    ):
        raise TypeError("signature-count telemetry must use exact int32 arrays")
    active_counts = active_counts_raw.astype(np.int64, copy=False)
    candidate_counts = candidate_counts_raw.astype(np.int64, copy=False)
    if (
        np.any(active_counts < 0)
        or np.any(candidate_counts < 0)
        or np.any(active_counts > control.ACTIVE_SLOTS)
        or np.any(candidate_counts > control.CANDIDATE_SLOTS)
    ):
        raise RuntimeError("signature-count telemetry is invalid")
    post_active_slots = np.asarray(scan.post_active_signature_slots)
    post_candidate_slots = np.asarray(scan.post_candidate_signature_slots)
    if (
        post_active_slots.dtype != np.dtype(np.bool_)
        or post_candidate_slots.dtype != np.dtype(np.bool_)
        or post_active_slots.shape
        != (
            geometry.total_steps,
            control.ACTIVE_SLOTS,
            len(control.SIGNATURE_NAMES),
        )
        or post_candidate_slots.shape
        != (
            geometry.total_steps,
            control.CANDIDATE_SLOTS,
            len(control.SIGNATURE_NAMES),
        )
    ):
        raise RuntimeError("post-update signature-slot telemetry has an invalid shape")
    reconstructed_active_counts = np.sum(post_active_slots, axis=1, dtype=np.int64)
    reconstructed_candidate_counts = np.sum(
        post_candidate_slots,
        axis=1,
        dtype=np.int64,
    )
    if not np.array_equal(
        active_counts,
        reconstructed_active_counts,
    ) or not np.array_equal(candidate_counts, reconstructed_candidate_counts):
        raise RuntimeError("signature counts do not match signature-slot telemetry")

    transitions_value = curation_audit.get("active_signature_transition_causes")
    outcomes_value = curation_audit.get("target_outcome_counts")
    if not isinstance(transitions_value, Mapping) or not isinstance(
        outcomes_value,
        Mapping,
    ):
        raise RuntimeError("curation audit lacks target transitions or outcomes")
    transitions = cast(Mapping[str, Mapping[str, object]], transitions_value)
    outcome_counts = cast(Mapping[str, Mapping[str, int]], outcomes_value)
    if not set(geometry.target_names) <= set(transitions) or not set(
        geometry.target_names
    ) <= set(outcome_counts):
        raise RuntimeError("curation audit does not cover every target")
    if curation_audit.get("all_target_due_events_accounted") is not True:
        raise RuntimeError("curation audit does not account for every target due event")

    cascade_losses: dict[str, object] = {}
    target_lifecycle: dict[str, object] = {}
    occupancy: dict[str, object] = {}
    for name in geometry.target_names:
        transition = transitions[name]
        loss_causes_value = transition.get("loss_slot_cause_counts")
        if not isinstance(loss_causes_value, Mapping):
            raise RuntimeError(f"target {name} lacks loss-cause counts")
        loss_causes = cast(Mapping[str, object], loss_causes_value)
        required_causes = (
            "promotion_root_replacement",
            "cascade_dependency_refill",
            "unmarked_signature_dependency_change",
        )
        if set(loss_causes) != set(required_causes) or any(
            type(loss_causes.get(cause)) is not int
            or cast(int, loss_causes[cause]) < 0
            for cause in required_causes
        ):
            raise RuntimeError(f"target {name} has invalid loss-cause counts")
        acquisition_episode_count = transition.get("acquisition_episode_count")
        loss_episode_count = transition.get("loss_episode_count")
        if (
            type(acquisition_episode_count) is not int
            or acquisition_episode_count < 0
            or type(loss_episode_count) is not int
            or loss_episode_count < 0
        ):
            raise RuntimeError(f"target {name} has invalid transition episode counts")
        if type(transition.get("all_changed_slots_accounted")) is not bool:
            raise RuntimeError(f"target {name} has an invalid transition closure")
        if transition["all_changed_slots_accounted"] is not True:
            raise RuntimeError(f"target {name} has an unclosed transition cause")
        cascade_losses[name] = {
            "loss_episode_count": loss_episode_count,
            "root_replacement_lost_slot_count": loss_causes[
                "promotion_root_replacement"
            ],
            "cascade_dependency_refill_lost_slot_count": loss_causes[
                "cascade_dependency_refill"
            ],
            "all_changed_slots_accounted": transition[
                "all_changed_slots_accounted"
            ],
        }

        trajectory = active_trajectories[name]
        signature_index = control.SIGNATURE_NAMES.index(name)
        reconstructed_trajectory = control._structural_trajectory(
            0,
            active_counts[:, signature_index],
        )
        if dict(trajectory) != reconstructed_trajectory:
            raise RuntimeError(
                f"target {name} structural lifecycle trajectory does not "
                "reconstruct from counts"
            )
        required_trajectory_fields = {
            "initially_present": bool,
            "acquisition_episode_count": int,
            "loss_episode_count": int,
            "present_at_end": bool,
            "structural_reacquisition_count": int,
        }
        if any(
            type(trajectory.get(field)) is not expected_type
            for field, expected_type in required_trajectory_fields.items()
        ):
            raise RuntimeError(f"target {name} has an invalid structural trajectory")
        acquisition_episodes = cast(int, trajectory["acquisition_episode_count"])
        loss_episodes = cast(int, trajectory["loss_episode_count"])
        present_at_end = cast(bool, trajectory["present_at_end"])
        structural_reacquisitions = cast(
            int,
            trajectory["structural_reacquisition_count"],
        )
        if (
            trajectory["initially_present"] is not False
            or acquisition_episodes < 0
            or loss_episodes < 0
            or acquisition_episodes - loss_episodes != int(present_at_end)
            or structural_reacquisitions != max(0, acquisition_episodes - 1)
            or acquisition_episode_count != acquisition_episodes
            or loss_episode_count != loss_episodes
            or loss_causes["unmarked_signature_dependency_change"] != 0
        ):
            raise RuntimeError(f"target {name} structural lifecycle does not close")
        target_outcomes = outcome_counts[name]
        if (
            not isinstance(target_outcomes, Mapping)
            or not target_outcomes
            or any(
                type(value) is not int or value < 0
                for value in target_outcomes.values()
            )
            or sum(target_outcomes.values()) != cadence_audit.due_opportunity_count
        ):
            raise RuntimeError(f"target {name} outcome counts do not close")
        admitted = target_outcomes.get("admitted", 0)
        if type(admitted) is not int or admitted < 0:
            raise RuntimeError(f"target {name} has an invalid admission count")
        target_lifecycle[name] = {
            "direct_candidate_admission_count": admitted,
            "admission_episode_count": trajectory["acquisition_episode_count"],
            "loss_episode_count": trajectory["loss_episode_count"],
            "present_at_end": trajectory["present_at_end"],
            "structural_reacquisition_count": trajectory[
                "structural_reacquisition_count"
            ],
        }

        active_values = active_counts[:, signature_index]
        candidate_values = candidate_counts[:, signature_index]
        occupancy[name] = {
            "active_present_post_steps": int(np.count_nonzero(active_values > 0)),
            "active_presence_fraction": float(
                np.mean(active_values > 0, dtype=np.float64)
            ),
            "active_slot_step_cells": int(np.sum(active_values, dtype=np.int64)),
            "candidate_present_post_steps": int(
                np.count_nonzero(candidate_values > 0)
            ),
            "candidate_presence_fraction": float(
                np.mean(candidate_values > 0, dtype=np.float64)
            ),
            "candidate_slot_step_cells": int(
                np.sum(candidate_values, dtype=np.int64)
            ),
        }

    diagnostic = cadence_audit.diagnostic_partitions
    mutations = cadence_audit.mutation_partitions
    margin = diagnostic["decision_margin_passed"]
    candidate_margin = diagnostic["decision_candidate_margin_eligible"]
    promotion_count = mutations["promotion_applied"].due_opportunity_count
    if not (
        promotion_count
        <= margin.due_opportunity_count
        <= candidate_margin.due_opportunity_count
    ):
        raise RuntimeError("promotion and due strict-margin diagnostics do not nest")
    pre_recurrence = pre_recurrence_records(geometry, events)
    coexistence = _target_coexistence_record(geometry, active_counts)
    target_retention = {
        name: {
            "pre_recurrence_phase_indices": [
                record["recurrence_phase_index"]
                for record in pre_recurrence
                if record["target"] == name
            ],
            "pre_recurrence_presence": [
                record["active_present"]
                for record in pre_recurrence
                if record["target"] == name
            ],
            "present_at_end": cast(Mapping[str, object], target_lifecycle[name])[
                "present_at_end"
            ],
        }
        for name in geometry.target_names
    }
    mutation_partition_records = {
        name: _opportunity_partition_record(mutations[name])
        for name in REQUIRED_CADENCE_MUTATION_MASK_NAMES
    }
    all_mutations_off_opportunity = sum(
        record["off_opportunity_count"]
        for record in mutation_partition_records.values()
    )
    if all_mutations_off_opportunity != 0:
        raise RuntimeError("cadence audit admitted an off-opportunity mutation")

    return {
        "endpoint_order": list(PRIMARY_ENDPOINT_NAMES),
        "margin_passes": {
            "selected_strict_margin_pass_count": margin.due_opportunity_count,
            "selected_strict_margin_all_step_diagnostic_count": margin.all_step_count,
            "selected_strict_margin_off_opportunity_diagnostic_count": (
                margin.off_opportunity_count
            ),
            "candidate_destination_strict_margin_pair_count": (
                candidate_margin.due_opportunity_count
            ),
            "candidate_destination_strict_margin_all_step_diagnostic_count": (
                candidate_margin.all_step_count
            ),
            "candidate_destination_strict_margin_off_opportunity_diagnostic_count": (
                candidate_margin.off_opportunity_count
            ),
            "due_curation_event_count": cadence_audit.due_opportunity_count,
        },
        "promotions": {
            "event_count": promotion_count
        },
        "cascade_refill_slot_count": curation_totals["cascade_refill"],
        "candidate_refreshes": {
            "decision_should_refresh_event_count": mutations[
                "decision_should_refresh"
            ].due_opportunity_count,
            "ordinary_refreshed_slot_count": mutations[
                "ordinary_candidate_refresh_mask"
            ].due_opportunity_count,
            "post_promotion_refreshed_slot_count": mutations[
                "post_promotion_candidate_refresh_mask"
            ].due_opportunity_count,
            "total_refreshed_slot_count": mutations[
                "candidate_refresh_mask"
            ].due_opportunity_count,
        },
        "cascade_losses": cascade_losses,
        "cascade_loss_definition": (
            "target-signature lost slots whose exact decision audit cause is "
            "cascade_dependency_refill"
        ),
        "target_admission_loss_end": target_lifecycle,
        "pre_recurrence_presence": [
            {
                "target": record["target"],
                "occurrence": record["occurrence"],
                "pre_recurrence_post_step": record["pre_recurrence_post_step"],
                "active_present": record["active_present"],
                "candidate_present": record["candidate_present"],
                "active_slot_count": record["active_slot_count"],
                "candidate_slot_count": record["candidate_slot_count"],
            }
            for record in pre_recurrence
        ],
        "target_retention": target_retention,
        "target_occupancy": {
            "post_update_state_count": geometry.total_steps,
            "per_target": occupancy,
            "coexistence": coexistence,
            "steps_by_distinct_active_target_count": coexistence[
                "steps_by_active_target_count"
            ],
            "maximum_distinct_active_target_count": coexistence[
                "maximum_active_target_count"
            ],
            "final_active_targets": coexistence["active_targets_at_end"],
        },
        "pre_recurrence_ranks": {
            "active_definition": (
                "best matching target slot among composed slots RAW_DIM:ACTIVE_SLOTS; "
                "tie-aware descending rank interval, with rank 1 highest"
            ),
            "candidate_definition": (
                "best matching target slot among all candidate slots; direct and "
                "novelty-augmented scores each use a tie-aware descending rank interval, "
                "with rank 1 highest"
            ),
            "records": pre_recurrence,
        },
        "cadence_integrity": {
            "diagnostic_partitions": {
                name: _opportunity_partition_record(partition)
                for name, partition in diagnostic.items()
            },
            "mutation_partitions": mutation_partition_records,
            "all_mutations_off_opportunity_count": (
                all_mutations_off_opportunity
            ),
            "curation_counts_close": count_closure["all_checked_counts_close"],
            "curation_count_closure": count_closure,
            "eventwise_curation_closure": eventwise_closure,
        },
        "identity_reacquisition_claimed": False,
    }


def logical_work_per_arm(geometry: FutureUtilityWorkGeometry) -> dict[str, object]:
    """Declare fixed shared-base logical work for one arm."""

    if type(geometry) is not FutureUtilityWorkGeometry:
        raise TypeError("geometry must be an exact FutureUtilityWorkGeometry")
    steps = geometry.steps
    active = geometry.active_slots
    candidates = geometry.candidate_slots
    heads = geometry.action_heads
    return {
        "learner_updates": steps,
        "curation_update_opportunities": steps // geometry.curation_interval,
        "behavior_active_feature_value_cells": steps * active,
        "learner_update_active_feature_value_cells": steps * active,
        "total_active_feature_value_cells": steps * active * 2,
        "learner_update_candidate_feature_value_cells": steps * candidates,
        "evaluator_full_q_dot_products": steps,
        "evaluator_raw_q_dot_products": steps,
        "learner_prediction_q_dot_products": steps,
        "full_and_raw_q_dot_products": steps * 2,
        "total_q_dot_products": steps * 3,
        "total_q_head_scalar_outputs": steps * heads * 3,
        "ranking_diagnostic_calls": steps + 1,
        "active_future_reduction_cells": steps * heads * active,
        "candidate_future_reduction_cells": steps * heads * candidates,
        "future_contribution_trace_cells": steps * heads * (active + candidates),
        "future_feature_energy_trace_cells": steps * (active + candidates),
        "persistent_candidate_active_correlation_cells": active * candidates,
        "candidate_active_correlation_statistical_accumulation_cells": 0,
        "candidate_active_correlation_reset_mask_cells": steps * active * candidates,
        "ranking_candidate_active_correlation_cells": (steps + 1) * active * candidates,
        "persistent_state_nbytes": control.compositional_control_state_nbytes_formula(
            active_slots=active,
            candidate_slots=candidates,
            action_heads=heads,
        ),
        "persistent_search_archive_entries": 0,
        "keys_stream_shapes_and_update_opportunities_matched": True,
        "behavioral_experience_matching_claimed": False,
        "compiled_flop_equivalence_claimed": False,
    }


def intervention_work_per_arm(
    geometry: FutureUtilityWorkGeometry,
    arm: FutureUtilityArmSpec,
) -> dict[str, int]:
    """Count cells conditional on the declared intervention."""

    if type(geometry) is not FutureUtilityWorkGeometry:
        raise TypeError("geometry must be an exact FutureUtilityWorkGeometry")
    if type(arm) is not FutureUtilityArmSpec:
        raise TypeError("arm must be an exact FutureUtilityArmSpec")
    active_cells = geometry.steps * geometry.active_slots
    candidate_cells = geometry.steps * geometry.candidate_slots
    normalized = arm.normalization == "uncertainty_age"
    return {
        "utility_mixture_cells": (
            0 if arm.mix == 0.0 else active_cells + candidate_cells
        ),
        "active_second_moment_cells": active_cells if normalized else 0,
        "candidate_second_moment_cells": candidate_cells if normalized else 0,
        "active_age_debias_cells": active_cells if normalized else 0,
        "candidate_age_debias_cells": candidate_cells if normalized else 0,
        "active_uncertainty_normalization_cells": active_cells if normalized else 0,
        "candidate_uncertainty_normalization_cells": (
            candidate_cells if normalized else 0
        ),
    }


def work_resource_contract(
    geometry: FutureUtilityWorkGeometry,
    arms: Sequence[FutureUtilityArmSpec],
) -> dict[str, object]:
    """Separate matched base work from intentionally unequal interventions."""

    if type(arms) is not tuple or not arms:
        raise TypeError("arms must be a non-empty exact tuple")
    if any(type(arm) is not FutureUtilityArmSpec for arm in arms):
        raise TypeError("arms must contain exact FutureUtilityArmSpec values")
    if len({arm.name for arm in arms}) != len(arms):
        raise ValueError("arm names must be unique")
    shared = logical_work_per_arm(geometry)
    interventions = {
        arm.name: intervention_work_per_arm(geometry, arm) for arm in arms
    }
    if len({_canonical_json(value) for value in interventions.values()}) == 1:
        raise RuntimeError("calibration interventions unexpectedly have equal named work")
    return {
        "selected_arm_count": len(arms),
        "per_arm_shared_base": shared,
        "intervention_specific_per_arm": interventions,
        "panel_learner_updates": geometry.steps * len(arms),
        "panel_curation_update_opportunities": (
            geometry.steps // geometry.curation_interval
        )
        * len(arms),
        "aggregate_arm_state_byte_equivalent": cast(
            int, shared["persistent_state_nbytes"]
        )
        * len(arms),
        "aggregate_arm_state_byte_equivalent_is_peak_memory": False,
        "accounting_scope": RESOURCE_ACCOUNTING_SCOPE,
        "shared_base_logical_work_matched": True,
        "stream_shapes_and_update_opportunities_matched": True,
        "intervention_specific_logical_work_matched": False,
        "total_named_logical_work_equivalence_claimed": False,
        "behavior_dependent_branch_work_equivalence_claimed": False,
        "persistent_shapes_matched": True,
        "source_array_bytes_included": False,
        "full_scan_telemetry_bytes_included": False,
        "compiler_workspace_bytes_included": False,
        "compiled_flop_equivalence_claimed": False,
        "behavioral_experience_matching_claimed": False,
    }


def _differing_fields(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> tuple[str, ...]:
    if tuple(left) != tuple(right):
        raise RuntimeError("arm learner-config schemas differ")
    return tuple(
        field
        for field in left
        if _canonical_json(left[field]) != _canonical_json(right[field])
    )


def validate_future_utility_arm_contrasts(
    historical_base: Mapping[str, Any],
    arms: Sequence[FutureUtilityArmSpec],
    configs: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    """Require the exact three-field intervention and four isolated contrasts."""

    if type(arms) is not tuple or len(arms) != 5:
        raise TypeError("arms must be an exact five-element tuple")
    names = tuple(arm.name for arm in arms)
    if len(set(names)) != len(names) or tuple(configs) != names:
        raise RuntimeError("arm configs are not unique and in declared order")
    first_fields = tuple(configs[names[0]])
    if any(tuple(configs[name]) != first_fields for name in names[1:]):
        raise RuntimeError("learner-config schemas differ")
    varying: dict[str, dict[str, object]] = {}
    for field in first_fields:
        values = {name: cast(object, configs[name][field]) for name in names}
        if len({_canonical_json(value) for value in values.values()}) > 1:
            varying[field] = values
    if tuple(varying) != INTERVENTION_FIELDS:
        raise RuntimeError("arms differ outside the exact three-field intervention")

    historical_expected: dict[str, object] = {
        "candidate_scoring_mode": "energy_novelty",
        "candidate_novelty_admission_bonus": 1.0,
        "future_utility_trace_mode": "marginal",
        "future_utility_rare_task_power": 0.0,
    }
    corrected_expected: dict[str, object] = {
        "candidate_scoring_mode": "legacy",
        "candidate_novelty_admission_bonus": 0.0,
        "future_utility_trace_mode": "contribution",
        "future_utility_rare_task_power": 0.0,
    }
    if tuple(historical_base) != first_fields:
        raise RuntimeError("corrected and historical config schemas differ")
    departures: dict[str, dict[str, object]] = {}
    for field in COMMON_DEPARTURE_FIELDS:
        if historical_base[field] != historical_expected[field]:
            raise RuntimeError(f"historical common-base field {field} drifted")
        corrected = configs[names[0]][field]
        if corrected != corrected_expected[field] or any(
            configs[name][field] != corrected for name in names
        ):
            raise RuntimeError(f"corrected common-base field {field} is not paired")
        departures[field] = {
            "historical_common_base": historical_base[field],
            "corrected_common_base": corrected,
            "is_value_departure": historical_base[field] != corrected,
        }

    declared = {*COMMON_DEPARTURE_FIELDS, *INTERVENTION_FIELDS}
    for arm in arms:
        config = configs[arm.name]
        undeclared = tuple(
            field
            for field in first_fields
            if field not in declared
            and _canonical_json(config[field]) != _canonical_json(historical_base[field])
        )
        if undeclared:
            raise RuntimeError(f"arm {arm.name} changes undeclared fields: {undeclared}")
        actual = (
            config["future_utility_mix"],
            config["future_utility_trace_decay"],
            config["future_utility_normalization"],
        )
        expected = (arm.mix, arm.trace_decay, arm.normalization)
        if _canonical_json(actual) != _canonical_json(expected):
            raise RuntimeError(f"arm {arm.name} does not match its mechanism tuple")
        if (
            config["future_utility_normalization_decay"] != NORMALIZATION_DECAY
            or config["future_utility_rare_task_power"] != 0.0
        ):
            raise RuntimeError("normalization decay or rare-task power drifted")

    contrasts = {
        "current_to_future": _differing_fields(configs[names[0]], configs[names[1]]),
        "calibrated_to_future": _differing_fields(
            configs[names[2]], configs[names[1]]
        ),
        "normalized_to_future": _differing_fields(
            configs[names[3]], configs[names[1]]
        ),
        "horizon_to_normalized": _differing_fields(
            configs[names[4]], configs[names[3]]
        ),
    }
    if contrasts != {
        "current_to_future": ("future_utility_mix",),
        "calibrated_to_future": ("future_utility_mix",),
        "normalized_to_future": ("future_utility_normalization",),
        "horizon_to_normalized": ("future_utility_trace_decay",),
    }:
        raise RuntimeError("the four calibration contrasts are not isolated")
    return varying, departures


__all__ = [
    "COMMON_DEPARTURE_FIELDS",
    "DEVELOPMENT_ONLY",
    "EVIDENCE_AUTHORIZED",
    "INTERVENTION_FIELDS",
    "OUTPUT_WRITES_ALLOWED",
    "PANEL_EXECUTION_AUTHORIZED",
    "PRIMARY_ENDPOINT_NAMES",
    "REQUIRED_CADENCE_MUTATION_MASK_NAMES",
    "ROOT_ISSUANCE_AUTHORIZED",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "FutureUtilityArmSpec",
    "FutureUtilityEndpointGeometry",
    "FutureUtilityWorkGeometry",
    "OpportunityPartition",
    "active_bank_descending_rank",
    "arm_definition",
    "build_future_utility_primary_endpoints",
    "build_future_utility_learner_config",
    "candidate_bank_descending_rank",
    "future_utility_cadence_audit_from_events",
    "intervention_work_per_arm",
    "logical_work_per_arm",
    "pre_recurrence_records",
    "validate_future_utility_arm_contrasts",
    "validate_future_utility_curation_count_closure",
    "validate_future_utility_experience_semantics",
    "validate_future_utility_eventwise_curation_semantics",
    "validate_future_utility_trace_shapes",
    "work_resource_contract",
]
